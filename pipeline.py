"""
pipeline.py — Core Detection + Tracking + Classification Pipeline

Task 5 upgrades:
  • Polygonal zone classification via ZoneManager (replaces static quadrants)
  • Visual ReID / global track stitching via GlobalTracker (CLIP, HSV fallback)
  • Object crop extraction (for UI thumbnails and ReID gallery)
  • Zone overlay drawing on annotated output video
  • Extended DB logging (global_track_id, crop_path, frame_path)
  • FAISS index of keyframe crop embeddings (memory.py)
"""

import cv2
from ultralytics import YOLO
import datetime
import os
import json
from db import (
    init_db,
    reset_db,
    insert_observation,
    max_frame_number,
    prune_older_than,
    window_cutoff,
)
from zones import ZoneManager
from reid import GlobalTracker
from memory import ObservationMemory


def capture_source(source):
    """OpenCV source: an integer webcam index, or a file path / RTSP URL."""
    if isinstance(source, int):
        return source
    text = str(source).strip()
    if text.isdigit():
        return int(text)
    return text


def is_live_source(source) -> bool:
    """True for a webcam index or a network camera URL."""
    if isinstance(source, int):
        return True
    text = str(source).strip().lower()
    return text.isdigit() or text.startswith(
        ("rtsp://", "rtsps://", "http://", "https://")
    )

PROMPTS = {
    "keys": "keys",
    "wallet": "wallet",
    "glasses": "glasses",
    "earphones": "earphones",
    "watch": "watch",
    "bottle": "bottle",
    "tube": "cream",
    "bag": "bag",
}
DISTRACTORS = []

DETECTION_CONFIDENCE = 0.08
DETECTION_IMGSZ = 1024


def load_detector():
    """Open-vocabulary detector: no training, classes are set from text."""
    model = YOLO("yolov8s-worldv2.pt")
    model.set_classes(list(PROMPTS) + DISTRACTORS)
    return model


def _resolve_class_name(raw_name: str) -> str | None:
    """Return a normalized class label, or None when the class is ignored."""
    if raw_name in DISTRACTORS:
        return None
    return PROMPTS.get(raw_name, raw_name)

def process_video(video_path, output_video_path="output.mp4",
                  frames_dir="frames", crops_dir="crops",
                  zones_config=None, progress_callback=None):
    """
    Run the full detection → tracking → classification → logging pipeline.

    Parameters
    ----------
    video_path : str
        Path to the input video file.
    output_video_path : str
        Where to write the annotated video.
    frames_dir : str
        Directory for saved keyframe images.
    crops_dir : str
        Directory for cropped object thumbnails.
    zones_config : str | None
        Path to a zones.json config.  None → default quadrant zones.
    progress_callback : callable | None
        Called as callback(current_frame, total_frames).

    Returns
    -------
    str : path to the output video
    """
    return _run_pipeline(
        video_path,
        output_video_path=output_video_path,
        frames_dir=frames_dir,
        crops_dir=crops_dir,
        zones_config=zones_config,
        progress_callback=progress_callback,
    )


LIVE_WINDOW_SECONDS = 30 * 60


def process_stream(source, output_video_path="output_live.mp4",
                   frames_dir="frames", crops_dir="crops",
                   zones_config=None, progress_callback=None,
                   frame_callback=None, max_frames=None,
                   window_seconds=LIVE_WINDOW_SECONDS):
    """
    Run the same detection, tracking, zone, and identity pipeline on a
    live source: a webcam index (``0``) or an RTSP/HTTP camera URL.

    Sightings are appended to the existing log. Rows older than
    ``window_seconds`` (30 minutes by default) are dropped.
    ``frame_callback(annotated_frame, frame_number)`` may return True to stop.
    ``max_frames`` stops after that many frames.
    """
    return _run_pipeline(
        source,
        output_video_path=output_video_path,
        frames_dir=frames_dir,
        crops_dir=crops_dir,
        zones_config=zones_config,
        progress_callback=progress_callback,
        frame_callback=frame_callback,
        max_frames=max_frames,
        reset=False,
        window_seconds=window_seconds,
    )


def _run_pipeline(source, output_video_path="output.mp4",
                  frames_dir="frames", crops_dir="crops",
                  zones_config=None, progress_callback=None,
                  frame_callback=None, max_frames=None,
                  reset=True, window_seconds=None):
    # A file run replaces the log. A live run keeps it and prunes by age.
    memory = ObservationMemory()
    if reset:
        reset_db()
        init_db()
        memory.reset()
    else:
        init_db()
        memory.load()
        if window_seconds:
            _prune_window(memory, window_seconds)

    # Initialise open-vocabulary YOLO detector.
    model = load_detector()

    # Create output directories
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(crops_dir, exist_ok=True)

    # Open a file, a webcam index, or an RTSP/HTTP URL
    cap = cv2.VideoCapture(capture_source(source))
    if not cap.isOpened():
        raise Exception(f"Error opening video source {source}")

    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise Exception(f"No frames from video source {source}")

    img_height, img_width = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps == 0 or fps != fps:  # NaN check
        fps = 30.0

    # Initialise zone manager (uses video resolution for default zones)
    zone_manager = ZoneManager(config_path=zones_config,
                               img_width=img_width, img_height=img_height)

    # Initialise ReID tracker
    global_tracker = GlobalTracker()

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps,
                          (img_width, img_height))

    frame_number = 0 if reset else max_frame_number()
    processed = 0

    while frame is not None:
        frame_number += 1
        processed += 1
        timestamp = str(datetime.datetime.now())

        # ── Detection + Tracking ─────────────────────────────────────────
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=DETECTION_CONFIDENCE,
            imgsz=DETECTION_IMGSZ,
            verbose=False,
        )

        annotated_frame = frame.copy()

        # Draw zone overlays first (so they appear behind bounding boxes)
        zone_manager.draw_zones(annotated_frame, alpha=0.15)

        # Determine keyframe path (saved every 30 frames or first frame)
        frame_path = None
        if frame_number % 30 == 0 or frame_number == 1:
            frame_path = os.path.join(frames_dir,
                                      f"frame_{frame_number:04d}.jpg")
            cv2.imwrite(frame_path, frame)

        # ── Process detections ───────────────────────────────────────────
        current_byte_ids = set()

        boxes_obj = results[0].boxes
        if boxes_obj is not None and len(boxes_obj) > 0:
            boxes = boxes_obj.xyxy.cpu().numpy()
            class_ids = boxes_obj.cls.int().cpu().tolist()
            confs = boxes_obj.conf.cpu().numpy()
            if boxes_obj.id is not None:
                track_ids = boxes_obj.id.int().cpu().tolist()
            else:
                track_ids = [-1] * len(boxes)

            for det_idx, (box, track_id, class_id, conf) in enumerate(
                    zip(boxes, track_ids, class_ids, confs)):
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                raw_name = model.names[class_id]
                class_name = _resolve_class_name(raw_name)
                if class_name is None:
                    continue

                # Zone classification (polygonal)
                zone = zone_manager.classify_point(cx, cy)

                global_id = None
                if track_id >= 0:
                    # ReID — get stable global track ID for tracked detections.
                    global_id = global_tracker.update(
                        frame, byte_track_id=track_id,
                        object_class=class_name,
                        bbox=(x1, y1, x2, y2),
                        frame_number=frame_number,
                    )
                    current_byte_ids.add(track_id)

                # Save object crop (every 30 frames to avoid disk flood)
                crop_path = None
                if frame_number % 30 == 0 or frame_number == 1:
                    crop_img = frame[max(0, y1):y2, max(0, x1):x2]
                    if crop_img.size > 0:
                        crop_id = (
                            f"g{global_id}" if global_id is not None
                            else f"u{frame_number:04d}_{det_idx:02d}"
                        )
                        crop_path = os.path.join(
                            crops_dir,
                            f"crop_f{frame_number:04d}_{crop_id}.jpg")
                        cv2.imwrite(crop_path, crop_img)

                # Store observation
                box_str = json.dumps(
                    {"x1": x1, "y1": y1, "x2": x2, "y2": y2})
                insert_observation(
                    timestamp, frame_number, class_name, track_id,
                    float(conf), box_str, zone,
                    global_track_id=global_id,
                    crop_path=crop_path,
                    frame_path=frame_path,
                )

                # Index the keyframe crop so text queries can search it later
                embedding = None
                if global_id is not None:
                    embedding = global_tracker.embedding_for(global_id)
                if crop_path and embedding is not None:
                    memory.add(embedding, {
                        "object_class": class_name,
                        "zone": zone,
                        "timestamp": timestamp,
                        "frame_number": frame_number,
                        "global_track_id": global_id,
                        "crop_path": crop_path,
                        "frame_path": frame_path,
                        "confidence": float(conf),
                    })

                # ── Draw on frame ────────────────────────────────────────
                if global_id is not None:
                    label = (f"{class_name} | GID:{global_id} "
                             f"| {conf:.2f} | {zone}")
                    # Colour by global ID for visual consistency.
                    color = _id_color(global_id)
                    thickness = 2
                else:
                    label = (f"{class_name} | UNTRACKED "
                             f"| {conf:.2f} | {zone}")
                    color = (0, 200, 200)
                    thickness = 1
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                # Background for text readability
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(annotated_frame,
                              (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
                cv2.putText(annotated_frame, label,
                            (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 1)

        # End-of-frame bookkeeping for ReID
        global_tracker.end_of_frame(current_byte_ids, frame_number)

        out.write(annotated_frame)

        if frame_number % 20 == 0:
            print(f"frame {frame_number}/{total_frames}", flush=True)

        if progress_callback:
            progress_callback(frame_number, total_frames)

        if window_seconds and frame_number % 30 == 0:
            _prune_window(memory, window_seconds)

        if frame_callback is not None and frame_callback(annotated_frame, frame_number):
            break
        if max_frames is not None and processed >= max_frames:
            break

        ok, frame = cap.read()
        if not ok:
            frame = None

    cap.release()
    out.release()
    if window_seconds:
        _prune_window(memory, window_seconds)
    memory.save()
    os.makedirs(memory.directory, exist_ok=True)
    with open(os.path.join(memory.directory, "run_stats.json"), "w",
              encoding="utf-8") as f:
        json.dump({
            "backend": global_tracker.backend,
            "stitches": global_tracker.stitch_count,
        }, f, indent=2)
    return output_video_path


def _prune_window(memory: ObservationMemory, window_seconds: float):
    """Drop database rows and visual-memory crops older than the window."""
    prune_older_than(window_seconds)
    memory.drop_older_than(window_cutoff(window_seconds))


def _id_color(track_id: int) -> tuple:
    """Deterministic BGR colour from a track ID (for consistent box colours)."""
    import hashlib
    h = hashlib.md5(str(track_id).encode()).digest()
    return (h[0], h[1], h[2])


# ── CLI entrypoint ───────────────────────────────────────────────────────────
def _preview_and_stop(annotated_frame, frame_number):
    """Show the live view. Press q to stop."""
    cv2.imshow("Where Is My Stuff?  (press q to stop)", annotated_frame)
    return (cv2.waitKey(1) & 0xFF) == ord("q")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <video_path | webcam_index | rtsp_url>")
    elif is_live_source(sys.argv[1]):
        process_stream(sys.argv[1], frame_callback=_preview_and_stop)
        cv2.destroyAllWindows()
    else:
        process_video(sys.argv[1])
