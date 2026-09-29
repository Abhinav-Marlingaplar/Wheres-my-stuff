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
from db import init_db, reset_db, insert_observation
from zones import ZoneManager
from reid import GlobalTracker
from memory import ObservationMemory


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
    # Reset DB and the visual-memory index for a fresh run
    reset_db()
    memory = ObservationMemory()
    memory.reset()

    # Initialise YOLO model (nano for CPU speed)
    model = YOLO("yolov8n.pt")

    # Create output directories
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(crops_dir, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Error opening video file {video_path}")

    img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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

    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        timestamp = str(datetime.datetime.now())

        # ── Detection + Tracking ─────────────────────────────────────────
        results = model.track(frame, persist=True, tracker="bytetrack.yaml",
                              verbose=False)

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

        if (results[0].boxes is not None
                and results[0].boxes.id is not None):
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            confs = results[0].boxes.conf.cpu().numpy()

            for box, track_id, class_id, conf in zip(
                    boxes, track_ids, class_ids, confs):
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                class_name = model.names[class_id]

                # Zone classification (polygonal)
                zone = zone_manager.classify_point(cx, cy)

                # ReID — get stable global track ID
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
                        crop_path = os.path.join(
                            crops_dir,
                            f"crop_f{frame_number:04d}_g{global_id}.jpg")
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
                label = (f"{class_name} | GID:{global_id} "
                         f"| {conf:.2f} | {zone}")
                # Colour by global ID for visual consistency
                color = _id_color(global_id)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
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

        if progress_callback:
            progress_callback(frame_number, total_frames)

    cap.release()
    out.release()
    memory.save()
    os.makedirs(memory.directory, exist_ok=True)
    with open(os.path.join(memory.directory, "run_stats.json"), "w",
              encoding="utf-8") as f:
        json.dump({
            "backend": global_tracker.backend,
            "stitches": global_tracker.stitch_count,
        }, f, indent=2)
    return output_video_path


def _id_color(track_id: int) -> tuple:
    """Deterministic BGR colour from a track ID (for consistent box colours)."""
    import hashlib
    h = hashlib.md5(str(track_id).encode()).digest()
    return (h[0], h[1], h[2])


# ── CLI entrypoint ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        process_video(sys.argv[1])
    else:
        print("Usage: python pipeline.py <video_path>")
