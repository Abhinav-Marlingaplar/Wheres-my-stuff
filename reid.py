"""
reid.py — Visual Re-Identification & Track Stitching

When ByteTrack loses an object (occlusion, leaving frame) and re-detects it
later with a *new* tracking ID, this module matches it back to its previous
identity.

Appearance features, in order:
  1. CLIP ViT-B/32 image embedding of the object crop (cosine distance).
  2. If CLIP cannot be loaded, a normalised HSV colour histogram
     (3 × 16 bins) compared with Bhattacharyya distance.

Both paths also use the bounding-box aspect ratio as a small penalty.
A gallery of recently lost tracks is kept per object class. A new ByteTrack
ID is stitched to an old global_track_id only when the best lost track of
the same class is close enough.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


# ── Configuration ────────────────────────────────────────────────────────────
HIST_BINS = 16                 # per HSV channel
SIMILARITY_THRESHOLD = 0.55   # Bhattacharyya distance below this ⇒ match
CLIP_DISTANCE_THRESHOLD = 0.28  # 1 - cosine; below this ⇒ match (~0.72 similarity)
EMBED_REFRESH_FRAMES = 10     # recompute CLIP on an active track at this interval
MAX_DISAPPEARED_FRAMES = 150  # frames before a lost track is evicted
MAX_GALLERY_SIZE = 200        # hard cap on gallery to bound memory
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distance in ``[0, 2]`` for two vectors. ``0`` means the same direction."""
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    na = float(np.linalg.norm(a)) + 1e-8
    nb = float(np.linalg.norm(b)) + 1e-8
    return 1.0 - float(np.dot(a, b) / (na * nb))


def _crop(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


class ClipEmbedder:
    """CLIP ViT-B/32 image encoder. Stays disabled if the weights cannot load."""

    def __init__(self):
        self.available = False
        self.error: Optional[str] = None
        self.model = None
        self.preprocess = None
        self._torch = None
        self._load()

    def _load(self):
        try:
            import torch
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
            )
            model.eval()
            self.model = model
            self.preprocess = preprocess
            self._torch = torch
            self._tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
            self.available = True
        except Exception as exc:
            self.available = False
            self.error = str(exc)
            self._tokenizer = None

    def embed(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> Optional[np.ndarray]:
        """L2-normalised CLIP embedding of the crop, or None."""
        if not self.available:
            return None
        crop = _crop(frame, bbox)
        if crop is None:
            return None
        try:
            from PIL import Image
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            image = self.preprocess(Image.fromarray(rgb)).unsqueeze(0)
            torch = self._torch
            with torch.no_grad():
                feat = self.model.encode_image(image)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            return feat.squeeze(0).cpu().numpy().astype(np.float32)
        except Exception:
            return None

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """L2-normalised CLIP text embedding, or None if the model is unavailable."""
        if not self.available or self._tokenizer is None:
            return None
        try:
            tokens = self._tokenizer([text])
            torch = self._torch
            with torch.no_grad():
                feat = self.model.encode_text(tokens)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            return feat.squeeze(0).cpu().numpy().astype(np.float32)
        except Exception:
            return None


_SHARED_EMBEDDER: Optional[ClipEmbedder] = None


def get_clip_embedder() -> ClipEmbedder:
    """One CLIP encoder per process, shared by tracking and text search."""
    global _SHARED_EMBEDDER
    if _SHARED_EMBEDDER is None:
        _SHARED_EMBEDDER = ClipEmbedder()
    return _SHARED_EMBEDDER


@dataclass
class TrackDescriptor:
    """Appearance descriptor for a single tracked object."""
    global_track_id: int
    object_class: str
    histogram: np.ndarray          # normalised HSV histogram (48-dim)
    aspect_ratio: float
    last_seen_frame: int
    bbox: tuple[int, int, int, int]   # (x1, y1, x2, y2)
    embedding: Optional[np.ndarray] = None
    last_embed_frame: int = -1


class GlobalTracker:
    """Assigns persistent global IDs by stitching fragmented ByteTrack IDs."""

    def __init__(self, embedder="auto"):
        """
        Parameters
        ----------
        embedder :
            ``"auto"`` loads CLIP and falls back to HSV.
            ``None`` forces the HSV matcher.
            Any other object with ``.available`` and ``.embed(frame, bbox)``
            is used as the encoder (for tests).
        """
        if embedder == "auto":
            self.embedder = get_clip_embedder()
        else:
            self.embedder = embedder
        self.backend = (
            "clip" if self.embedder is not None and getattr(self.embedder, "available", False)
            else "hsv"
        )
        self._next_global_id: int = 1
        self.stitch_count: int = 0
        # byte_track_id  →  global_track_id   (for currently active tracks)
        self._active_map: dict[int, int] = {}
        # global_track_id  →  TrackDescriptor (gallery of all known tracks)
        self._gallery: dict[int, TrackDescriptor] = {}
        # byte_track_id set from previous frame (to detect new / lost IDs)
        self._prev_byte_ids: set[int] = set()

    # ── Public API ───────────────────────────────────────────────────────────
    def update(
        self,
        frame: np.ndarray,
        byte_track_id: int,
        object_class: str,
        bbox: tuple[int, int, int, int],
        frame_number: int,
    ) -> int:
        """
        Given a ByteTrack detection, return its global_track_id.

        If this byte_track_id was seen last frame, the existing mapping is
        reused.  If it is *new*, we try to match it against recently lost
        tracks of the same class via appearance similarity.
        """
        # Fast path — already mapped
        if byte_track_id in self._active_map:
            gid = self._active_map[byte_track_id]
            # Refresh gallery descriptor
            desc = self._compute_descriptor(
                frame, gid, object_class, bbox, frame_number,
                previous=self._gallery.get(gid),
            )
            self._gallery[gid] = desc
            return gid

        # New ByteTrack ID — try to match against disappeared tracks
        descriptor = self._compute_descriptor(
            frame, -1, object_class, bbox, frame_number
        )
        matched_gid = self._match_against_gallery(descriptor, frame_number)

        if matched_gid is not None:
            gid = matched_gid
            self.stitch_count += 1
        else:
            gid = self._next_global_id
            self._next_global_id += 1

        self._active_map[byte_track_id] = gid
        descriptor.global_track_id = gid
        self._gallery[gid] = descriptor
        return gid

    def embedding_for(self, global_track_id: int) -> Optional[np.ndarray]:
        """Latest CLIP embedding stored for this global id, if any."""
        desc = self._gallery.get(global_track_id)
        if desc is None:
            return None
        return desc.embedding

    def end_of_frame(self, current_byte_ids: set[int], frame_number: int):
        """
        Call once per frame after all detections have been processed.
        Removes mappings for ByteTrack IDs that disappeared this frame
        (the gallery entry is kept for potential future re-identification).
        """
        lost = set(self._active_map.keys()) - current_byte_ids
        for bid in lost:
            del self._active_map[bid]

        # Evict very old gallery entries to bound memory
        to_remove = []
        for gid, desc in self._gallery.items():
            if (frame_number - desc.last_seen_frame) > MAX_DISAPPEARED_FRAMES:
                to_remove.append(gid)
        for gid in to_remove:
            del self._gallery[gid]

        # Hard cap
        if len(self._gallery) > MAX_GALLERY_SIZE:
            # keep only the most recently seen
            sorted_items = sorted(self._gallery.items(),
                                  key=lambda kv: kv[1].last_seen_frame, reverse=True)
            self._gallery = dict(sorted_items[:MAX_GALLERY_SIZE])

        self._prev_byte_ids = current_byte_ids.copy()

    # ── Internals ────────────────────────────────────────────────────────────
    @staticmethod
    def _extract_histogram(frame: np.ndarray,
                           bbox: tuple[int, int, int, int]) -> np.ndarray:
        """Compute a normalised HSV colour histogram for the cropped region."""
        crop = _crop(frame, bbox)
        if crop is None:
            return np.zeros(HIST_BINS * 3, dtype=np.float32)

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        hists = []
        for ch in range(3):
            hist = cv2.calcHist([hsv], [ch], None, [HIST_BINS], [0, 256])
            cv2.normalize(hist, hist)
            hists.append(hist.flatten())
        return np.concatenate(hists).astype(np.float32)

    def _compute_descriptor(
        self,
        frame: np.ndarray,
        gid: int,
        object_class: str,
        bbox: tuple[int, int, int, int],
        frame_number: int,
        previous: Optional[TrackDescriptor] = None,
    ) -> TrackDescriptor:
        x1, y1, x2, y2 = bbox
        bw = max(x2 - x1, 1)
        bh = max(y2 - y1, 1)
        hist = self._extract_histogram(frame, bbox)
        embedding = None
        last_embed_frame = -1
        if self.embedder is not None and getattr(self.embedder, "available", False):
            stale = (
                previous is None
                or previous.embedding is None
                or frame_number - previous.last_embed_frame >= EMBED_REFRESH_FRAMES
            )
            if stale:
                embedding = self.embedder.embed(frame, bbox)
                if embedding is not None:
                    last_embed_frame = frame_number
            else:
                embedding = previous.embedding
                last_embed_frame = previous.last_embed_frame
        return TrackDescriptor(
            global_track_id=gid,
            object_class=object_class,
            histogram=hist,
            aspect_ratio=bw / bh,
            last_seen_frame=frame_number,
            bbox=bbox,
            embedding=embedding,
            last_embed_frame=last_embed_frame,
        )

    def _match_against_gallery(
        self,
        query: TrackDescriptor,
        current_frame: int,
    ) -> Optional[int]:
        """
        Find the best-matching recently-disappeared track of the same class.
        Returns the global_track_id if a good match is found, else None.
        """
        # Only consider tracks that are NOT currently active
        active_gids = set(self._active_map.values())
        use_clip = query.embedding is not None and any(
            desc.embedding is not None
            and desc.object_class == query.object_class
            and gid not in active_gids
            for gid, desc in self._gallery.items()
        )
        best_gid: Optional[int] = None
        best_distance = float("inf")
        threshold = CLIP_DISTANCE_THRESHOLD if use_clip else SIMILARITY_THRESHOLD

        for gid, gallery_desc in self._gallery.items():
            # Must be same object class
            if gallery_desc.object_class != query.object_class:
                continue
            # Must not be currently active (we want *lost* tracks)
            if gid in active_gids:
                continue
            # Must have disappeared recently
            frames_gone = current_frame - gallery_desc.last_seen_frame
            if frames_gone > MAX_DISAPPEARED_FRAMES or frames_gone <= 0:
                continue

            ar_diff = abs(query.aspect_ratio - gallery_desc.aspect_ratio)
            if use_clip:
                if gallery_desc.embedding is None:
                    continue
                dist = cosine_distance(query.embedding, gallery_desc.embedding)
                dist += 0.05 * ar_diff
            else:
                # Bhattacharyya distance (lower = more similar, 0 = identical)
                dist = cv2.compareHist(
                    query.histogram, gallery_desc.histogram,
                    cv2.HISTCMP_BHATTACHARYYA,
                )
                dist += 0.1 * ar_diff

            if dist < best_distance:
                best_distance = dist
                best_gid = gid

        if best_gid is not None and best_distance < threshold:
            return best_gid
        return None


# ── Quick self-test ──────────────────────────────────────────────────────────
class _ColorEmbedder:
    """Test embedder: the crop's mean BGR colour, L2-normalised."""

    available = True

    def embed(self, frame, bbox):
        crop = _crop(frame, bbox)
        if crop is None:
            return None
        vec = crop.mean(axis=(0, 1)).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-8)


def _solid(color, w=640, h=480):
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = color
    return frame


if __name__ == "__main__":
    red = _solid((0, 0, 220))
    blue = _solid((220, 0, 0))

    hsv = GlobalTracker(embedder=None)
    assert hsv.backend == "hsv"
    gid_red = hsv.update(red, 1, "bottle", (100, 100, 180, 260), 1)
    hsv.update(blue, 2, "bottle", (300, 100, 380, 260), 1)
    hsv.end_of_frame({1, 2}, 1)
    hsv.end_of_frame({2}, 2)  # red bottle lost
    stitched = hsv.update(red, 9, "bottle", (110, 110, 190, 270), 3)
    assert stitched == gid_red, stitched
    assert hsv.stitch_count == 1, hsv.stitch_count
    print(f"HSV stitch ok: lost id came back as gid {stitched}")

    clip = GlobalTracker(embedder=_ColorEmbedder())
    assert clip.backend == "clip"
    gid_red = clip.update(red, 1, "cup", (40, 40, 120, 160), 1)
    clip.update(blue, 2, "cup", (200, 40, 280, 160), 1)
    clip.end_of_frame({1, 2}, 1)
    clip.end_of_frame({2}, 2)
    stitched = clip.update(red, 7, "cup", (50, 50, 130, 170), 4)
    other = clip.update(blue, 8, "cup", (400, 40, 480, 160), 4)
    assert stitched == gid_red, stitched
    assert other != gid_red
    print(f"CLIP stitch ok: red gid {stitched}, new blue gid {other}")

    live = ClipEmbedder()
    print("CLIP weights:", "loaded" if live.available else f"unavailable ({live.error})")
