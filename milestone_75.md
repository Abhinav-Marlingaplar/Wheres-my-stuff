# Where Is My Stuff? — 75% Milestone

**Project:** Where Is My Stuff?  
**Team:** 13 — Saksham Saklani, Niranjan Alase, Abhinav Marlingaplar, Kedar Vaishnav

This branch takes the Task 5 system (about 50%) to about **75%** of the proposed pipeline. The last 25% is listed at the end and is not in this branch.

## What this increment adds

1. **Zone drawing.** On the video tab, the first frame is a polygon canvas. Named zones are saved to `zones.json`, which the tracker and `/zones` already use.
2. **CLIP re-identification.** A new ByteTrack ID is matched to a recently lost track of the same class with a ViT-B/32 embedding. If the weights do not load, matching uses the HSV histogram.
3. **Visual memory.** Keyframe crop embeddings are stored in a FAISS index with class, zone, time, frame, and global track id. A question that misses the class parser is searched with a CLIP text embedding.
4. **Evaluation harness.** `evaluate.py` reports detection counts, zone dwell, stitched ByteTrack IDs, and the Task 5 sample questions. It does not compute MOTA, IDF1, or HOTA.

## How to produce the numbers

After the dependencies in `requirements.txt` are installed:

```powershell
python evaluate.py path\to\room.mp4
```

With no argument, the script scores whatever is already in `observations.db` and writes `evaluation_results.md`.

Stitched IDs are ByteTrack IDs beyond the first one that share a `global_track_id`. Zone dwell is the share of a class's detections that landed in each polygon. A sample question is `resolved` when the class parser hits a logged class, `visual match` when FAISS answers, and `not observed` when the item was never seen.

## Still deferred (last 25%)

- Fine-tune YOLOv8n for keys, wallets, spectacles, and earphones.
- Live webcam and RTSP input with a sliding observation window.
- MOTA, IDF1, HOTA, and detection mAP once frame-level ground truth exists.
