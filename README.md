# Where Is My Stuff?

A computer vision system that answers the question *"Where did I last leave this object?"*

It watches a room through a video file, webcam or RTSP camera, detects and tracks the objects it sees, records which zone of the scene each object is in, and lets you ask questions in plain English such as *"Where are my keys?"*.

Developed for **CSE411 Computer Vision** by **Team 13**:

| Name | Roll Number |
| :--- | :--- |
| Saksham Saklani | 2023BCD0049 |
| Niranjan Alase | 2023BCD0055 |
| Abhinav Marlingaplar | 2023BCD0013 |
| Kedar Vaishnav | 2023BCS0163 |

The full project report is in [`Where_Is_My_Stuff_Final_Report.pdf`](Where_Is_My_Stuff_Final_Report.pdf).

---

## How It Works

For every video frame the pipeline runs these steps:

1. **Detect** objects with YOLO-World, an open-vocabulary detector.
2. **Track** them across frames with ByteTrack, which gives each object a track ID.
3. **Re-identify** objects that were lost and reappear, so they keep one stable global ID.
4. **Locate** each object by checking which polygon zone contains the centre of its box.
5. **Store** the sighting (time, frame, class, IDs, confidence, zone, image paths) in a SQLite database.
6. **Answer** questions by searching that database, with an image search as a fallback.

---

## Architecture

| Component | File | What it does |
| :--- | :--- | :--- |
| Detection and tracking | `pipeline.py` | Runs YOLO-World (`yolov8s-worldv2.pt`) with ByteTrack. Classes are given as text prompts, so no training is needed. Handles video files, webcams and RTSP streams. |
| Re-identification | `reid.py` | `GlobalTracker` matches a new ByteTrack ID to a recently lost track of the same class using CLIP ViT-B/32 embeddings and cosine distance. It falls back to a 48-bin HSV histogram if the CLIP weights cannot be loaded. |
| Zones | `zones.py` | `ZoneManager` classifies an object's position with OpenCV's point-in-polygon test (`cv2.pointPolygonTest`). Zones are polygons saved as JSON. |
| Visual memory | `memory.py` | Stores CLIP embeddings of keyframe object crops in a FAISS index (folder `memory/`) so questions can be matched against images. |
| Query engine | `query_engine.py` | Turns a question into an object class using aliases and fuzzy matching, then returns the latest zone and movement history. If no class matches, the question is embedded with CLIP and searched against the FAISS index. |
| Storage | `db.py` | SQLite database `observations.db` holding timestamps, frame numbers, classes, ByteTrack IDs, global track IDs, confidence, zones, crop paths and keyframe paths. |
| Backend API | `api.py` | FastAPI service for natural-language queries, zone management, observation summaries and Gemini scene analysis. |
| Dashboard | `app.py` | Streamlit interface for uploading video, drawing zones, live processing, searching and viewing history. |
| Evaluation | `evaluate.py` | Reports detection counts, zone dwell, stitched track IDs and the results of sample questions. |

### Detection behaviour

- Detected classes are set by the text prompts at the top of `pipeline.py`. To detect a new kind of object, add a prompt there; no retraining is required.
- Tracked detections are drawn with a stable `GID:<id>` label, the confidence and the zone.
- Detections that the tracker has not yet assigned an ID to are still drawn, labelled `UNTRACKED`, so objects are not hidden when they first appear or during short tracker dropouts.
- SQLite stores both tracked and untracked detections. Untracked rows have `tracking_id = -1`.

### Zones

- Zones are polygons, each with a name and a colour. A point outside every polygon is reported as `Unknown / In Transit`.
- You can draw zones on the first frame of a video in the dashboard. They are saved to `zones.json`.
- If no `zones.json` exists, four quadrant zones named **Towel Top-Left**, **Towel Top-Right**, **Towel Bottom-Left** and **Towel Bottom-Right** are generated at the resolution of the video.
- A saved `zones.json` belongs to the camera resolution it was drawn on. If you change the camera or the video size, delete `zones.json` or redraw the zones, otherwise most objects will be reported as `Unknown / In Transit`.

---

## Setup

### 1. Install dependencies

Python 3.13 was used for development.

```powershell
pip install -r requirements.txt
```

The first run downloads the model weights automatically: the YOLO-World detector (`yolov8s-worldv2.pt`) and the CLIP ViT-B/32 weights. These files are large and are not stored in the repository. If the CLIP download fails, tracking still runs and identity matching uses the HSV histogram.

### 2. Gemini scene analysis (optional)

To enable scene analysis with Google Gemini 2.5 Flash, set an API key:

```powershell
$env:LLM_API_KEY="your_google_gemini_api_key"
```

Without a key, the analysis endpoint returns a mock response for testing.

### 3. Start the backend API

```powershell
python api.py
```

The API runs at `http://localhost:8000`.

### 4. Start the dashboard

In a second terminal:

```powershell
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

---

## Usage

### Process a video

1. Open the **Video Processing & Zones** tab.
2. Upload a video (`.mp4`, `.avi` or `.mov`).
3. Draw your zones on the first frame, or keep the default ones.
4. Click **Start Processing** and wait for the progress bar to finish.
5. Review the annotated video and the observation table.

Processing a video file starts a fresh log. You can also run it from the terminal. The annotated video is written to `output.mp4`:

```powershell
python pipeline.py path\to\video.mp4
```

### Live camera or RTSP

The **Live camera** section of the dashboard starts and stops a webcam or RTSP stream and shows the latest zone of each object. From a terminal:

```powershell
python pipeline.py 0
python pipeline.py rtsp://camera-address/stream
```

Press `q` in the preview window to stop. The annotated recording is saved to `output_live.mp4`. A live run adds to the existing log and drops any sighting older than 30 minutes.

### Ask where something is

Open the **Where Is My Stuff?** tab and type a question, for example:

- *Where are my keys?*
- *Where did I leave my wallet?*
- *What objects do you see?*

The answer shows the last-seen zone, time, confidence, the movement history, and a cropped image of the object next to the full frame. Common alternative names work, such as *spectacles* for glasses or *earbuds* for earphones. If an object was never seen, the system says so instead of guessing.

### Run the evaluation

After a video has been processed, or by passing a video so the script processes it first:

```powershell
python evaluate.py
python evaluate.py path\to\video.mp4
```

This writes `evaluation_results.md` with detection counts per class, zone dwell, stitched track IDs and the outcome of the sample questions. It does not compute MOTA, IDF1, HOTA or mAP, because those need labelled ground truth.

---

## Dashboard Tabs

- **Video Processing & Zones**: upload a video, draw polygon zones on the first frame, run the pipeline with live progress, download the annotated video with translucent zone overlays, and browse the SQLite records.
- **Where Is My Stuff?**: natural-language search with example chips. Shows the last-seen zone, timestamp, confidence, movement timeline, and the object crop beside the full frame.
- **Object Movement History**: cards with the last-seen zone and total detections for each object class, plus chronological zone transitions and snapshots.
- **AI Scene Analysis**: pick a keyframe from the video and ask Gemini 2.5 Flash for a description of the scene.

---

## API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/analyze-image` | POST | Send a frame image to the vision model (Gemini 2.5 Flash) |
| `/query` | POST | Ask a natural-language question about the observation log |
| `/zones` | GET | Get the current zone configuration |
| `/zones` | POST | Update and save the zone configuration |
| `/classes` | GET | List the object classes seen in the current log |
| `/summary` | GET | Observation summary for each object class |

---

## Output Files

| Path | Contents |
| :--- | :--- |
| `observations.db` | SQLite log of every sighting |
| `output.mp4` | Annotated video from a file run |
| `output_live.mp4` | Annotated recording from a live run |
| `frames/` | Keyframes saved every 30 frames |
| `crops/` | Object crops saved every 30 frames |
| `memory/` | FAISS index and metadata for visual search |
| `zones.json` | Saved zone polygons |
| `evaluation_results.md` | Report written by `evaluate.py` |

---

## Known Limitations

- Open-vocabulary detection is less reliable than a trained detector for small, dark or partly hidden items such as keys, earphones or a wallet held in a hand.
- Zones belong to one camera position and resolution.
- Without a GPU, detection plus re-identification runs at only a few frames per second, so long videos take time.
- An object that lies across two zones is reported in the zone that contains the centre of its box.

## Troubleshooting Low Detection Quality

- Make sure the scene is well lit and the object is not motion-blurred.
- Use a clear camera angle and keep objects fully in frame and apart from each other.
- If small objects are still missed, lower the detector confidence slightly in `pipeline.py`, or reword the text prompt for that object.