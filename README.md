# Where Is My Stuff? — 75% Milestone

A Computer Vision-based tracking and observation system designed to answer: *"Where did I last leave this object?"*

This repository contains the prototype implementation for **CSE411 Computer Vision Project — Task 5 (Implementation Part 2)** by **Team 13**:
- Saksham Saklani (2023BCD0049)
- Niranjan Alase (2023BCD0055)
- Abhinav Marlingaplar (2023BCD0013)
- Kedar Vaishnav (2023BCS0163)

---

## Architecture Overview

1. **Detection & Tracking**: YOLOv8 Nano (`yolov8n.pt`) paired with ByteTrack (`bytetrack.yaml`) for fast, CPU-friendly multi-object detection and tracking.
2. **Visual Re-Identification (`reid.py`)**: `GlobalTracker` matches a new ByteTrack ID to a recently lost track of the same class using a CLIP ViT-B/32 embedding (cosine distance). If the CLIP weights cannot be loaded, it falls back to a 48-bin HSV histogram.
3. **Dynamic Zone Classification (`zones.py`)**: `ZoneManager` utilizing OpenCV's Point-in-Polygon algorithm (`cv2.pointPolygonTest`) to evaluate arbitrary polygonal spatial zones persisted in JSON.
4. **Natural Language Query Engine (`query_engine.py`)**: Parses natural questions (*"Where is my laptop?"*, *"Where did I leave my bottle?"*), resolves object classes via aliases and fuzzy matching, and retrieves latest locations with movement history. If no class matches, the question is embedded with CLIP and searched against the FAISS crop index.
5. **Persistent Observation Storage (`db.py`)**: SQLite database (`observations.db`) logging timestamps, frame numbers, classes, ByteTrack IDs, global track IDs, confidence, zones, crop paths, and keyframe paths.
6. **Backend REST API (`api.py`)**: FastAPI service hosting endpoints for Vision LLM scene analysis (Google Gemini 2.5 Flash), natural language queries, zone management, and observation summaries.
7. **Frontend Dashboard (`app.py`)**: Multi-tab Streamlit dashboard providing video upload, live processing, interactive zone previews, natural language search with visual evidence cards, movement history timelines, and AI scene analysis.

---

## Setup Instructions

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```
The first video run downloads CLIP ViT-B/32 weights (OpenAI). If that download fails, tracking still runs and identity stitching uses the HSV histogram.

### 2. Configure Environment Variable (Optional for AI Scene Analysis)
To enable Google Gemini 2.5 Flash scene analysis:
```powershell
$env:LLM_API_KEY="your_google_gemini_api_key"
```
*(If omitted, the AI analysis endpoint returns a mock response for testing).*

### 3. Run Backend API
```powershell
python api.py
```
The API server runs on `http://localhost:8000`.

### 4. Run Streamlit Dashboard
Open a second terminal:
```powershell
streamlit run app.py
```
The dashboard will open in your browser at `http://localhost:8501`.

### 5. Run the evaluation
After a video has been processed, or by passing a video path so the script processes it first:
```powershell
python evaluate.py
python evaluate.py path\to\room.mp4
```
This writes `evaluation_results.md` with detection counts, zone dwell, stitched track IDs, and the Task 5 sample questions. See [`milestone_75.md`](milestone_75.md).

---

## Dashboard Features

- **Tab 1: Video Processing & Zones**: Upload room videos (`.mp4`, `.avi`, `.mov`), draw polygonal zones on the first frame, run the tracking pipeline with live progress, download annotated videos with translucent zone overlays, and browse SQLite records.
- **Tab 2: Where Is My Stuff?**: Natural language search bar with example chips (*"Where is my laptop?"*, *"Where did I leave the bottle?"*, *"What objects do you see?"*). Displays last-seen zone, timestamp, confidence, movement timeline, and side-by-side cropped object preview + full frame.
- **Tab 3: Object Movement History**: Metric cards showing last-seen zones and total detections per object class, along with chronological zone transition histories and snapshot images.
- **Tab 4: AI Scene Analysis**: Select keyframes extracted from the video and query Google Gemini 2.5 Flash for detailed semantic scene descriptions.

---

## API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/analyze-image` | POST | Send a frame image to the Vision LLM (Gemini 2.5 Flash) |
| `/query` | POST | Natural-language query against the observation database |
| `/zones` | GET | Retrieve current polygonal zone configuration |
| `/zones` | POST | Update and persist polygonal zone configuration |
| `/classes` | GET | List all distinct object classes tracked in session |
| `/summary` | GET | Aggregated observation summary per object class |

---

## Reports
- **Task 4 Report (25% Milestone):** [`project_report.md`](project_report.md)
- **Task 5 Report (50% Milestone):** [`report5.md`](report5.md)
- **75% Milestone:** [`milestone_75.md`](milestone_75.md)
