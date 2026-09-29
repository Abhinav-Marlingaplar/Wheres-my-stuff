"""
app.py — Streamlit Frontend (Task 5 — Multi-Tab Dashboard)

Tabs:
  1. Video Processing & Zone Config
  2. "Where Is My Stuff?" — NL Query Search
  3. Object Movement History
  4. AI Scene Analysis (Gemini)
"""

import streamlit as st
import os
import requests
import cv2
import pandas as pd
from PIL import Image
from pipeline import process_video
from db import (
    init_db,
    get_all_observations,
    get_distinct_tracked_classes,
    get_observation_summary,
    get_latest_observation,
    get_object_history,
)
from query_engine import answer_query
from zones import (
    ZoneManager,
    canvas_polygon_points,
    default_zones,
    hex_to_bgr,
    hex_to_rgba,
)

_CANVAS_MAX_WIDTH = 900
_ZONE_FRAME_PATH = os.path.join("uploads", "_zone_freeze.jpg")


def _first_frame(video_path: str):
    """Return the first frame of a video, or None if it cannot be read."""
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return frame


def _preview_bgr(frame_bgr, zones: list[dict]):
    """Draw the current draft zones on a copy of the freeze-frame."""
    preview = frame_bgr.copy()
    if zones:
        mgr = ZoneManager()
        mgr.zones = zones
        mgr.draw_zones(preview, alpha=0.35)
    return preview

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Where Is My Stuff?", layout="wide")
st.title("🔎 Where Is My Stuff?")
st.caption("Computer Vision Object Tracking & Observation System — 75% Milestone")

os.makedirs("uploads", exist_ok=True)
os.makedirs("frames", exist_ok=True)
os.makedirs("crops", exist_ok=True)

# Ensure DB exists
init_db()

if "draft_zones" not in st.session_state:
    st.session_state.draft_zones = []
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0
if "zone_video_name" not in st.session_state:
    st.session_state.zone_video_name = None

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📹 Video Processing & Zones",
    "🔍 Where Is My Stuff?",
    "📋 Object Movement History",
    "🤖 AI Scene Analysis",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Video Processing & Zone Configuration
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Video Processing & Zone Configuration")

    col_upload, col_zones = st.columns([3, 2])

    with col_zones:
        st.subheader("Saved Zones")
        zm = ZoneManager()
        if zm.zones and os.path.exists(zm.config_path):
            st.caption("Loaded from `zones.json`.")
        else:
            st.caption("No saved file yet. Defaults are the four quadrants.")

        for i, zone in enumerate(zm.zones):
            st.markdown(f"**{i + 1}. {zone['name']}** — "
                        f"{len(zone['points'])} vertices")

    with col_upload:
        st.subheader("Upload & Process Video")
        uploaded_file = st.file_uploader("Upload Room Video",
                                         type=["mp4", "avi", "mov"])

        if uploaded_file is not None:
            video_path = os.path.join("uploads", uploaded_file.name)
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"✅ Uploaded **{uploaded_file.name}**")

            if st.session_state.zone_video_name != uploaded_file.name:
                freeze = _first_frame(video_path)
                if freeze is None:
                    st.error("Could not read a frame from this video.")
                else:
                    cv2.imwrite(_ZONE_FRAME_PATH, freeze)
                    h, w = freeze.shape[:2]
                    st.session_state.zone_video_name = uploaded_file.name
                    st.session_state.zone_frame_size = (w, h)
                    st.session_state.draft_zones = []
                    st.session_state.canvas_key = 0

            if st.button("▶️ Start Processing", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(current, total):
                    if total > 0:
                        progress_bar.progress(current / total)
                    status_text.text(
                        f"Processing frame {current}/{total}")

                with st.spinner("Running YOLO + ByteTrack + ReID pipeline…"):
                    output_video_path = "output.mp4"
                    if os.path.exists(output_video_path):
                        os.remove(output_video_path)

                    output_video_path = process_video(
                        video_path,
                        output_video_path=output_video_path,
                        progress_callback=update_progress,
                    )

                st.success("✅ Processing Complete!")

                st.subheader("Annotated Video")
                try:
                    st.video(output_video_path)
                except Exception:
                    st.warning("Video cannot play in-browser (codec). "
                               "Download below.")

                with open(output_video_path, "rb") as f:
                    st.download_button("⬇️ Download Annotated Video", f,
                                       file_name="annotated_output.mp4")

    # ── Draw zones on the uploaded freeze-frame ──────────────────────────
    st.markdown("---")
    st.subheader("Draw Zones")
    if st.session_state.pop("zone_save_ok", False):
        st.success("Saved to `zones.json`. Processing will use these polygons.")
    if (st.session_state.zone_video_name is None
            or not os.path.exists(_ZONE_FRAME_PATH)):
        st.info("Upload a video to draw zones on its first frame. "
                "Until then, processing uses `zones.json` or the default "
                "quadrants.")
    else:
        freeze = cv2.imread(_ZONE_FRAME_PATH)
        frame_w, frame_h = st.session_state.zone_frame_size
        scale = min(1.0, _CANVAS_MAX_WIDTH / max(frame_w, 1))
        disp_w = max(1, int(frame_w * scale))
        disp_h = max(1, int(frame_h * scale))
        x_scale = frame_w / disp_w
        y_scale = frame_h / disp_h

        preview = _preview_bgr(freeze, st.session_state.draft_zones)
        rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        background = Image.fromarray(rgb).resize((disp_w, disp_h))

        draw_col, list_col = st.columns([3, 2])
        with list_col:
            st.markdown("**Zones for this video**")
            zone_name = st.text_input("Zone name", value="Desk")
            zone_color = st.color_picker("Zone colour", value="#00FF00")
            st.caption("Left-click to place vertices. Right-click to close "
                       "the polygon. Double-click removes the last vertex.")

            if st.session_state.draft_zones:
                for i, zone in enumerate(st.session_state.draft_zones):
                    c1, c2 = st.columns([4, 1])
                    c1.markdown(
                        f"**{i + 1}. {zone['name']}** — "
                        f"{len(zone['points'])} vertices"
                    )
                    if c2.button("✕", key=f"drop_zone_{i}"):
                        st.session_state.draft_zones.pop(i)
                        st.rerun()
            else:
                st.caption("No zones drawn yet.")

            if st.button("Use quadrant defaults"):
                st.session_state.draft_zones = default_zones(frame_w, frame_h)
                st.session_state.canvas_key += 1
                st.rerun()

            if st.button("Save zones", type="primary"):
                if not st.session_state.draft_zones:
                    st.warning("Draw at least one zone before saving.")
                else:
                    ZoneManager().update_zones(st.session_state.draft_zones)
                    st.session_state.zone_save_ok = True
                    st.rerun()

        with draw_col:
            try:
                from streamlit_drawable_canvas import st_canvas
            except ImportError:
                st.error("Install `streamlit-drawable-canvas-fix` "
                         "(`pip install -r requirements.txt`) to draw zones.")
                st_canvas = None

            canvas_result = None
            if st_canvas is not None:
                canvas_result = st_canvas(
                    fill_color=hex_to_rgba(zone_color, 0.35),
                    stroke_width=2,
                    stroke_color=zone_color,
                    background_image=background,
                    update_streamlit=True,
                    height=disp_h,
                    width=disp_w,
                    drawing_mode="polygon",
                    display_toolbar=True,
                    key=f"zone_canvas_{st.session_state.canvas_key}",
                )

            if st.button("Add polygon as zone", type="primary"):
                objects = []
                if canvas_result is not None and canvas_result.json_data:
                    objects = canvas_result.json_data.get("objects") or []
                polygons = [
                    canvas_polygon_points(obj, x_scale, y_scale)
                    for obj in objects
                    if obj.get("type") == "polygon"
                ]
                polygons = [pts for pts in polygons if len(pts) >= 3]
                name = zone_name.strip()
                if not name:
                    st.warning("Give the zone a name.")
                elif not polygons:
                    st.warning("Draw a polygon first, then right-click to "
                               "close it.")
                else:
                    points = polygons[-1]
                    points = [
                        [min(max(0, x), frame_w - 1),
                         min(max(0, y), frame_h - 1)]
                        for x, y in points
                    ]
                    st.session_state.draft_zones.append({
                        "name": name,
                        "color": hex_to_bgr(zone_color),
                        "points": points,
                    })
                    st.session_state.canvas_key += 1
                    st.rerun()

    # Observations table
    st.markdown("---")
    st.subheader("📊 Observation Database")
    try:
        obs = get_all_observations()
        if obs:
            df = pd.DataFrame(obs)
            # Clean up display
            display_cols = ["frame_number", "object_class",
                            "global_track_id", "tracking_id",
                            "zone", "confidence", "timestamp"]
            available = [c for c in display_cols if c in df.columns]
            st.dataframe(df[available], use_container_width=True,
                         height=300)
            st.caption(f"Total observations: {len(df)}")
        else:
            st.info("No observations found. Process a video first.")
    except Exception:
        st.info("Database not initialised or no observations yet.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Natural Language Query ("Where Is My Stuff?")
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🔍 Where Is My Stuff?")
    st.markdown("Ask a question about any object tracked in the video.")

    # Example chips
    example_queries = [
        "Where is my laptop?",
        "Where did I leave the bottle?",
        "What objects do you see?",
        "Find my backpack",
        "Where is the chair?",
    ]
    st.caption("💡 Try one of these:")
    chip_cols = st.columns(len(example_queries))
    selected_example = None
    for i, eq in enumerate(example_queries):
        with chip_cols[i]:
            if st.button(eq, key=f"chip_{i}"):
                selected_example = eq

    query_input = st.text_input(
        "Your question:",
        value=selected_example or "",
        placeholder="e.g. Where is my laptop?",
    )

    if st.button("🔎 Search", type="primary") and query_input:
        with st.spinner("Searching…"):
            result = answer_query(query_input)

        if result["success"]:
            st.success("Found!")
        else:
            st.warning("No match found.")

        st.markdown(result["answer"])
        if result.get("retrieval") == "faiss":
            st.caption(
                "Answered from the visual memory index "
                f"(similarity {result.get('score', 0):.0%})."
            )

        # Show evidence images if available
        latest = result.get("latest")
        if latest:
            evidence_cols = st.columns(2)
            # Show crop
            crop = latest.get("crop_path")
            if crop and os.path.exists(crop):
                with evidence_cols[0]:
                    st.image(crop, caption="Object Crop",
                             use_container_width=True)
            # Show full frame
            fp = latest.get("frame_path")
            if fp and os.path.exists(fp):
                with evidence_cols[1]:
                    st.image(fp, caption="Full Frame",
                             use_container_width=True)

    # Show tracked classes for reference
    st.markdown("---")
    classes = get_distinct_tracked_classes()
    if classes:
        st.caption(f"**Tracked objects in DB:** {', '.join(classes)}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Object Movement History
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("📋 Object Movement History")
    st.markdown("See how objects moved between zones over time.")

    classes = get_distinct_tracked_classes()
    if not classes:
        st.info("No data yet. Process a video first.")
    else:
        # Summary cards
        st.subheader("Summary")
        summary = get_observation_summary()
        if summary:
            card_cols = st.columns(min(len(summary), 4))
            for i, s in enumerate(summary):
                with card_cols[i % len(card_cols)]:
                    st.metric(
                        label=s["object_class"].title(),
                        value=s["last_zone"],
                        delta=f"{s['total_detections']} detections",
                    )

        st.markdown("---")
        st.subheader("Zone Transition Timeline")
        selected_class = st.selectbox(
            "Select object class:", classes)

        if selected_class:
            history = get_object_history(object_class=selected_class)
            if history:
                st.markdown(f"**{selected_class.title()}** — "
                            f"{len(history)} zone transition(s):")

                for i, h in enumerate(history):
                    icon = "🟢" if i == 0 else "➡️"
                    st.markdown(
                        f"{icon} **{h['zone']}** — frame "
                        f"{h['frame_number']} "
                        f"(confidence {h['confidence']:.1%})"
                    )
                    # Show frame if available
                    fp = h.get("frame_path")
                    if fp and os.path.exists(fp):
                        st.image(fp, width=320,
                                 caption=f"Frame {h['frame_number']}")
            else:
                st.info(f"No transitions found for **{selected_class}**.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — AI Scene Analysis (Gemini)
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("🤖 AI Scene Analysis")
    st.markdown("Select a saved frame and send it to the Vision LLM "
                "(Google Gemini 2.5 Flash) for a detailed description.")

    frame_files = sorted(
        [f for f in os.listdir("frames") if f.endswith(".jpg")]
    ) if os.path.exists("frames") else []

    if frame_files:
        selected_frame = st.selectbox("Select a frame:", frame_files)
        frame_path = os.path.join("frames", selected_frame)
        st.image(frame_path, caption=selected_frame,
                 use_container_width=True)

        if st.button("🧠 Analyze with AI", type="primary"):
            with st.spinner("Analyzing with Vision LLM…"):
                try:
                    with open(frame_path, "rb") as f:
                        response = requests.post(
                            "http://localhost:8000/analyze-image",
                            files={"file": f},
                        )

                    if response.status_code == 200:
                        result = response.json()
                        st.subheader("Analysis Result")
                        st.write("**Description:**",
                                 result.get("description"))
                        st.write("**Objects:**",
                                 result.get("objects"))
                        st.write("**Summary:**",
                                 result.get("summary"))
                    else:
                        st.error(f"API Error: {response.status_code}")
                except Exception as e:
                    st.error(
                        f"Failed to connect to API: {e}. "
                        "Is the FastAPI server running on port 8000?"
                    )
    else:
        st.info("No frames available. Process a video first.")
