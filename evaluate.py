"""
evaluate.py — Lightweight evaluation without MOT ground truth

Reads the observation log from the last processed video and reports:
  • detection counts per class
  • zone dwell (share of detections spent in each zone)
  • how many ByteTrack IDs were stitched onto an older global track
  • whether the Task 5 sample questions still resolve

Usage:
    python evaluate.py
    python evaluate.py path/to/room.mp4
"""

import json
import os
import sys

from db import get_all_observations, init_db
from query_engine import answer_query

SAMPLE_QUESTIONS = [
    "Where is my laptop?",
    "Where did I leave my bottle?",
    "What objects do you see?",
    "Find my backpack",
    "Where is my wallet?",
]

RUN_STATS_PATH = os.path.join("memory", "run_stats.json")


def detection_counts(observations: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for obs in observations:
        name = obs.get("object_class") or "unknown"
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def zone_dwell(observations: list[dict]) -> dict[str, dict[str, int]]:
    """Detections per class per zone. A proxy for how long an object stayed put."""
    table: dict[str, dict[str, int]] = {}
    for obs in observations:
        name = obs.get("object_class") or "unknown"
        zone = obs.get("zone") or "Unknown / In Transit"
        table.setdefault(name, {})
        table[name][zone] = table[name].get(zone, 0) + 1
    return table


def stitch_summary(observations: list[dict]) -> dict:
    """
    A global track that collects more than one ByteTrack ID was re-identified.
    Extra IDs (count minus one per track) are the stitches.
    """
    groups: dict[int, set] = {}
    for obs in observations:
        gid = obs.get("global_track_id")
        bid = obs.get("tracking_id")
        if gid is None or bid is None:
            continue
        groups.setdefault(gid, set()).add(bid)
    stitched_tracks = 0
    stitched_ids = 0
    for bids in groups.values():
        if len(bids) > 1:
            stitched_tracks += 1
            stitched_ids += len(bids) - 1
    return {
        "global_tracks": len(groups),
        "byte_ids": sum(len(bids) for bids in groups.values()),
        "stitched_tracks": stitched_tracks,
        "stitched_ids": stitched_ids,
    }


def load_run_stats(path: str = RUN_STATS_PATH) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def query_status(result: dict) -> str:
    if result.get("retrieval") == "faiss":
        return "visual match"
    answer = result.get("answer") or ""
    if "Here are all the objects" in answer:
        return "summary"
    if result.get("success") and result.get("object_class"):
        return "resolved"
    if "No objects have been tracked" in answer:
        return "empty database"
    return "not observed"


def evaluate_queries(questions: list[str] | None = None) -> list[dict]:
    rows = []
    for question in questions or SAMPLE_QUESTIONS:
        result = answer_query(question)
        rows.append({
            "question": question,
            "status": query_status(result),
            "object_class": result.get("object_class"),
            "zone": (result.get("latest") or {}).get("zone"),
            "retrieval": result.get("retrieval") or "class",
        })
    return rows


def render_report(observations: list[dict], queries: list[dict],
                  run_stats: dict | None) -> str:
    counts = detection_counts(observations)
    dwell = zone_dwell(observations)
    stitches = stitch_summary(observations)
    lines = [
        "# Evaluation results",
        "",
        "Computed from the current `observations.db` log. "
        "These are not MOTA, IDF1, or HOTA — there is no tracking ground truth.",
        "",
        f"Observations: **{len(observations)}**",
        "",
        "## Detections per class",
        "",
        "| Class | Detections |",
        "| :--- | ---: |",
    ]
    if counts:
        for name, count in counts.items():
            lines.append(f"| {name} | {count} |")
    else:
        lines.append("| — | 0 |")

    lines += [
        "",
        "## Zone dwell",
        "",
        "Share of that class's detections that fell inside each zone.",
        "",
        "| Class | Zone | Detections | Share |",
        "| :--- | :--- | ---: | ---: |",
    ]
    if dwell:
        for name in sorted(dwell):
            total = sum(dwell[name].values()) or 1
            for zone, count in sorted(dwell[name].items(), key=lambda kv: -kv[1]):
                lines.append(
                    f"| {name} | {zone} | {count} | {count / total:.0%} |"
                )
    else:
        lines.append("| — | — | 0 | — |")

    lines += [
        "",
        "## Re-identification",
        "",
        f"- Global tracks: {stitches['global_tracks']}",
        f"- Distinct ByteTrack IDs: {stitches['byte_ids']}",
        f"- Tracks that absorbed a later ByteTrack ID: {stitches['stitched_tracks']}",
        f"- Stitched ByteTrack IDs (extra IDs beyond the first): {stitches['stitched_ids']}",
    ]
    if run_stats:
        lines.append(
            f"- Tracker backend recorded for this run: `{run_stats.get('backend', 'unknown')}`"
        )
        lines.append(
            f"- Stitches counted while tracking: {run_stats.get('stitches', 'n/a')}"
        )
    else:
        lines.append("- No `memory/run_stats.json` from the last pipeline run.")

    lines += [
        "",
        "## Sample questions",
        "",
        "Same prompts as the Task 5 report. "
        "`resolved` means the class parser hit a logged class. "
        "`visual match` means the FAISS index answered. "
        "`not observed` is the expected graceful miss when the item was never seen.",
        "",
        "| Question | Status | Class | Zone | Retrieval |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for row in queries:
        lines.append(
            "| {question} | {status} | {object_class} | {zone} | {retrieval} |".format(
                question=row["question"],
                status=row["status"],
                object_class=row["object_class"] or "—",
                zone=row["zone"] or "—",
                retrieval=row["retrieval"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] not in ("-h", "--help"):
        from pipeline import process_video
        print(f"Processing {args[0]} …")
        process_video(args[0])
    elif args:
        print("Usage: python evaluate.py [video_path]")
        return

    init_db()
    observations = get_all_observations()
    report = render_report(observations, evaluate_queries(), load_run_stats())
    out_path = "evaluation_results.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        sample = [
            {"object_class": "bottle", "zone": "Desk", "tracking_id": 1,
             "global_track_id": 1, "frame_number": 1},
            {"object_class": "bottle", "zone": "Desk", "tracking_id": 1,
             "global_track_id": 1, "frame_number": 2},
            {"object_class": "bottle", "zone": "Floor", "tracking_id": 5,
             "global_track_id": 1, "frame_number": 10},
            {"object_class": "laptop", "zone": "Bed", "tracking_id": 2,
             "global_track_id": 2, "frame_number": 3},
        ]
        assert detection_counts(sample) == {"bottle": 3, "laptop": 1}
        assert zone_dwell(sample)["bottle"] == {"Desk": 2, "Floor": 1}
        summary = stitch_summary(sample)
        assert summary["stitched_ids"] == 1, summary
        assert summary["global_tracks"] == 2
        assert query_status({"success": True, "object_class": "laptop", "answer": "x"}) == "resolved"
        assert query_status({"retrieval": "faiss", "success": True, "answer": "x"}) == "visual match"
        assert query_status({"success": False, "answer": "I haven't seen any **wallet**."}) == "not observed"
        print("evaluate checks ok")
    else:
        main()
