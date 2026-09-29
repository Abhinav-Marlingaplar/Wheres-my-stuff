"""
query_engine.py — Natural Language Query Engine

Parses casual questions like "Where is my laptop?" or "Where did I leave
my bottle?", extracts the target object class, queries the SQLite database,
and formats a human-friendly answer with supporting evidence (zone, timestamp,
frame reference, and movement history).

Approach:
  1. Normalise the question text.
  2. Try to extract a known COCO object class via keyword / fuzzy matching
     against the list of classes actually observed in the current database.
  3. Query db.py for the latest observation and the zone-transition history.
  4. If no class matches, embed the phrase with CLIP and search the FAISS
     crop index for the nearest stored sighting.
  5. Format and return a structured result dict.
"""

import re
from difflib import get_close_matches
from db import (
    get_latest_observation,
    get_object_history,
    get_distinct_tracked_classes,
    get_observation_summary,
)
from memory import ObservationMemory, best_text_hit
from reid import get_clip_embedder


# ── Common COCO class aliases (user word → YOLO class name) ────────────────
_ALIASES: dict[str, str] = {
    "phone": "cell phone",
    "cellphone": "cell phone",
    "mobile": "cell phone",
    "tv": "tv",
    "television": "tv",
    "monitor": "tv",
    "couch": "couch",
    "sofa": "couch",
    "bag": "backpack",
    "rucksack": "backpack",
    "glasses": "wine glass",
    "spectacles": "wine glass",
    "cup": "cup",
    "mug": "cup",
    "remote": "remote",
    "controller": "remote",
    "keys": "cell phone",   # YOLO can't detect keys — best-effort alias
    "mouse": "mouse",
    "keyboard": "keyboard",
    "charger": "cell phone",
}


# ── Extraction ───────────────────────────────────────────────────────────────
_STRIP_PATTERNS = [
    r"\bwhere\s+(is|are|was|were|did)\b",
    r"\bwhat\s+(is|are|was|were)\b",
    r"\bcan\s+you\b",
    r"\bplease\b",
    r"\bdid\b",
    r"\bdo\b",
    r"\bhave\b",
    r"\bhas\b",
    r"\bi\b",
    r"\bmy\b",
    r"\bthe\b",
    r"\ba\b",
    r"\ban\b",
    r"\bfind\b",
    r"\blocate\b",
    r"\bshow\s+me\b",
    r"\bshow\b",
    r"\btell\s+me\b",
    r"\bleft\b",
    r"\bleave\b",
    r"\bput\b",
    r"\bplaced\b",
    r"\bdropped\b",
    r"\bkept\b",
    r"\bmoved\b",
    r"\btook\b",
    r"\blast\s+(seen|known|location|time)\b",
    r"\blast\b",
    r"\babout\b",
    r"\?",
]


def _extract_object_name(question: str) -> str:
    """Best-effort extraction of the target object class from a question."""
    text = question.lower().strip()
    for pat in _STRIP_PATTERNS:
        text = re.sub(pat, "", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Remove leading/trailing filler
    text = text.strip(" ?,.'\"!")
    return text


def _resolve_class(raw_name: str, tracked_classes: list[str]) -> str | None:
    """
    Map the extracted name to a class actually present in the database.
    Priority: exact match → alias → fuzzy match.
    """
    lower_tracked = {c.lower(): c for c in tracked_classes}

    # 1. Exact match
    if raw_name in lower_tracked:
        return lower_tracked[raw_name]

    # 2. Alias lookup
    if raw_name in _ALIASES:
        alias = _ALIASES[raw_name]
        if alias.lower() in lower_tracked:
            return lower_tracked[alias.lower()]

    # 3. Substring match — e.g. "bottle" matches "bottle"
    for key, original in lower_tracked.items():
        if raw_name in key or key in raw_name:
            return original

    # 4. Fuzzy match (difflib)
    close = get_close_matches(raw_name, list(lower_tracked.keys()), n=1, cutoff=0.5)
    if close:
        return lower_tracked[close[0]]

    return None


# ── Public API ───────────────────────────────────────────────────────────────
def answer_query(question: str) -> dict:
    """
    Process a natural-language question and return a structured answer.

    Returns dict with keys:
      - success: bool
      - question: original question
      - object_class: resolved class (or None)
      - answer: human-readable answer string
      - latest: latest observation dict (or None)
      - history: list of zone-transition dicts
      - tracked_classes: list of all classes in the DB (for UI hints)
    """
    tracked_classes = get_distinct_tracked_classes()

    # Special case: "what objects" / "list" / "show everything"
    general_keywords = ["what objects", "list", "everything", "all objects",
                        "what do you see", "what can you see", "summary"]
    if any(kw in question.lower() for kw in general_keywords):
        summary = get_observation_summary()
        if not summary:
            return _result(question, None, "No objects have been tracked yet.",
                           tracked_classes=tracked_classes)
        lines = ["Here are all the objects I've tracked:\n"]
        for s in summary:
            lines.append(
                f"• **{s['object_class']}** — last seen in **{s['last_zone']}** "
                f"(frame {s['last_seen_frame']}, {s['total_detections']} detections)"
            )
        return _result(question, None, "\n".join(lines),
                       tracked_classes=tracked_classes)

    # Extract object name
    raw_name = _extract_object_name(question)
    if not raw_name:
        return _result(
            question, None,
            "I couldn't understand which object you're asking about. "
            f"Try asking about one of these: {', '.join(tracked_classes)}.",
            tracked_classes=tracked_classes,
        )

    resolved = _resolve_class(raw_name, tracked_classes)
    if resolved is None:
        visual = _visual_search(question, raw_name, tracked_classes)
        if visual is not None:
            return visual
        return _result(
            question, raw_name,
            f"I haven't seen any **{raw_name}** in the video. "
            f"Objects I've tracked: {', '.join(tracked_classes) if tracked_classes else 'none yet'}.",
            tracked_classes=tracked_classes,
        )

    # Fetch data
    latest = get_latest_observation(resolved)
    history = get_object_history(object_class=resolved)

    if latest is None:
        return _result(
            question, resolved,
            f"No observations found for **{resolved}**.",
            tracked_classes=tracked_classes,
        )

    # Build answer
    answer_lines = [
        f"🔍 **{resolved.title()}** was last seen in **{latest['zone']}**.",
        f"   • Frame: {latest['frame_number']}",
        f"   • Confidence: {latest['confidence']:.1%}",
        f"   • Timestamp: {latest['timestamp']}",
    ]
    if latest.get("global_track_id"):
        answer_lines.append(f"   • Global Track ID: {latest['global_track_id']}")

    if len(history) > 1:
        answer_lines.append("\n📋 **Movement history** (zone transitions):")
        for i, h in enumerate(history):
            arrow = "→" if i > 0 else "⬤"
            answer_lines.append(
                f"   {arrow} **{h['zone']}** at frame {h['frame_number']}"
            )

    return _result(
        question, resolved, "\n".join(answer_lines),
        latest=latest, history=history, tracked_classes=tracked_classes,
    )


def _visual_search(question: str, raw_name: str, tracked_classes: list[str]) -> dict | None:
    """CLIP text search over the FAISS crop index when no class name matches."""
    memory = ObservationMemory().load()
    if not memory.meta:
        return None
    embedder = get_clip_embedder()
    if not embedder.available:
        return None
    vector = embedder.embed_text(f"a photo of a {raw_name}")
    if vector is None:
        return None
    hit = best_text_hit(memory.search(vector, k=5))
    if hit is None:
        return None

    latest = {
        "frame_number": hit.get("frame_number"),
        "object_class": hit.get("object_class"),
        "zone": hit.get("zone"),
        "timestamp": hit.get("timestamp"),
        "confidence": hit.get("confidence") or 0,
        "global_track_id": hit.get("global_track_id"),
        "crop_path": hit.get("crop_path"),
        "frame_path": hit.get("frame_path"),
    }
    history = []
    if hit.get("global_track_id") is not None:
        history = get_object_history(global_track_id=hit["global_track_id"])

    score = hit["score"]
    answer_lines = [
        f"No tracked class matched **{raw_name}**.",
        f"The closest sighting in the visual memory is a "
        f"**{latest['object_class']}** in **{latest['zone']}** "
        f"(similarity {score:.0%}).",
        f"   • Frame: {latest['frame_number']}",
        f"   • Timestamp: {latest['timestamp']}",
    ]
    if latest.get("global_track_id"):
        answer_lines.append(f"   • Global Track ID: {latest['global_track_id']}")
    result = _result(
        question, latest["object_class"], "\n".join(answer_lines),
        latest=latest, history=history, tracked_classes=tracked_classes,
    )
    result["retrieval"] = "faiss"
    result["score"] = score
    return result


def _result(question, object_class, answer, latest=None, history=None,
            tracked_classes=None):
    return {
        "success": latest is not None or "Here are" in answer,
        "question": question,
        "object_class": object_class,
        "answer": answer,
        "latest": latest,
        "history": history or [],
        "tracked_classes": tracked_classes or [],
    }


# ── Self-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from db import init_db
    init_db()
    test_questions = [
        "Where is my laptop?",
        "Where did I leave my bottle?",
        "What objects do you see?",
        "Find my backpack",
        "Show me the remote",
    ]
    for q in test_questions:
        result = answer_query(q)
        print(f"\nQ: {q}")
        print(f"A: {result['answer']}")
