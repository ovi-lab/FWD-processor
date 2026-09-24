"""Tests for diagram generation (process_utils/diagram_prep)."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "conversation_analysis"))

from process_utils.diagram_prep.diagram_process import generate_diagram_from_file  # noqa: E402


def _line(idx, turn, topic):
    return {
        "idx": idx, "sentence": f"line {idx}", "turn": turn, "topic": topic,
        "intent": "default-intent", "emotion_label": "neutral", "emotion_score": 0.5,
        "sentiment_label": "neutral", "sentiment_score": 0.0,
        "last_interaction": False, "timestamp": f"2026-05-06 18:00:{idx:02d}", "slots": [],
    }


def test_node_linked_only_by_robot_is_shown(tmp_path):
    # The last link created is user -> robot, so the old code only checked user links and
    # hid "robot: Pets", whose only links come from robot prompts (School -> Pets -> user Pets).
    # (The first and last nodes in the list are always shown, so they can't test this.)
    transcript = [
        _line(0, "robot", "Intro"),
        _line(1, "robot", "School"),
        _line(2, "robot", "Pets"),
        _line(3, "user", "Pets"),
        _line(4, "robot", "End"),
    ]
    diagram = generate_diagram_from_file(transcript, str(tmp_path / "diagram.json"))

    by_name = {n["name"]: n for n in diagram["nodes"]}
    assert by_name["robot: Pets"]["show"] is True
