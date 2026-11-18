from __future__ import annotations

import math
import re
import sqlite3
import time
from pathlib import Path
from uuid import uuid4

from ..config import settings


RESPONSE_STYLES = ("concise", "step_by_step", "conversational")
STYLE_INSTRUCTIONS = {
    "concise": "Prefer a concise answer, while still covering the important details and caveats.",
    "step_by_step": "When useful, explain the answer in a few clear steps or with a short example.",
    "conversational": "Use a natural, conversational tone while keeping the answer direct and useful.",
}
RESEARCH_TERMS = re.compile(r"\b(research|study|studies|paper|publication|source|citation|evidence)\b", re.I)
PENDING_TTL_SECONDS = 30 * 24 * 60 * 60


def _database_path() -> Path:
    return settings.report_dir / "assistant_learning.sqlite3"


def _connect() -> sqlite3.Connection:
    path = _database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=5)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """CREATE TABLE IF NOT EXISTS response_rewards (
            category TEXT NOT NULL,
            style TEXT NOT NULL,
            helpful INTEGER NOT NULL DEFAULT 0,
            unhelpful INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (category, style)
        )"""
    )
    connection.execute(
        """CREATE TABLE IF NOT EXISTS pending_feedback (
            feedback_id TEXT PRIMARY KEY,
            category TEXT NOT NULL,
            style TEXT NOT NULL,
            created_at INTEGER NOT NULL
        )"""
    )
    return connection


def _category_for(question: str, context: dict[str, object]) -> str:
    if RESEARCH_TERMS.search(question) or context.get("sources"):
        return "research"
    if context.get("predicted_class") or context.get("visible_evidence"):
        return "plant_analysis"
    return "general"


def _choose_style(connection: sqlite3.Connection, category: str) -> str:
    rewards = {
        row["style"]: (row["helpful"], row["unhelpful"])
        for row in connection.execute(
            "SELECT style, helpful, unhelpful FROM response_rewards WHERE category = ?",
            (category,),
        )
    }
    pending = {
        row["style"]: row["count"]
        for row in connection.execute(
            "SELECT style, COUNT(*) AS count FROM pending_feedback WHERE category = ? GROUP BY style",
            (category,),
        )
    }
    for style in RESPONSE_STYLES:
        helpful, unhelpful = rewards.get(style, (0, 0))
        if helpful + unhelpful + pending.get(style, 0) == 0:
            return style

    total_pulls = sum(sum(rewards.get(style, (0, 0))) + pending.get(style, 0) for style in RESPONSE_STYLES)
    return max(
        RESPONSE_STYLES,
        key=lambda style: (
            (rewards.get(style, (0, 0))[0] + 1)
            / (sum(rewards.get(style, (0, 0))) + 2)
            + math.sqrt(2 * math.log(max(total_pulls, 2)) / max(sum(rewards.get(style, (0, 0))) + pending.get(style, 0), 1))
        ),
    )


def register_answer(question: str, context: dict[str, object]) -> dict[str, str]:
    category = _category_for(question, context)
    now = int(time.time())
    feedback_id = str(uuid4())
    try:
        with _connect() as connection:
            connection.execute("DELETE FROM pending_feedback WHERE created_at < ?", (now - PENDING_TTL_SECONDS,))
            style = _choose_style(connection, category)
            connection.execute(
                "INSERT INTO pending_feedback (feedback_id, category, style, created_at) VALUES (?, ?, ?, ?)",
                (feedback_id, category, style, now),
            )
    except (OSError, sqlite3.Error):
        return {"feedback_id": "", "category": category, "style": "concise"}
    return {"feedback_id": feedback_id, "category": category, "style": style}


def record_feedback(feedback_id: str, helpful: bool) -> bool:
    with _connect() as connection:
        row = connection.execute(
            "SELECT category, style FROM pending_feedback WHERE feedback_id = ?",
            (feedback_id,),
        ).fetchone()
        if row is None:
            return False
        helpful_column = "helpful" if helpful else "unhelpful"
        connection.execute(
            f"""INSERT INTO response_rewards (category, style, {helpful_column}) VALUES (?, ?, 1)
                ON CONFLICT(category, style) DO UPDATE SET {helpful_column} = {helpful_column} + 1""",
            (row["category"], row["style"]),
        )
        connection.execute("DELETE FROM pending_feedback WHERE feedback_id = ?", (feedback_id,))
    return True


def cancel_feedback(feedback_id: str) -> None:
    if not feedback_id:
        return
    try:
        with _connect() as connection:
            connection.execute("DELETE FROM pending_feedback WHERE feedback_id = ?", (feedback_id,))
    except (OSError, sqlite3.Error):
        pass


def learning_status() -> dict[str, object]:
    with _connect() as connection:
        rows = connection.execute(
            "SELECT category, style, helpful, unhelpful FROM response_rewards ORDER BY category, style"
        ).fetchall()
    categories: dict[str, dict[str, object]] = {}
    for row in rows:
        category = categories.setdefault(row["category"], {"ratings": 0, "helpful": 0, "styles": {}})
        ratings = row["helpful"] + row["unhelpful"]
        category["ratings"] += ratings
        category["helpful"] += row["helpful"]
        category["styles"][row["style"]] = {
            "helpful": row["helpful"],
            "unhelpful": row["unhelpful"],
        }
    for category in categories.values():
        category["preferred_style"] = max(
            RESPONSE_STYLES,
            key=lambda style: (
                (category["styles"].get(style, {}).get("helpful", 0) + 1)
                / (sum(category["styles"].get(style, {}).values()) + 2)
            ),
        )
    return {
        "ratings": sum(category["ratings"] for category in categories.values()),
        "categories": categories,
    }


def clear_learning() -> None:
    with _connect() as connection:
        connection.execute("DELETE FROM response_rewards")
        connection.execute("DELETE FROM pending_feedback")