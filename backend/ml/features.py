"""Feature definitions shared by historical training and live prediction.

All windows are half-open on their left and closed at the prediction instant:
``cutoff - days < commit_time <= cutoff``. Future commits never enter a feature.
The target window is ``cutoff < commit_time <= cutoff + 183 days``.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hashlib import sha256
from math import isfinite
from typing import Any, Iterable, Mapping

FEATURE_NAMES = [
    "commits_30d", "commits_90d", "commit_trend", "contributors_90d",
    "contributor_trend", "days_since_last_commit", "repository_age_days",
]


def utc_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, (int, float)):
        result = datetime.fromtimestamp(value, timezone.utc)
    else:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)


KNOWN_BOT_HASHES = {
    "bd5a8d6c673b738d52b0",  # 49699333+dependabot[bot]@users.noreply.github.com
    "22859c22aa42a847117f",  # 27856297+dependabot-preview[bot]@users.noreply.github.com
    "4836c036306c20663d39",  # 29139614+renovate[bot]@users.noreply.github.com
}
BOT_SUBSTRINGS = ("dependabot", "github-actions", "renovate", "[bot]")


def is_bot_identity(val: Any) -> bool:
    if not val:
        return False
    low = str(val).strip().lower()
    if any(sub in low for sub in BOT_SUBSTRINGS):
        return True
    if low in ("action@github.com", "actions@github.com"):
        return True
    if low in KNOWN_BOT_HASHES:
        return True
    h = sha256(low.encode()).hexdigest()[:20]
    return h in KNOWN_BOT_HASHES


def is_bot_commit(commit: Mapping[str, Any]) -> bool:
    if "timestamp" in commit:
        author = str(commit.get("author", ""))
        if is_bot_identity(author):
            return True
    author_obj = commit.get("author")
    if isinstance(author_obj, dict):
        if is_bot_identity(author_obj.get("login", "")):
            return True
        user_obj = author_obj.get("user")
        if isinstance(user_obj, dict) and is_bot_identity(user_obj.get("login", "")):
            return True
    elif isinstance(author_obj, str) and is_bot_identity(author_obj):
        return True
    detail = commit.get("commit") or {}
    author_detail = detail.get("author") or {}
    if is_bot_identity(author_detail.get("name", "")) or is_bot_identity(author_detail.get("email", "")):
        return True
    committer_detail = detail.get("committer") or {}
    if is_bot_identity(committer_detail.get("name", "")) or is_bot_identity(committer_detail.get("email", "")):
        return True
    return False


def normalize_commit(commit: Mapping[str, Any]) -> tuple[datetime, str]:
    """Accept a GitHub REST commit or our privacy-preserving historical row."""
    if "timestamp" in commit:
        return utc_datetime(commit["timestamp"]), str(commit.get("author", "unknown"))
    detail = commit.get("commit") or {}
    stamp = (detail.get("committer") or {}).get("date") or (detail.get("author") or {}).get("date")
    author = detail.get("author") or {}
    identity = author.get("email") or (commit.get("author") or {}).get("login") or author.get("name") or "unknown"
    return utc_datetime(stamp), sha256(str(identity).strip().lower().encode()).hexdigest()[:20]


def extract_features(commits: Iterable[Mapping[str, Any]], repository: Mapping[str, Any], now: Any) -> dict[str, float]:
    cutoff = utc_datetime(now)
    created = utc_datetime(repository["created_at"])
    if created > cutoff:
        raise ValueError("Repository creation is after the observation cutoff")
    history = sorted((normalize_commit(c) for c in commits if not is_bot_commit(c)), key=lambda c: c[0])
    observed = [c for c in history if c[0] <= cutoff]
    if not observed:
        raise ValueError("No default-branch commits available before the observation cutoff")
    recent = [c for c in observed if c[0] > cutoff - timedelta(days=30)]
    prior = [c for c in observed if cutoff - timedelta(days=90) < c[0] <= cutoff - timedelta(days=30)]
    commits_90d = len(recent) + len(prior)
    recent_authors = {c[1] for c in recent}
    prior_authors = {c[1] for c in prior}
    return {
        "commits_30d": float(len(recent)),
        "commits_90d": float(commits_90d),
        "commit_trend": len(recent) / (len(prior) / 2 + 1) - 1,
        "contributors_90d": float(len(recent_authors | prior_authors)),
        "contributor_trend": len(recent_authors) / (len(prior_authors) + 1) - 1,
        "days_since_last_commit": max(0.0, (cutoff - observed[-1][0]).total_seconds() / 86400),
        "repository_age_days": max(0.0, (cutoff - created).total_seconds() / 86400),
    }


def feature_vector(features: Mapping[str, Any]) -> list[float]:
    values = [float(features[name]) for name in FEATURE_NAMES]
    if not all(isfinite(v) for v in values):
        raise ValueError("Model features must be finite numbers")
    return values


def inactivity_label(commits: Iterable[Mapping[str, Any]], cutoff: Any, collected_at: Any, horizon_days: int = 183) -> int:
    cutoff_time = utc_datetime(cutoff)
    end = cutoff_time + timedelta(days=horizon_days)
    if end > utc_datetime(collected_at):
        raise ValueError("Future outcome window is incomplete; censored examples cannot be labeled")
    return int(not any(cutoff_time < normalize_commit(c)[0] <= end for c in commits))

