"""GitHub collection and portable random-forest inference; no database required."""
from __future__ import annotations

import asyncio
import json
import math
import re
import struct
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from .ml.features import extract_features as trained_features, normalize_commit

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["commits_30d", "commits_90d", "commit_trend", "contributors_90d", "contributor_trend", "days_since_last_commit", "repository_age_days"]
LABELS = ["Commits in 30 days", "Commits in 90 days", "Commit momentum", "Contributors in 90 days", "Contributor momentum", "Days since last commit", "Repository age"]


class ApiError(Exception):
    def __init__(self, code, message, status=400, retry_after=None):
        super().__init__(message)
        self.code, self.message, self.status, self.retry_after = code, message, status, retry_after

    def payload(self):
        result = {"code": self.code, "message": self.message}
        if self.retry_after is not None:
            result["retry_after"] = self.retry_after
        return {"error": result}


def parse_repository(value):
    if not isinstance(value, str) or len(value) > 250:
        raise ApiError("invalid_repository", "Enter a public github.com/owner/repository URL or owner/repository.")
    value = value.strip()
    if "://" in value:
        parsed = urlsplit(value)
        if parsed.scheme != "https" or parsed.netloc.lower() != "github.com" or parsed.query or parsed.fragment:
            raise ApiError("invalid_repository", "Use an HTTPS github.com repository URL without a query or fragment.")
        value = parsed.path.strip("/")
    elif value.startswith("github.com/"):
        value = value[len("github.com/"):].rstrip("/")
    else:
        value = value.rstrip("/")
    if value.endswith(".git"):
        value = value[:-4]
    parts = value.split("/")
    if len(parts) != 2 or not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?", parts[0]) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", parts[1]) or parts[1] in (".", ".."):
        raise ApiError("invalid_repository", "Enter a repository root such as https://github.com/owner/repository.")
    return "/".join(parts)


def date(value):
    if value is None or value == "":
        return None
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, timezone.utc)
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc)
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except (ValueError, TypeError, AttributeError):
        return None


def iso(value):
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def commit_date(commit):
    return date(commit.get("commit", {}).get("committer", {}).get("date") or commit.get("commit", {}).get("author", {}).get("date") or commit.get("timestamp"))


def author_key(commit):
    return normalize_commit(commit)[1]


def extract_features(commits, repository, now):
    created = date(repository.get("created_at"))
    if created is None or created > now:
        raise ApiError("insufficient_data", "GitHub returned invalid repository creation metadata. A reliable repository age is required.", 422)
    if any(commit_date(item) is not None and commit_date(item) <= now for item in commits):
        try:
            return trained_features(commits, repository, now)
        except (ValueError, TypeError, KeyError) as exc:
            raise ApiError("insufficient_data", "GitHub returned incomplete commit metadata. Reliable timestamps are required for prediction.", 422) from exc
    return {
        "commits_30d": 0, "commits_90d": 0, "commit_trend": -1,
        "contributors_90d": 0, "contributor_trend": -1,
        "days_since_last_commit": None,
        "repository_age_days": max(0, (now - created).total_seconds() / 86400),
    }


def load_model():
    try:
        forest = json.loads((ROOT / "model/forest.json").read_text(encoding="utf-8"))
        info = json.loads((ROOT / "model/model-info.json").read_text(encoding="utf-8"))
        if forest.get("feature_names") != FEATURES or not forest.get("trees"):
            raise ValueError("invalid forest")
        return forest, info
    except (OSError, ValueError, KeyError) as exc:
        raise ApiError("model_unavailable", "The trained model artifact is unavailable. Rebuild the model before predicting.", 503) from exc


def predict_forest(features, forest):
    vector = [features[name] for name in forest["feature_names"]]
    if any(v is None or not math.isfinite(v) for v in vector):
        raise ApiError("insufficient_data", "Complete recent commit history and a last commit are required for prediction.", 422)
    # sklearn converts inference inputs to float32 before tree traversal.
    vector = [struct.unpack("f", struct.pack("f", value))[0] for value in vector]
    predictions = []
    for tree in forest["trees"]:
        node = 0
        while tree["children_left"][node] != -1:
            node = tree["children_left"][node] if vector[tree["feature"][node]] <= tree["threshold"][node] else tree["children_right"][node]
        leaf = tree["values"][node]
        predictions.append(leaf[1] / sum(leaf))
    return sum(predictions) / len(predictions)


class GitHub:
    def __init__(self, client, token=None):
        self.client = client
        self.token = token
        self.headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "RepoPulse-Research/1.0"}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

    async def graphql(self, query, variables=None):
        if not self.token:
            return None
        try:
            response = await self.client.post(
                "https://api.github.com/graphql",
                json={"query": query, "variables": variables or {}},
                headers=self.headers,
                timeout=20,
            )
        except httpx.HTTPError:
            return None
        if response.status_code >= 400:
            return None
        try:
            payload = response.json()
        except ValueError:
            return None
        if payload.get("errors") and not payload.get("data"):
            return None
        return payload.get("data")

    async def get(self, path, params=None, empty_ok=False):
        try:
            response = await self.client.get("https://api.github.com" + path, params=params, headers=self.headers, timeout=20, follow_redirects=False)
        except httpx.HTTPError as exc:
            raise ApiError("github_unavailable", "GitHub could not be reached. Try again shortly.", 502) from exc
        if response.status_code == 409 and empty_ok:
            return [], response.headers
        if response.status_code in (403, 429):
            reset = response.headers.get("x-ratelimit-reset", "0")
            remaining = response.headers.get("x-ratelimit-remaining")
            retry = response.headers.get("retry-after")
            if remaining == "0" or response.status_code == 429 or retry:
                retry_seconds = max(1, int(retry)) if retry and retry.isdigit() else max(1, int(reset) - int(datetime.now(timezone.utc).timestamp())) if reset.isdigit() else 60
                raise ApiError("github_rate_limit", "GitHub's API rate limit was reached. Retry later or configure a server-side GitHub token.", 429, retry_seconds)
            raise ApiError("github_forbidden", "GitHub denied access to this public repository.", 403)
        if response.status_code == 404:
            raise ApiError("repository_not_found", "Public repository not found. Check the owner and repository name.", 404)
        if response.status_code in (301, 302, 307, 308):
            raise ApiError("repository_moved", "This repository has moved. Use its current GitHub URL.", 422)
        if response.status_code >= 400:
            raise ApiError("github_unavailable", "GitHub returned an error. Try again shortly.", 502)
        try:
            return response.json(), response.headers
        except ValueError as exc:
            raise ApiError("github_unavailable", "GitHub returned an invalid response.", 502) from exc

    async def pages(self, name, resource, params, cap, since, ordered_field=None):
        rows = []
        truncated = False
        for page in range(1, cap + 1):
            items, headers = await self.get(f"/repos/{name}/{resource}", {**params, "per_page": 100, "page": page}, empty_ok=resource == "commits")
            if not isinstance(items, list):
                raise ApiError("github_unavailable", "GitHub returned an unexpected response.", 502)
            rows.extend(items)
            reached_cutoff = ordered_field and any((date(x.get(ordered_field)) or since) < since for x in items)
            has_next = 'rel="next"' in headers.get("link", "")
            if not has_next or reached_cutoff:
                break
            if page == cap:
                truncated = True
        if resource == "commits":
            rows = list({item.get("sha", str(index)): item for index, item in enumerate(rows)}.values())
        return {"items": rows, "truncated": truncated, "pages": page, "limit": cap * 100}


async def collect_commits_graphql(name, github, since, since90, now):
    if not hasattr(github, "graphql"):
        return None
    parts = name.split("/")
    if len(parts) != 2:
        return None
    owner, repo = parts
    query = """
    query($owner: String!, $name: String!, $since: GitTimestamp, $until: GitTimestamp, $since90: GitTimestamp, $cursor: String) {
      repository(owner: $owner, name: $name) {
        defaultBranchRef {
          target {
            ... on Commit {
              history90d: history(since: $since90, until: $until, first: 1) {
                totalCount
              }
              history6m: history(since: $since, until: $until, first: 100, after: $cursor) {
                totalCount
                pageInfo {
                  hasNextPage
                  endCursor
                }
                nodes {
                  oid
                  committedDate
                  author {
                    name
                    email
                    user { login }
                  }
                }
              }
            }
          }
        }
      }
    }
    """
    try:
        data = await github.graphql(query, {
            "owner": owner,
            "name": repo,
            "since": iso(since),
            "since90": iso(since90),
            "until": iso(now),
        })
        target = ((data or {}).get("repository") or {}).get("defaultBranchRef", {}).get("target")
        if not target:
            return None
        history90d = target.get("history90d")
        history6m = target.get("history6m")
        if not history90d or not history6m:
            return None

        total90 = history90d.get("totalCount", 0)
        total6m = history6m.get("totalCount", 0)
        nodes = list(history6m.get("nodes") or [])
        page = 1
        cap = 10

        while history6m.get("pageInfo", {}).get("hasNextPage") and page < cap:
            count90 = sum(1 for n in nodes if date(n.get("committedDate")) and date(n.get("committedDate")) >= since90)
            if count90 >= total90 and len(nodes) >= total6m:
                break
            cursor = history6m["pageInfo"]["endCursor"]
            page_query = """
            query($owner: String!, $name: String!, $since: GitTimestamp, $until: GitTimestamp, $cursor: String) {
              repository(owner: $owner, name: $name) {
                defaultBranchRef {
                  target {
                    ... on Commit {
                      history(since: $since, until: $until, first: 100, after: $cursor) {
                        pageInfo {
                          hasNextPage
                          endCursor
                        }
                        nodes {
                          oid
                          committedDate
                          author {
                            name
                            email
                            user { login }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
            """
            next_data = await github.graphql(page_query, {
                "owner": owner,
                "name": repo,
                "since": iso(since),
                "until": iso(now),
                "cursor": cursor,
            })
            next_hist = (((next_data or {}).get("repository") or {}).get("defaultBranchRef") or {}).get("target", {}).get("history")
            if not next_hist or not next_hist.get("nodes"):
                break
            nodes.extend(next_hist["nodes"])
            history6m = next_hist
            page += 1

        items = [
            {
                "sha": n.get("oid"),
                "commit": {
                    "committer": {"date": n.get("committedDate")},
                    "author": {
                        "date": n.get("committedDate"),
                        "name": (n.get("author") or {}).get("name") or "unknown",
                        "email": (n.get("author") or {}).get("email") or "unknown",
                    },
                },
                "author": {"login": (n.get("author") or {}).get("user", {}).get("login")} if (n.get("author") or {}).get("user") else None,
            }
            for n in nodes
        ]
        count90_collected = sum(1 for c in items if commit_date(c) and commit_date(c) >= since90)
        complete90 = count90_collected >= total90
        truncated = len(items) < total6m
        return {
            "items": items,
            "truncated": truncated,
            "pages": page,
            "limit": cap * 100,
            "complete90": complete90,
            "source": "graphql",
        }
    except Exception:
        return None


def month_bins(now):
    year_month = now.year * 12 + now.month - 1
    return [f"{(year_month - offset) // 12:04d}-{(year_month - offset) % 12 + 1:02d}" for offset in range(5, -1, -1)]


def count_month(rows, field, months, complete):
    counts = {month: 0 for month in months}
    for item in rows:
        when = date(item.get(field))
        if when and when.strftime("%Y-%m") in counts:
            counts[when.strftime("%Y-%m")] += 1
    return {month: counts[month] if complete else None for month in months}


async def analyze(name, github, now=None):
    now = now or datetime.now(timezone.utc)
    name = parse_repository(name)
    repository, _ = await github.get(f"/repos/{name}")
    if not isinstance(repository, dict):
        raise ApiError("github_unavailable", "GitHub returned an unexpected repository response.", 502)
    if repository.get("private"):
        raise ApiError("private_repository", "Only public repositories are supported.", 403)
    months = month_bins(now)
    since = date(months[0] + "-01T00:00:00Z")
    since90 = date(now) - timedelta(days=90)
    warnings = []

    async def optional(coroutine, label):
        try:
            return await coroutine
        except ApiError as exc:
            warnings.append(f"{label}: {exc.message}")
            return {"items": [], "truncated": True, "pages": 0, "limit": 0, "error": exc.code}

    commits_graphql = await collect_commits_graphql(name, github, since, since90, now)
    issues, pulls, releases = await asyncio.gather(
        optional(github.pages(name, "issues", {"state": "all", "since": iso(since), "sort": "updated", "direction": "desc"}, 3, since, "updated_at"), "Issues"),
        optional(github.pages(name, "pulls", {"state": "all", "sort": "updated", "direction": "desc"}, 3, since, "updated_at"), "Pull requests"),
        optional(github.pages(name, "releases", {}, 2, since), "Releases"),
    )
    if commits_graphql:
        commits = commits_graphql
    else:
        commits = await github.pages(name, "commits", {"since": iso(since), "until": iso(now)}, 5, since)

    feature_commits = list(commits["items"])
    if not feature_commits:
        latest, _ = await github.get(f"/repos/{name}/commits", {"per_page": 1}, empty_ok=True)
        if isinstance(latest, list):
            feature_commits.extend(latest)
    features = extract_features(feature_commits, repository, now)
    # Commit dates can be backdated and do not prove an ordered complete window.
    # Any capped collection is therefore insufficient for a reliable prediction.
    complete90 = commits.get("complete90", not commits["truncated"])
    for resource, collection in (("Commits", commits), ("Issues", issues), ("Pull requests", pulls), ("Releases", releases)):
        if collection["truncated"] and not collection.get("error"):
            warnings.append(f"{resource} reached the {collection['limit']}-record collection cap. Incomplete chart values are left blank.")
    if repository.get("archived"):
        warnings.append("This repository is archived. Archive status is context, not an input feature in this exploratory model.")
    forest, model_info = load_model()
    prediction, prediction_error = None, None
    if not complete90:
        prediction_error = "Recent commit history exceeds the collection cap. A score would undercount activity, so prediction is unavailable."
    elif features["days_since_last_commit"] is None:
        prediction_error = "This repository has no commits. A last commit is required for this model."
    else:
        probability = predict_forest(features, forest)
        importances = forest.get("feature_importances", [0] * len(FEATURES))
        drivers = sorted([{"feature": key, "label": LABELS[i], "value": features[key], "importance": importances[i]} for i, key in enumerate(FEATURES)], key=lambda item: item["importance"], reverse=True)
        prediction = {"probability": probability, "risk": "Low" if probability < .3 else "Medium" if probability < .65 else "High", "model_version": forest["model_version"], "predicted_at": iso(now), "explanation": "Estimated probability of no default-branch commits in the next 183 days (six months). Exploratory research estimate; feature importances describe the trained model globally, not causes for this repository.", "drivers": drivers}
    if prediction_error:
        warnings.append(prediction_error)
    clean_issues = [item for item in issues["items"] if "pull_request" not in item]
    issue_open = count_month(clean_issues, "created_at", months, not issues["truncated"])
    issue_closed = count_month(clean_issues, "closed_at", months, not issues["truncated"])
    pull_open = count_month(pulls["items"], "created_at", months, not pulls["truncated"])
    pull_merged = count_month(pulls["items"], "merged_at", months, not pulls["truncated"])
    release_counts = count_month(releases["items"], "published_at", months, not releases["truncated"])
    activity_commits = []
    for month in months:
        complete = not commits["truncated"]
        selected = [item for item in commits["items"] if commit_date(item) and commit_date(item).strftime("%Y-%m") == month]
        activity_commits.append({"month": month, "commits": len(selected) if complete else None, "contributors": len({author_key(item) for item in selected}) if complete else None})
    async def open_count(kind):
        try:
            data, _ = await github.get("/search/issues", {"q": f"repo:{name} is:{kind} is:open", "per_page": 1})
            if not isinstance(data, dict):
                raise ApiError("github_unavailable", "GitHub returned an unexpected count response.", 502)
            if data.get("incomplete_results"):
                warnings.append(f"Open {kind} count is incomplete in GitHub search.")
                return None
            return data.get("total_count")
        except ApiError as exc:
            warnings.append(f"Open {kind} count unavailable: {exc.message}")
            return None
    open_issues, open_pulls = await asyncio.gather(open_count("issue"), open_count("pr"))
    latest_date = max([commit_date(item) for item in feature_commits if commit_date(item)], default=None)
    insights = [
        {"title": "Commit rhythm", "detail": f"{features['commits_30d']} commits in the last 30 days; {features['commits_90d']} in 90 days." if complete90 else "Recent commit volume exceeds the collection cap.", "tone": "neutral"},
        {"title": "Maintainer footprint", "detail": f"{features['contributors_90d']} distinct commit authors in the last 90 days." if complete90 else "Contributor coverage is incomplete.", "tone": "neutral"},
        {"title": "Research, with context", "detail": "Inactivity can reflect stability, seasonal work, or a completed project. Review the model card and repository context before relying on this estimate.", "tone": "warning"},
    ]
    return {
        "repository": {"full_name": repository["full_name"], "description": repository.get("description"), "url": repository["html_url"], "language": repository.get("language"), "stars": repository.get("stargazers_count", 0), "forks": repository.get("forks_count", 0), "age_days": features["repository_age_days"], "last_commit_at": iso(latest_date) if latest_date else None, "archived": repository.get("archived", False), "open_issues": open_issues, "open_pull_requests": open_pulls},
        "prediction": prediction, "prediction_error": prediction_error,
        "activity": {"commits": activity_commits, "issues": [{"month": m, "opened": issue_open[m], "closed": issue_closed[m]} for m in months], "pull_requests": [{"month": m, "opened": pull_open[m], "merged": pull_merged[m]} for m in months], "releases": [{"month": m, "releases": release_counts[m]} for m in months]},
        "insights": insights, "features": features,
        "coverage": {"observed_at": iso(now), "history_start": iso(since), "prediction_complete": complete90, "warnings": warnings, "default_branch_only": True, "collections": {key: {k: value[k] for k in ("truncated", "pages", "limit")} for key, value in (("commits", commits), ("issues", issues), ("pull_requests", pulls), ("releases", releases))}},
    }
