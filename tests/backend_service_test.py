"""Offline failure-path and coverage tests. Fixtures are never product data."""
import asyncio
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from backend.app import AnalyzeRequest, CompareRequest, app
from backend.core import ApiError, GitHub, analyze, extract_features, parse_repository, predict_forest, explain_tree_path, format_contribution_sentence, get_top_contributions, load_model, render_badge
from backend.ml.features import extract_features as training_features

NOW = datetime(2026, 10, 17, 12, tzinfo=timezone.utc)
REPO = {"full_name": "owner/project", "html_url": "https://github.com/owner/project", "created_at": "2020-01-01T00:00:00Z", "private": False}
FOREST = {"model_version": "test-only", "feature_names": ["commits_30d", "commits_90d", "commit_trend", "contributors_90d", "contributor_trend", "days_since_last_commit", "repository_age_days"], "feature_importances": [1, 0, 0, 0, 0, 0, 0], "trees": [{"children_left": [-1], "children_right": [-1], "feature": [-2], "threshold": [-2], "values": [[.7, .3]]}]}


def commit(days, login="one", email="person@example.com", sha=None):
    return {"sha": sha or f"{days}-{login}", "author": {"login": login}, "commit": {"committer": {"date": (NOW - timedelta(days=days)).isoformat()}, "author": {"email": email}}}


@pytest.mark.parametrize("value", ["https://github.com/owner/project", "owner/project", "github.com/owner/project", "https://github.com/owner/project.git/", "Owner/Project"])
def test_valid_repository(value):
    assert parse_repository(value).lower() == "owner/project"


@pytest.mark.parametrize("value", [None, 12, "https://evil.com/owner/project", "http://github.com/owner/project", "https://github.com@evil.com/owner/project", "https://github.com/owner/project/issues", "owner/..", "owner/%2e%2e", "https://github.com/owner/../project", "https://github.com/owner/project?token=secret", "https://github.com/owner/project#readme", "owner\\project", "owner/project/../../secret"])
def test_invalid_repository(value):
    with pytest.raises(ApiError):
        parse_repository(value)


def test_feature_boundaries_and_author_identity_equal_training():
    commits = [commit(0), commit(29.999, "different", " PERSON@EXAMPLE.COM "), commit(30), commit(89.999, "two", "other@example.com"), commit(90), commit(-1)]
    actual = extract_features(commits, REPO, NOW)
    assert actual == training_features(commits, REPO, NOW)
    assert actual["commits_30d"] == 2
    assert actual["commits_90d"] == 4
    assert actual["contributors_90d"] == 2
    assert actual["contributor_trend"] == 1 / 3 - 1


def test_float32_traversal_matches_sklearn_input_rounding():
    forest = {"feature_names": ["value"], "trees": [{"children_left": [1, -1, -1], "children_right": [2, -1, -1], "feature": [0, -2, -2], "threshold": [1.00000001, -2, -2], "values": [[1, 1], [1, 0], [0, 1]]}]}
    assert predict_forest({"value": 1.00000002}, forest) == 0


@pytest.mark.parametrize("created", [None, "not-a-date", "2027-01-01T00:00:00Z"])
def test_invalid_creation_metadata_cannot_generate_a_score(created):
    with pytest.raises(ApiError) as caught:
        extract_features([commit(2)], {"created_at": created}, NOW)
    assert caught.value.code == "insufficient_data"


def test_typed_request_aliases():
    assert AnalyzeRequest.model_validate({"url": "owner/project"}).repository == "owner/project"
    assert CompareRequest.model_validate({"urls": ["owner/project", "owner/other"]}).repositories == ["owner/project", "owner/other"]
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            assert (await client.post("/api/compare", json={"urls": ["owner/project"]})).status_code == 422
            assert (await client.post("/api/analyze", json={"url": "https://evil.com/a/b"})).json()["error"]["code"] == "invalid_repository"
            assert (await client.post("/api/analyze", content="not json")).status_code == 422
    asyncio.run(run())


def test_github_rate_limit_and_not_found():
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(403, headers={"x-ratelimit-remaining": "0", "retry-after": "60"}, json={}))) as client:
            with pytest.raises(ApiError) as caught:
                await GitHub(client).get("/repos/owner/project")
            assert caught.value.code == "github_rate_limit"
            assert caught.value.retry_after == 60
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(404, json={}))) as client:
            with pytest.raises(ApiError) as caught:
                await GitHub(client).get("/repos/owner/project")
            assert caught.value.status == 404
    asyncio.run(run())


def test_collection_separates_pull_requests_and_keeps_merge_counts(monkeypatch):
    import backend.core as core
    monkeypatch.setattr(core, "load_model", lambda: (FOREST, {}))
    def handler(request):
        path = request.url.path
        if path == "/repos/owner/project":
            return httpx.Response(200, json=REPO)
        if path == "/search/issues":
            return httpx.Response(200, json={"total_count": 4, "incomplete_results": False})
        if path.endswith("/commits"):
            return httpx.Response(200, json=[commit(2), commit(40, "two", "other@example.com")])
        if path.endswith("/issues"):
            return httpx.Response(200, json=[{"created_at": "2026-10-01T00:00:00Z", "closed_at": "2026-10-05T00:00:00Z"}, {"created_at": "2026-10-02T00:00:00Z", "pull_request": {}}])
        if path.endswith("/pulls"):
            return httpx.Response(200, json=[{"created_at": "2026-10-02T00:00:00Z", "merged_at": "2026-10-06T00:00:00Z"}])
        if path.endswith("/releases"):
            return httpx.Response(200, json=[])
        raise AssertionError(path)
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await analyze("owner/project", GitHub(client), NOW)
    result = asyncio.run(run())
    assert result["prediction"]["probability"] == .3
    assert result["activity"]["issues"][-1] == {"month": "2026-10", "opened": 1, "closed": 1}
    assert result["activity"]["pull_requests"][-1]["merged"] == 1
    assert len(result["activity"]["commits"]) == 6
    assert result["coverage"]["prediction_complete"] is True


def test_truncated_recent_history_withholds_prediction(monkeypatch):
    import backend.core as core
    monkeypatch.setattr(core, "load_model", lambda: (FOREST, {}))
    class Capped:
        async def get(self, path, params=None, empty_ok=False):
            return (REPO if "/repos/" in path else {"total_count": 0}), {}
        async def pages(self, name, resource, *args):
            return {"items": [commit(2), commit(120)] if resource == "commits" else [], "truncated": resource == "commits", "pages": 5 if resource == "commits" else 1, "limit": 500}
    result = asyncio.run(analyze("owner/project", Capped(), NOW))
    assert result["prediction"] is None
    assert result["coverage"]["prediction_complete"] is False
    assert result["activity"]["commits"][-1]["commits"] is None
    assert "collection cap" in result["prediction_error"]


def test_graphql_commits_collection_and_fallback(monkeypatch):
    import backend.core as core
    monkeypatch.setattr(core, "load_model", lambda: (FOREST, {}))

    graphql_payload = {
        "data": {
            "repository": {
                "defaultBranchRef": {
                    "target": {
                        "history90d": {"totalCount": 2},
                        "history6m": {
                            "totalCount": 650,
                            "pageInfo": {"hasNextPage": False, "endCursor": None},
                            "nodes": [
                                {
                                    "oid": "sha-1",
                                    "committedDate": (NOW - timedelta(days=2)).isoformat(),
                                    "author": {"name": "Alice", "email": "alice@example.com", "user": {"login": "alice"}},
                                },
                                {
                                    "oid": "sha-2",
                                    "committedDate": (NOW - timedelta(days=20)).isoformat(),
                                    "author": {"name": "Bob", "email": "bob@example.com", "user": {"login": "bob"}},
                                },
                            ],
                        },
                    }
                }
            }
        }
    }

    def handler(request):
        if request.url.path == "/graphql":
            return httpx.Response(200, json=graphql_payload)
        if request.url.path == "/repos/owner/project":
            return httpx.Response(200, json=REPO)
        if request.url.path == "/search/issues":
            return httpx.Response(200, json={"total_count": 0})
        if request.url.path.endswith("/issues") or request.url.path.endswith("/pulls") or request.url.path.endswith("/releases"):
            return httpx.Response(200, json=[])
        raise AssertionError(request.url.path)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await analyze("owner/project", GitHub(client, token="test-token"), NOW)

    result = asyncio.run(run())
    assert result["prediction"] is not None
    assert result["coverage"]["prediction_complete"] is True
    assert result["features"]["commits_30d"] == 2
    assert result["features"]["contributors_90d"] == 2


def test_graphql_truncated_history_withholds_prediction(monkeypatch):
    import backend.core as core
    monkeypatch.setattr(core, "load_model", lambda: (FOREST, {}))

    graphql_payload = {
        "data": {
            "repository": {
                "defaultBranchRef": {
                    "target": {
                        "history90d": {"totalCount": 1500},
                        "history6m": {
                            "totalCount": 2500,
                            "pageInfo": {"hasNextPage": False, "endCursor": None},
                            "nodes": [
                                {
                                    "oid": "sha-1",
                                    "committedDate": (NOW - timedelta(days=2)).isoformat(),
                                    "author": {"name": "Alice", "email": "alice@example.com", "user": {"login": "alice"}},
                                },
                            ],
                        },
                    }
                }
            }
        }
    }

    def handler(request):
        if request.url.path == "/graphql":
            return httpx.Response(200, json=graphql_payload)
        if request.url.path == "/repos/owner/project":
            return httpx.Response(200, json=REPO)
        if request.url.path == "/search/issues":
            return httpx.Response(200, json={"total_count": 0})
        return httpx.Response(200, json=[])

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await analyze("owner/project", GitHub(client, token="test-token"), NOW)

    result = asyncio.run(run())
    assert result["prediction"] is None
    assert result["coverage"]["prediction_complete"] is False
    assert "collection cap" in result["prediction_error"]


def test_tree_path_explanations():
    forest, _ = load_model()
    sample = {"commits_30d": 0, "commits_90d": 2, "commit_trend": -1, "contributors_90d": 1, "contributor_trend": -1, "days_since_last_commit": 60, "repository_age_days": 300}
    res = explain_tree_path(sample, forest)
    prob = predict_forest(sample, forest)
    sum_contrib = sum(res["contributions"].values())
    assert abs(prob - (res["bias"] + sum_contrib)) < 1e-12

    top3 = get_top_contributions(sample, forest, limit=3)
    assert len(top3) == 3
    for item in top3:
        assert item["feature"]
        assert isinstance(item["contribution"], float)
        assert isinstance(item["sentence"], str) and len(item["sentence"]) > 5
    sentence = format_contribution_sentence("commits_30d", 0, 0.15)
    assert sentence == "commits_30d dropped sharply"


def test_render_badge_outputs():
    svg = render_badge("Low")
    assert svg.strip().startswith("<svg")
    assert "low risk" in svg
    assert "#2ea44f" in svg

    svg = render_badge("Medium")
    assert "medium risk" in svg
    assert "#dfb317" in svg

    svg = render_badge("High")
    assert "high risk" in svg
    assert "#e05d44" in svg

    svg = render_badge(None)
    assert "unknown" in svg
    assert "#6e7681" in svg


def test_badge_endpoint(monkeypatch):
    import backend.core as core
    monkeypatch.setattr(core, "load_model", lambda: (FOREST, {}))

    def handler(request):
        path = request.url.path
        if path == "/graphql":
            return httpx.Response(200, json={"data": {"repository": {"defaultBranchRef": {"target": {"history90d": {"totalCount": 2}, "history6m": {"totalCount": 2, "pageInfo": {"hasNextPage": False, "endCursor": None}, "nodes": [{"oid": "sha-1", "committedDate": (NOW - timedelta(days=2)).isoformat(), "author": {"name": "Alice", "email": "alice@example.com", "user": {"login": "alice"}}}]}}}}}})
        if "/repos/" in path:
            return httpx.Response(200, json=REPO)
        if path == "/search/issues":
            return httpx.Response(200, json={"total_count": 0})
        return httpx.Response(200, json=[])

    OriginalAsyncClient = httpx.AsyncClient

    async def run():
        transport = httpx.MockTransport(handler)
        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: OriginalAsyncClient(transport=transport))
        async with OriginalAsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/api/badge/owner/project")
            assert resp.status_code == 200
            assert "image/svg+xml" in resp.headers["content-type"]
            assert resp.text.strip().startswith("<svg")
            assert "repopulse" in resp.text

            resp2 = await client.get("/badge/owner/project")
            assert resp2.status_code == 200
            assert "image/svg+xml" in resp2.headers["content-type"]

    asyncio.run(run())
