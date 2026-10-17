"""Collect real, complete default-branch Git histories and repository metadata.

Usage: python scripts/collect_history.py --workers 6
No credentials are required. GITHUB_TOKEN, when set, is used only for metadata.
Commit authors are hashed before writing the historical data. No source blobs
are downloaded: bare partial clones retain only Git history metadata.
"""
from __future__ import annotations

import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REPOSITORIES = [
    "pallets/click", "pallets/itsdangerous", "pallets/markupsafe", "pallets/jinja",
    "pypa/sampleproject", "pypa/packaging", "python/typing_extensions", "python-attrs/attrs",
    "more-itertools/more-itertools", "python-humanize/humanize", "dbader/schedule", "mahmoud/boltons",
    "sindresorhus/is", "sindresorhus/p-limit", "sindresorhus/p-map", "sindresorhus/p-queue",
    "chalk/chalk", "chalk/ansi-styles", "chalk/supports-color", "debug-js/debug",
    "motdotla/dotenv", "tj/commander.js", "minimistjs/minimist", "jashkenas/underscore",
    "jashkenas/backbone", "madrobby/zepto", "request/request", "expressjs/csurf",
    "expressjs/cookie-parser", "visionmedia/page.js", "component/component", "shouldjs/should.js",
    "dylanaraps/pure-bash-bible", "dylanaraps/neofetch", "rupa/z", "rxi/log.c",
    "rxi/vec", "rxi/map", "rxi/dyad", "clibs/clib",
    "BurntSushi/termcolor", "rust-lang/bitflags", "dtolnay/anyhow", "dtolnay/thiserror",
    "matklad/once_cell", "ruby/rake", "sinatra/mustermann", "sinatra/sinatra",
]


def get_metadata(repo: str) -> dict:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "RepoPulse-research/1.0", "X-GitHub-Api-Version": "2022-11-28"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(f"https://api.github.com/repos/{repo}", headers=headers)
    with urllib.request.urlopen(request, timeout=35) as response:
        source = json.load(response)
    return {k: source.get(k) for k in ["full_name", "created_at", "default_branch", "archived", "language", "html_url"]}


def collect_one(repo: str, cache: Path, destination: Path, refresh: bool) -> dict:
    safe_name = repo.replace("/", "__")
    output = destination / f"{safe_name}.json"
    if output.exists() and not refresh:
        saved = json.loads(output.read_text(encoding="utf-8"))
        return {"repository": repo, "status": "cached", "commits": len(saved["commits"]), "source_file": str(output.relative_to(ROOT))}
    clone = cache / f"{safe_name}.git"
    try:
        metadata = get_metadata(repo)
        if not clone.exists():
            result = subprocess.run(["git", "clone", "--bare", "--single-branch", "--filter=blob:none", f"https://github.com/{repo}.git", str(clone)], capture_output=True, text=True, timeout=180)
            if result.returncode:
                raise RuntimeError(result.stderr[-700:])
        elif refresh:
            # A bare clone has no checked-out branch and may have no fetch
            # refspec. Explicitly update the current default branch, including
            # when GitHub's default branch changed since the cached snapshot.
            branch = metadata["default_branch"]
            subprocess.run(["git", "-C", str(clone), "fetch", "--filter=blob:none", "--no-tags", "origin", f"+refs/heads/{branch}:refs/heads/{branch}"], capture_output=True, check=True, timeout=120)
            subprocess.run(["git", "-C", str(clone), "symbolic-ref", "HEAD", f"refs/heads/{branch}"], capture_output=True, check=True, timeout=30)
        head_sha = subprocess.check_output(["git", "-C", str(clone), "rev-parse", "HEAD"], text=True).strip()
        log = subprocess.check_output(["git", "-C", str(clone), "log", "HEAD", "--format=%ct%x09%ae%x09%H"], text=True, encoding="utf-8", errors="replace")
        commits = []
        for line in log.splitlines():
            timestamp, author, sha = line.split("\t", 2)
            commits.append({"timestamp": int(timestamp), "author": hashlib.sha256(author.strip().lower().encode()).hexdigest()[:20], "sha": sha})
        commits.sort(key=lambda c: (c["timestamp"], c["sha"]))
        collected_at = datetime.now(timezone.utc).isoformat()
        saved = {"repository": repo, "metadata": metadata, "collected_at": collected_at, "head_sha": head_sha, "history_complete": True, "timestamp_source": "git committer timestamp (%ct)", "scope": "All ancestors reachable from the collected default-branch HEAD; source blobs excluded", "commits": commits}
        output.write_text(json.dumps(saved, separators=(",", ":")), encoding="utf-8")
        return {"repository": repo, "status": "collected", "commits": len(commits), "head_sha": head_sha, "created_at": metadata["created_at"], "collected_at": collected_at, "source_file": str(output.relative_to(ROOT))}
    except Exception as error:
        return {"repository": repo, "status": "error", "error": str(error)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--repositories", nargs="*")
    args = parser.parse_args()
    cache = ROOT / "data" / ".git-cache"
    destination = ROOT / "data" / "history"
    cache.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    started = datetime.now(timezone.utc).isoformat()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(collect_one, repo, cache, destination, args.refresh): repo for repo in (args.repositories or REPOSITORIES)}
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            records.append(row)
            print(json.dumps(row), flush=True)
    manifest = {"started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(), "collector": "scripts/collect_history.py", "selection": "Convenience sample of small public libraries across Python, JavaScript, C, shell, Rust and Ruby, including archived and maintained projects. Not representative of all GitHub.", "collection_method": "GitHub REST repository metadata and full Git default-branch ancestry via bare partial clone. No source blobs, synthetic records, or manual target labels.", "author_privacy": "Commit author emails hashed with SHA-256; only 20-character digest retained.", "records": sorted(records, key=lambda r: r["repository"])}
    (ROOT / "data" / "provenance.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    errors = [r for r in records if r["status"] == "error"]
    print(f"Collected {len(records)-len(errors)}/{len(records)} repositories; {len(errors)} errors.")


if __name__ == "__main__":
    main()
