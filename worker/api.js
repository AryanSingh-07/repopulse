import forest from '../model/forest.json';
import modelInfo from '../model/model-info.json';
import { CACHE_TTL_MS, createCacheStore, sharedCache, credentialNamespace, readCache, writeCache, deduplicate, observeQuota, setCooldown, cooldownUntil, rateLimitStatus, retryTime } from './cache.js';

export { createCacheStore } from './cache.js';
const nativeFetch = (...args) => globalThis.fetch(...args);

export const FEATURES = ['commits_30d', 'commits_90d', 'commit_trend', 'contributors_90d', 'contributor_trend', 'days_since_last_commit', 'repository_age_days'];
const LABELS = ['Commits in 30 days', 'Commits in 90 days', 'Commit momentum', 'Contributors in 90 days', 'Contributor momentum', 'Days since last commit', 'Repository age'];
const DAY = 86400000;

export class ApiError extends Error {
  constructor(code, message, status = 400, retryAfter) {
    super(message); Object.assign(this, { code, status, retryAfter });
  }
}

export function parseRepository(input) {
  if (typeof input !== 'string' || input.length > 250) throw new ApiError('invalid_repository', 'Enter a public github.com/owner/repository URL or owner/repository.');
  let value = input.trim();
  if (value.includes('://')) {
    let url;
    try { url = new URL(value); } catch { throw new ApiError('invalid_repository', 'Enter a valid HTTPS GitHub repository URL.'); }
    // Validate the original path too: URL() normalizes traversal components.
    const rawPath = value.slice(value.indexOf('://') + 3).replace(/^[^/]+/, '');
    if (url.protocol !== 'https:' || url.host.toLowerCase() !== 'github.com' || url.username || url.password || url.search || url.hash || /%|\\|(?:^|\/)\.{1,2}(?:\/|$)/.test(rawPath)) throw new ApiError('invalid_repository', 'Use an HTTPS github.com repository URL without a query or fragment.');
    value = url.pathname.replace(/^\/+|\/+$/g, '');
  } else if (value.startsWith('github.com/')) value = value.slice(11).replace(/\/+$/, '');
  else value = value.replace(/\/+$/, '');
  if (value.endsWith('.git')) value = value.slice(0, -4);
  const parts = value.split('/');
  if (parts.length !== 2 || !/^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$/.test(parts[0]) || !/^[A-Za-z0-9_.-]{1,100}$/.test(parts[1]) || ['.', '..'].includes(parts[1])) throw new ApiError('invalid_repository', 'Enter a repository root such as https://github.com/owner/repository.');
  return parts.join('/');
}

const date = value => { const time = typeof value === 'number' ? value * 1000 : value ? Date.parse(value) : NaN; return Number.isFinite(time) ? time : null; };
const commitDate = c => date(c.commit?.committer?.date || c.commit?.author?.date || c.timestamp);
const authorKey = c => 'timestamp' in c ? String(c.author ?? 'unknown') : String(c.commit?.author?.email || c.author?.login || c.commit?.author?.name || 'unknown').trim().toLowerCase();
const iso = time => new Date(time).toISOString();

export function extractFeatures(commits, repository, now = Date.now()) {
  const created = date(repository.created_at);
  if (created === null || created > now) throw new ApiError('insufficient_data', 'GitHub returned invalid repository creation metadata. A reliable repository age is required.', 422);
  const recent = [], prior = [], dates = [];
  for (const commit of commits) {
    const time = commitDate(commit);
    if (time === null) throw new ApiError('insufficient_data', 'GitHub returned incomplete commit metadata. Reliable timestamps are required for prediction.', 422);
    if (time > now) continue;
    dates.push(time);
    const days = (now - time) / DAY;
    if (days < 30) recent.push(commit); else if (days < 90) prior.push(commit);
  }
  return {
    commits_30d: recent.length, commits_90d: recent.length + prior.length,
    commit_trend: recent.length / (prior.length / 2 + 1) - 1,
    contributors_90d: new Set([...recent, ...prior].map(authorKey)).size,
    contributor_trend: new Set(recent.map(authorKey)).size / (new Set(prior.map(authorKey)).size + 1) - 1,
    days_since_last_commit: dates.length ? Math.max(0, (now - Math.max(...dates)) / DAY) : null,
    repository_age_days: Math.max(0, (now - created) / DAY),
  };
}

function requireModel() {
  if (!forest?.trees?.length || JSON.stringify(forest.feature_names) !== JSON.stringify(FEATURES)) throw new ApiError('model_unavailable', 'The trained model artifact is unavailable. Rebuild the model before predicting.', 503);
}

export function predictForest(features, model = forest) {
  let vector = model.feature_names.map(name => features[name]);
  if (vector.some(value => value === null || !Number.isFinite(value))) throw new ApiError('insufficient_data', 'Complete recent commit history and a last commit are required for prediction.', 422);
  vector = vector.map(Math.fround);
  const scores = model.trees.map(tree => {
    let node = 0;
    while (tree.children_left[node] !== -1) node = vector[tree.feature[node]] <= tree.threshold[node] ? tree.children_left[node] : tree.children_right[node];
    const leaf = tree.values[node];
    return leaf[1] / leaf.reduce((sum, value) => sum + value, 0);
  });
  return scores.reduce((sum, score) => sum + score, 0) / scores.length;
}

export class GitHub {
  constructor(token, fetcher = nativeFetch, cacheStore) {
    this.fetcher = fetcher;
    // Injected transports use isolated stores unless a test explicitly shares one.
    this.store = cacheStore || (fetcher === nativeFetch ? sharedCache : createCacheStore());
    this.namespace = credentialNamespace(token);
    this.cacheHits = 0;
    this.oldestObservationAt = null;
    this.dataExpiresAt = null;
    this.sanitizeError = error => {
      const raw = String(error?.message || 'Request failed');
      const scrubbed = token ? raw.split(token).join('[redacted]') : raw;
      return scrubbed
        .replace(/https?:\/\/\S+/gi, '[url]')
        .replace(/\b(?:authorization|cookie|set-cookie)\s*[:=][^\r\n]*/gi, '[header redacted]')
        .replace(/\b(?:Bearer\s+)?(?:gh[pousr]_[A-Za-z0-9_]+|github_pat_[A-Za-z0-9_]+)/gi, '[redacted]')
        .slice(0, 240);
    };
    this.headers = { Accept: 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28', 'User-Agent': 'RepoPulse-Research/1.0' };
    if (token) this.headers.Authorization = `Bearer ${token}`;
  }
  async get(path, params = {}, emptyOk = false) {
    const url = new URL(`https://api.github.com${path}`);
    for (const [key, value] of Object.entries(params)) url.searchParams.set(key, value);
    url.searchParams.sort();
    const namespace = await this.namespace, resource = path.startsWith('/search/') ? 'search' : 'core';
    const cacheUrl = new URL(url);
    // Live analyses only move `until` by seconds. Reuse the actual observed
    // commit response for at most ten minutes, and expose its original data time.
    // Explicit historical queries retain their exact cutoff in the cache key.
    if (path.endsWith('/commits') && Math.abs(Date.parse(params.until) - this.store.clock()) <= CACHE_TTL_MS) cacheUrl.searchParams.delete('until');
    const key = `${namespace}|http|${cacheUrl}|${emptyOk}`;
    const cached = readCache(this.store, key);
    if (cached) {
      this.recordObservation(cached.observedAt, cached.expiresAt, true);
      return [cached.value.data, new Headers(cached.value.headers)];
    }
    const result = await deduplicate(this.store, key, async () => {
      const blocked = cooldownUntil(this.store, namespace, resource);
      if (blocked) throw this.rateError(namespace, resource, blocked);
      const result = await this.request(url, emptyOk, namespace, resource);
      if (result.cacheable && !result.data?.private) writeCache(this.store, key, { data: result.data, headers: result.headers });
      return result;
    });
    this.recordObservation(result.observedAt, result.observedAt + CACHE_TTL_MS, false);
    return [result.data, new Headers(result.headers)];
  }
  recordObservation(observedAt, expiresAt, cached) {
    this.oldestObservationAt = Math.min(this.oldestObservationAt ?? observedAt, observedAt);
    this.dataExpiresAt = Math.min(this.dataExpiresAt ?? expiresAt, expiresAt);
    if (cached) this.cacheHits++;
  }
  rateError(namespace, resource, until) {
    const snapshot = rateLimitStatus(this.store, namespace, resource);
    const error = new ApiError('github_rate_limit', namespace === 'anonymous'
      ? "GitHub's shared public API quota is exhausted. Retry at the listed time or configure a server-side GitHub token."
      : "GitHub's API quota for the configured token is exhausted. Retry at the listed time.",
    429, Math.max(1, Math.ceil((until - this.store.clock()) / 1000)));
    Object.assign(error, { retryAt: iso(until), resetAt: snapshot.reset_at || null, rateLimit: snapshot, tokenConfigured: namespace !== 'anonymous' });
    return error;
  }
  async quotaStatus() { return rateLimitStatus(this.store, await this.namespace); }
  async request(url, emptyOk, namespace, resource) {
    let response, timer;
    const controller = new AbortController();
    const signal = typeof globalThis.AbortSignal?.timeout === 'function'
      ? globalThis.AbortSignal.timeout(20000)
      : (timer = setTimeout(() => controller.abort(), 20000), controller.signal);
    try { response = await this.fetcher(url.toString(), { headers: this.headers, redirect: 'manual', signal }); }
    catch (error) {
      console.error('GitHub network failure', String(error?.name || 'Error').replace(/[^A-Za-z0-9_]/g, '').slice(0, 64), this.sanitizeError(error));
      throw new ApiError('github_unavailable', 'GitHub could not be reached. Try again shortly.', 502);
    } finally { if (timer !== undefined) clearTimeout(timer); }
    observeQuota(this.store, namespace, resource, response.headers);
    const headers = ['link', 'date', 'etag', 'last-modified', 'x-ratelimit-limit', 'x-ratelimit-remaining', 'x-ratelimit-used', 'x-ratelimit-reset', 'x-ratelimit-resource']
      .filter(name => response.headers.has(name)).map(name => [name, response.headers.get(name)]);
    if (response.status === 409 && emptyOk) return { data: [], headers, observedAt: this.store.clock(), cacheable: false };
    if (response.status === 401) throw new ApiError('github_authentication', 'The configured GitHub token was rejected. Replace or remove the server-side GITHUB_TOKEN.', 503);
    if ([403, 429].includes(response.status)) {
      const remaining = response.headers.get('x-ratelimit-remaining'), retry = response.headers.get('retry-after');
      if (remaining === '0' || response.status === 429 || retry) {
        const until = retryTime(response.headers, this.store.clock());
        const limitedResource = response.headers.get('x-ratelimit-resource') || resource;
        // Secondary limits apply across resources; primary search/core buckets differ.
        setCooldown(this.store, namespace, remaining === '0' ? limitedResource : '*', until);
        throw this.rateError(namespace, limitedResource, until);
      }
      throw new ApiError('github_forbidden', 'GitHub denied access to this public repository.', 403);
    }
    if (response.status === 404) throw new ApiError('repository_not_found', 'Public repository not found. Check the owner and repository name.', 404);
    if ([301, 302, 307, 308].includes(response.status)) throw new ApiError('repository_moved', 'This repository has moved. Use its current GitHub URL.', 422);
    if (!response.ok) throw new ApiError('github_unavailable', 'GitHub returned an error. Try again shortly.', 502);
    try { return { data: await response.json(), headers, observedAt: this.store.clock(), cacheable: true }; }
    catch { throw new ApiError('github_unavailable', 'GitHub returned an invalid response.', 502); }
  }
  async pages(name, resource, params, cap, since, orderedField) {
    const rows = [];
    let truncated = false, page = 1;
    for (; page <= cap; page++) {
      const [items, headers] = await this.get(`/repos/${name}/${resource}`, { ...params, per_page: 100, page }, resource === 'commits');
      if (!Array.isArray(items)) throw new ApiError('github_unavailable', 'GitHub returned an unexpected response.', 502);
      rows.push(...items);
      const reachedCutoff = orderedField && items.some(item => (date(item[orderedField]) ?? since) < since);
      if (!headers.get('link')?.includes('rel="next"') || reachedCutoff) break;
      if (page === cap) truncated = true;
    }
    const items = resource === 'commits' ? [...new Map(rows.map((item, index) => [item.sha ?? String(index), item])).values()] : rows;
    return { items, truncated, pages: Math.min(page, cap), limit: cap * 100 };
  }
}

const monthBins = now => {
  const d = new Date(now), index = d.getUTCFullYear() * 12 + d.getUTCMonth();
  return Array.from({ length: 6 }, (_, n) => { const i = index - 5 + n; return `${Math.floor(i / 12)}-${String(i % 12 + 1).padStart(2, '0')}`; });
};
const monthOf = time => time === null ? null : iso(time).slice(0, 7);
function countMonth(rows, field, months, complete) {
  return Object.fromEntries(months.map(month => [month, complete ? rows.filter(item => monthOf(date(item[field])) === month).length : null]));
}

export async function analyze(input, github, now = Date.now()) {
  const name = parseRepository(input);
  const [repository] = await github.get(`/repos/${name}`);
  if (!repository || typeof repository !== 'object' || Array.isArray(repository)) throw new ApiError('github_unavailable', 'GitHub returned an unexpected repository response.', 502);
  if (repository.private) throw new ApiError('private_repository', 'Only public repositories are supported.', 403);
  const months = monthBins(now), since = date(`${months[0]}-01T00:00:00Z`), warnings = [];
  const optional = async (promise, label) => {
    try { return await promise; } catch (error) {
      if (!(error instanceof ApiError)) throw error;
      warnings.push(`${label}: ${error.message}`);
      return { items: [], truncated: true, pages: 0, limit: 0, error: error.code };
    }
  };
  const [commits, issues, pulls, releases] = await Promise.all([
    github.pages(name, 'commits', { since: iso(since), until: iso(now) }, 5, since),
    optional(github.pages(name, 'issues', { state: 'all', since: iso(since), sort: 'updated', direction: 'desc' }, 3, since, 'updated_at'), 'Issues'),
    optional(github.pages(name, 'pulls', { state: 'all', sort: 'updated', direction: 'desc' }, 3, since, 'updated_at'), 'Pull requests'),
    optional(github.pages(name, 'releases', {}, 2, since), 'Releases'),
  ]);
  const featureCommits = [...commits.items];
  if (!featureCommits.length) { const [latest] = await github.get(`/repos/${name}/commits`, { per_page: 1 }, true); featureCommits.push(...latest); }
  const features = extractFeatures(featureCommits, repository, now);
  // Backdated timestamps do not establish completeness when a page cap is hit.
  const complete90 = !commits.truncated;
  for (const [label, collection] of [['Commits', commits], ['Issues', issues], ['Pull requests', pulls], ['Releases', releases]]) {
    if (collection.truncated && !collection.error) warnings.push(`${label} reached the ${collection.limit}-record collection cap. Incomplete chart values are left blank.`);
  }
  if (repository.archived) warnings.push('This repository is archived. Archive status is context, not an input feature in this exploratory model.');
  requireModel();
  let prediction = null, predictionError = null;
  if (!complete90) predictionError = 'Recent commit history exceeds the collection cap. A score would undercount activity, so prediction is unavailable.';
  else if (features.days_since_last_commit === null) predictionError = 'This repository has no commits. A last commit is required for this model.';
  else {
    const probability = predictForest(features);
    const drivers = FEATURES.map((feature, i) => ({ feature, label: LABELS[i], value: features[feature], importance: forest.feature_importances?.[i] ?? 0 })).sort((a, b) => b.importance - a.importance);
    prediction = { probability, risk: probability < .3 ? 'Low' : probability < .65 ? 'Medium' : 'High', model_version: forest.model_version, predicted_at: iso(now), explanation: 'Estimated probability of no default-branch commits in the next 183 days (six months). Exploratory research estimate; feature importances describe the trained model globally, not causes for this repository.', drivers };
  }
  if (predictionError) warnings.push(predictionError);
  const cleanIssues = issues.items.filter(item => !('pull_request' in item));
  const issueOpen = countMonth(cleanIssues, 'created_at', months, !issues.truncated), issueClosed = countMonth(cleanIssues, 'closed_at', months, !issues.truncated);
  const pullOpen = countMonth(pulls.items, 'created_at', months, !pulls.truncated), pullMerged = countMonth(pulls.items, 'merged_at', months, !pulls.truncated);
  const releaseCounts = countMonth(releases.items, 'published_at', months, !releases.truncated);
  const activityCommits = months.map(month => {
    const complete = !commits.truncated;
    const selected = commits.items.filter(item => monthOf(commitDate(item)) === month);
    return { month, commits: complete ? selected.length : null, contributors: complete ? new Set(selected.map(authorKey)).size : null };
  });
  async function openCount(kind) {
    try {
      const [data] = await github.get('/search/issues', { q: `repo:${name} is:${kind} is:open`, per_page: 1 });
      if (!data || typeof data !== 'object' || Array.isArray(data)) throw new ApiError('github_unavailable', 'GitHub returned an unexpected count response.', 502);
      if (data.incomplete_results) { warnings.push(`Open ${kind} count is incomplete in GitHub search.`); return null; }
      return data.total_count ?? null;
    } catch (error) { if (!(error instanceof ApiError)) throw error; warnings.push(`Open ${kind} count unavailable: ${error.message}`); return null; }
  }
  const [openIssues, openPulls] = await Promise.all([openCount('issue'), openCount('pr')]);
  const allDates = featureCommits.map(commitDate).filter(time => time !== null);
  return {
    repository: { full_name: repository.full_name, description: repository.description ?? null, url: repository.html_url, language: repository.language ?? null, stars: repository.stargazers_count ?? 0, forks: repository.forks_count ?? 0, age_days: features.repository_age_days, last_commit_at: allDates.length ? iso(Math.max(...allDates)) : null, archived: !!repository.archived, open_issues: openIssues, open_pull_requests: openPulls },
    prediction, prediction_error: predictionError, features,
    activity: { commits: activityCommits, issues: months.map(month => ({ month, opened: issueOpen[month], closed: issueClosed[month] })), pull_requests: months.map(month => ({ month, opened: pullOpen[month], merged: pullMerged[month] })), releases: months.map(month => ({ month, releases: releaseCounts[month] })) },
    insights: [
      { title: 'Commit rhythm', detail: complete90 ? `${features.commits_30d} commits in the last 30 days; ${features.commits_90d} in 90 days.` : 'Recent commit volume exceeds the collection cap.', tone: 'neutral' },
      { title: 'Maintainer footprint', detail: complete90 ? `${features.contributors_90d} distinct commit authors in the last 90 days.` : 'Contributor coverage is incomplete.', tone: 'neutral' },
      { title: 'Research, with context', detail: 'Inactivity can reflect stability, seasonal work, or a completed project. Review the model card and repository context before relying on this estimate.', tone: 'warning' },
    ],
    coverage: { observed_at: iso(now), history_start: iso(since), prediction_complete: complete90, warnings, default_branch_only: true, collections: Object.fromEntries([['commits', commits], ['issues', issues], ['pull_requests', pulls], ['releases', releases]].map(([name, value]) => [name, { truncated: value.truncated, pages: value.pages, limit: value.limit, ...(value.error ? { error: value.error } : {}) }])) },
  };
}

export async function analyzeCached(input, github, now = Date.now()) {
  const name = parseRepository(input), namespace = await github.namespace;
  const key = `${namespace}|analysis|${forest.model_version}|${name.toLowerCase()}`;
  const cached = readCache(github.store, key);
  if (cached) {
    return { ...cached.value, coverage: { ...cached.value.coverage, cached: true, cache_scope: 'worker-memory', cache_expires_at: iso(cached.expiresAt), github_rate_limit: await github.quotaStatus() } };
  }
  const result = await deduplicate(github.store, key, async () => {
    const value = await analyze(name, github, now);
    const expiresAt = Math.min(github.store.clock() + CACHE_TTL_MS, github.dataExpiresAt ?? Infinity);
    value.coverage = {
      ...value.coverage, cached: github.cacheHits > 0, cache_scope: 'worker-memory',
      data_observed_at: iso(github.oldestObservationAt ?? github.store.clock()),
      cache_expires_at: iso(expiresAt), cache_ttl_seconds: CACHE_TTL_MS / 1000,
      github_rate_limit: await github.quotaStatus(),
    };
    // Optional upstream failures stay visible but are retried on the next request;
    // successful underlying GETs still avoid repeating the rest of the work.
    const completeFetch = Object.values(value.coverage.collections).every(collection => !collection.error)
      && value.repository.open_issues !== null && value.repository.open_pull_requests !== null;
    value.coverage.cacheable = completeFetch;
    if (completeFetch && expiresAt > github.store.clock()) writeCache(github.store, key, value, expiresAt - github.store.clock());
    return value;
  });
  return result;
}

export async function handleApi(request, env = {}) {
  const respond = (payload, status = 200, extra = {}) => new Response(JSON.stringify(payload), { status, headers: { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff', ...extra } });
  try {
    const path = new URL(request.url).pathname.replace(/^\/api/, '');
    if (request.method === 'GET' && path === '/health') { requireModel(); return respond({ status: 'ok', model_version: forest.model_version, runtime: 'edge', github_token_configured: !!env.GITHUB_TOKEN, rate_limit: await new GitHub(env.GITHUB_TOKEN).quotaStatus(), cache: { scope: 'worker-memory', ttl_seconds: CACHE_TTL_MS / 1000, persistent: false } }); }
    if (request.method === 'GET' && path === '/model-info') { requireModel(); return respond(modelInfo); }
    if (!['/analyze', '/predict', '/compare'].includes(path)) return respond({ error: { code: 'not_found', message: 'API endpoint not found.' } }, 404);
    if (request.method !== 'POST') return respond({ error: { code: 'method_not_allowed', message: 'Use POST with a JSON request body.' } }, 405, { Allow: 'POST' });
    const text = await request.text();
    if (text.length > 4096) throw new ApiError('invalid_request', 'Request body is too large.', 413);
    let body;
    try { body = JSON.parse(text); } catch { throw new ApiError('invalid_request', 'Send a JSON request body.'); }
    if (!body || Array.isArray(body) || typeof body !== 'object') throw new ApiError('invalid_request', 'Send a JSON object.');
    const github = new GitHub(env.GITHUB_TOKEN);
    if (path === '/compare') {
      const names = body.repositories ?? body.urls;
      if (!Array.isArray(names) || names.length !== 2) throw new ApiError('invalid_request', 'Compare exactly two public repository URLs.');
      const valid = names.map(parseRepository);
      return respond({ repositories: await Promise.all(valid.map(name => analyzeCached(name, github))) });
    }
    return respond(await analyzeCached(parseRepository(body.repository ?? body.url), github));
  } catch (error) {
    if (error instanceof ApiError) return respond({ error: { code: error.code, message: error.message, ...(error.retryAfter ? { retry_after: error.retryAfter, retry_at: error.retryAt, reset_at: error.resetAt, rate_limit: error.rateLimit, github_token_configured: error.tokenConfigured } : {}) } }, error.status, error.retryAfter ? { 'Retry-After': String(error.retryAfter) } : {});
    console.error('API failure', error?.name, error?.message);
    return respond({ error: { code: 'internal_error', message: 'Analysis could not be completed. Try again shortly.' } }, 500);
  }
}
