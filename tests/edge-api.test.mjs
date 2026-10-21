import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { build } from 'esbuild';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const bundle = await build({ entryPoints: [path.join(root, 'worker/api.js')], bundle: true, platform: 'node', format: 'esm', write: false });
const api = await import(`data:text/javascript;base64,${Buffer.from(bundle.outputFiles[0].text).toString('base64')}`);
const { parseRepository, extractFeatures, predictForest, GitHub, ApiError, analyze, analyzeCached, createCacheStore, handleApi } = api;
const now = Date.parse('2026-10-17T12:00:00Z');
const repository = { full_name: 'owner/project', html_url: 'https://github.com/owner/project', created_at: '2020-01-01T00:00:00Z', private: false };
const commit = days => ({ sha: String(days), author: { login: 'one' }, commit: { committer: { date: new Date(now - days * 86400000).toISOString() }, author: { email: 'person@example.com' } } });

test('strict repository validation prevents off-host URLs and normalized traversal', () => {
  for (const value of ['owner/project', 'https://github.com/owner/project.git/', 'github.com/owner/project']) assert.equal(parseRepository(value), 'owner/project');
  for (const value of [null, 'https://evil.com/owner/project', 'http://github.com/owner/project', 'https://github.com@evil.com/owner/project', 'https://github.com/owner/../project', 'https://github.com/owner/%2e%2e/project', 'https://github.com/owner/project/issues', 'owner/..', 'owner\\project', 'https://github.com/owner/project?token=secret']) assert.throws(() => parseRepository(value), ApiError);
});

test('GitHub rate limit errors carry a retry interval', async () => {
  const github = new GitHub(null, async () => new Response('{}', { status: 403, headers: { 'x-ratelimit-remaining': '0', 'retry-after': '60' } }));
  await assert.rejects(github.get('/repos/owner/project'), error => error.code === 'github_rate_limit' && error.retryAfter === 60 && error.status === 429);
});

test('native fetch keeps its global receiver in edge runtimes', async () => {
  const original = globalThis.fetch;
  globalThis.fetch = function () {
    assert.equal(this, globalThis, 'Native edge fetch requires the global receiver');
    return Promise.resolve(Response.json({ live: true }));
  };
  try {
    const [result] = await new GitHub().get('/repos/owner/project');
    assert.deepEqual(result, { live: true });
  } finally { globalThis.fetch = original; }
});

test('request cancellation also works without AbortSignal.timeout', async () => {
  const descriptor = Object.getOwnPropertyDescriptor(AbortSignal, 'timeout');
  Object.defineProperty(AbortSignal, 'timeout', { configurable: true, writable: true, value: undefined });
  try {
    const github = new GitHub(null, async (_url, options) => {
      assert.ok(options.signal instanceof AbortSignal);
      assert.equal(options.signal.aborted, false);
      return Response.json({ fallback: true });
    });
    assert.deepEqual((await github.get('/repos/owner/project'))[0], { fallback: true });
  } finally { Object.defineProperty(AbortSignal, 'timeout', descriptor); }
});

test('network diagnostics redact the configured token', async () => {
  const token = 'private-test-token', messages = [], original = console.error;
  console.error = (...args) => messages.push(args.join(' '));
  try {
    const github = new GitHub(token, async () => { throw new TypeError(`network failure ${token} Bearer ghp_exampleSECRET`); });
    await assert.rejects(github.get('/repos/owner/project'), error => error.code === 'github_unavailable');
    assert.equal(messages.length, 1);
    assert.ok(messages[0].includes('network failure'));
    assert.ok(!messages[0].includes(token));
    assert.ok(!messages[0].includes('ghp_exampleSECRET'));
  } finally { console.error = original; }
});

test('successful GET cache deduplicates requests, preserves Link, and expires in ten minutes', async () => {
  let clock = now, calls = 0;
  const store = createCacheStore(() => clock);
  const github = new GitHub('test-cache-token', async () => {
    calls++;
    await Promise.resolve();
    return Response.json({ calls }, { headers: { link: '<https://api.github.com/repos/cache/one/commits?page=2>; rel="next"' } });
  }, store);
  const results = await Promise.all(Array.from({ length: 4 }, () => github.get('/repos/cache/one')));
  assert.equal(calls, 1);
  results[0][0].calls = 999;
  clock += 599999;
  const cached = await github.get('/repos/cache/one');
  assert.equal(cached[0].calls, 1);
  assert.match(cached[1].get('link'), /rel="next"/);
  assert.equal(calls, 1);
  assert.ok([...store.entries.keys(), ...store.pending.keys()].every(key => !key.includes('test-cache-token')));
  clock++;
  assert.equal((await github.get('/repos/cache/one'))[0].calls, 2);
  assert.equal(calls, 2);
});

test('cache scopes separate credentials and all stores remain bounded', async () => {
  const store = createCacheStore(() => now);
  let calls = 0;
  const fetcher = async () => Response.json({ call: ++calls });
  const first = new GitHub('credential-one', fetcher, store), same = new GitHub('credential-one', fetcher, store), other = new GitHub('credential-two', fetcher, store);
  assert.equal((await first.get('/repos/cache/scoped'))[0].call, 1);
  assert.equal((await same.get('/repos/cache/scoped'))[0].call, 1);
  assert.equal((await other.get('/repos/cache/scoped'))[0].call, 2);
  for (let index = 0; index < 280; index++) await first.get(`/repos/cache/item-${index}`);
  assert.ok(store.entries.size <= 256);
  assert.ok(store.bytes <= 12 * 1024 * 1024);
  assert.ok([...store.entries.keys()].every(key => !/credential-(one|two)/.test(key)));
});

test('upstream failures never enter caches and rejected inflight requests clean up', async () => {
  const store = createCacheStore(() => now);
  let calls = 0;
  const github = new GitHub(null, async () => {
    calls++;
    await Promise.resolve();
    return Response.json({}, { status: 500 });
  }, store);
  const failures = await Promise.allSettled(Array.from({ length: 4 }, () => github.get('/repos/cache/failure')));
  assert.ok(failures.every(result => result.status === 'rejected'));
  assert.equal(calls, 1);
  assert.equal(store.entries.size, 0);
  assert.equal(store.pending.size, 0);
  await assert.rejects(github.get('/repos/cache/failure'), error => error.code === 'github_unavailable');
  assert.equal(calls, 2);
  assert.equal(store.pending.size, 0);
});

test('primary cooldown is shared by credential, keeps search separate, and honors reset', async () => {
  let clock = now, calls = 0;
  const store = createCacheStore(() => clock);
  const fetcher = async url => {
    calls++;
    if (url.includes('/search/')) return Response.json({ total_count: 0 });
    if (calls === 1) return Response.json({}, { status: 403, headers: { 'x-ratelimit-resource': 'core', 'x-ratelimit-limit': '60', 'x-ratelimit-remaining': '0', 'x-ratelimit-reset': String((now + 120000) / 1000), 'retry-after': '60' } });
    return Response.json({ recovered: true });
  };
  const first = new GitHub(null, fetcher, store), second = new GitHub(null, fetcher, store);
  await assert.rejects(first.get('/repos/cooldown/one'), error => error.retryAfter === 120 && error.retryAt === new Date(now + 120000).toISOString() && error.resetAt === error.retryAt);
  await assert.rejects(second.get('/repos/cooldown/two'), error => error.code === 'github_rate_limit');
  assert.equal(calls, 1);
  await second.get('/search/issues', { q: 'repo:cooldown/one is:open' });
  assert.equal(calls, 2);
  assert.equal((await second.quotaStatus()).resources.core.remaining, 0);
  clock += 120000;
  assert.deepEqual((await second.get('/repos/cooldown/two'))[0], { recovered: true });
  assert.equal(calls, 3);
});

test('secondary cooldown honors HTTP-date Retry-After across core and search with no retry loops', async () => {
  let clock = now, calls = 0;
  const store = createCacheStore(() => clock);
  const github = new GitHub(null, async () => {
    calls++;
    return Response.json({}, { status: 429, headers: { 'retry-after': new Date(now + 60000).toUTCString(), 'x-ratelimit-remaining': '42', 'x-ratelimit-resource': 'core' } });
  }, store);
  await assert.rejects(github.get('/repos/cooldown/secondary'), error => error.retryAfter === 60);
  await assert.rejects(github.get('/search/issues'), error => error.retryAfter === 60);
  assert.equal(calls, 1);
  clock += 30000;
  await assert.rejects(github.get('/repos/cooldown/other'), error => error.retryAfter === 30);
  assert.equal(calls, 1);
});

test('invalid GitHub credentials have a distinct error and are not cached', async () => {
  let calls = 0;
  const github = new GitHub('invalid-test-credential', async () => { calls++; return Response.json({}, { status: 401 }); });
  for (let index = 0; index < 2; index++) await assert.rejects(github.get('/repos/cache/auth'), error => error.code === 'github_authentication' && error.status === 503);
  assert.equal(calls, 2);
  assert.equal(github.store.entries.size, 0);
});

function analysisTransport(counter) {
  return async value => {
    counter.calls++;
    const url = new URL(value), route = url.pathname;
    let result;
    if (route === '/graphql') {
      result = {
        data: {
          repository: {
            defaultBranchRef: {
              target: {
                history90d: { totalCount: 2 },
                history6m: {
                  totalCount: 2,
                  pageInfo: { hasNextPage: false, endCursor: null },
                  nodes: [
                    { oid: '2', committedDate: new Date(now - 2 * 86400000).toISOString(), author: { name: 'one', email: 'person@example.com', user: { login: 'one' } } },
                    { oid: '40', committedDate: new Date(now - 40 * 86400000).toISOString(), author: { name: 'one', email: 'person@example.com', user: { login: 'one' } } },
                  ],
                },
              },
            },
          },
        },
      };
    } else if (route === '/search/issues') result = { total_count: 1, incomplete_results: false };
    else if (route.endsWith('/commits')) result = [commit(2), commit(40)];
    else if (route.endsWith('/issues') || route.endsWith('/pulls') || route.endsWith('/releases')) result = [];
    else {
      const name = route.replace(/^\/repos\//, '');
      result = { ...repository, full_name: name, html_url: `https://github.com/${name}` };
    }
    return Response.json(result, { headers: { 'x-ratelimit-resource': route === '/search/issues' ? 'search' : 'core', 'x-ratelimit-remaining': '50', 'x-ratelimit-limit': '60', 'x-ratelimit-reset': String(Math.floor(Date.now() / 1000) + 3600) } });
  };
}

test('analysis cache preserves original observations, expires, and deduplicates compare-style requests', async () => {
  let clock = now;
  const counter = { calls: 0 }, store = createCacheStore(() => clock);
  const github = new GitHub(null, analysisTransport(counter), store);
  const first = await analyzeCached('cache/result', github, now);
  assert.equal(counter.calls, 7);
  assert.equal(first.coverage.cached, false);
  clock += 60000;
  const second = await analyzeCached('https://github.com/cache/result', new GitHub(null, analysisTransport(counter), store), now + 60000);
  assert.equal(counter.calls, 7);
  assert.equal(second.coverage.cached, true);
  assert.equal(second.coverage.observed_at, first.coverage.observed_at);
  assert.equal(second.prediction.predicted_at, first.prediction.predicted_at);
  assert.equal(second.coverage.cache_expires_at, first.coverage.cache_expires_at);
  const next = new GitHub(null, analysisTransport(counter), store);
  await Promise.all([analyzeCached('cache/compare', next, now), analyzeCached('cache/compare', next, now)]);
  assert.equal(counter.calls, 14);
  clock = now + 600000;
  await analyzeCached('cache/result', new GitHub(null, analysisTransport(counter), store), clock);
  assert.equal(counter.calls, 21);
});

test('optional search failures are not result-cached and retries reuse successful commit observations', async () => {
  let clock = now;
  const store = createCacheStore(() => clock), counter = { calls: 0 }, base = analysisTransport(counter);
  const fetcher = async value => {
    if (new URL(value).pathname === '/search/issues') {
      counter.calls++;
      return Response.json({}, { status: 403, headers: { 'x-ratelimit-resource': 'search', 'x-ratelimit-remaining': '0', 'x-ratelimit-reset': String((now + 60000) / 1000) } });
    }
    return base(value);
  };
  const first = await analyzeCached('cache/partial', new GitHub(null, fetcher, store), clock);
  assert.equal(first.coverage.cacheable, false);
  assert.equal(first.repository.open_issues, null);
  const callsAfterFirst = counter.calls;
  clock += 1000;
  const second = await analyzeCached('cache/partial', new GitHub(null, fetcher, store), clock);
  assert.equal(counter.calls, callsAfterFirst, 'Successful core GETs should be reused and search should remain in cooldown');
  assert.equal(second.coverage.cacheable, false);
  assert.equal(second.coverage.cached, true);
  assert.equal(second.coverage.data_observed_at, first.coverage.data_observed_at);
  assert.ok([...store.entries.keys()].every(key => !key.includes('|analysis|')));
});

test('API analysis/compare reuse real successful results and health never spends GitHub quota', async () => {
  const original = globalThis.fetch, counter = { calls: 0 }, env = { GITHUB_TOKEN: 'api-result-cache-test-only' };
  globalThis.fetch = analysisTransport(counter);
  const post = (endpoint, body) => handleApi(new Request(`https://app.test/api/${endpoint}`, { method: 'POST', body: JSON.stringify(body) }), env);
  try {
    assert.equal((await post('analyze', { url: 'cache/api-one' })).status, 200);
    assert.equal(counter.calls, 7);
    const cached = await (await post('analyze', { url: 'cache/api-one' })).json();
    assert.equal(cached.coverage.cached, true);
    assert.equal(counter.calls, 7);
    assert.equal((await post('compare', { urls: ['cache/api-one', 'cache/api-two'] })).status, 200);
    assert.equal(counter.calls, 14);
    assert.equal((await post('compare', { urls: ['cache/api-three', 'cache/api-three'] })).status, 200);
    assert.equal(counter.calls, 21);
    const health = await (await handleApi(new Request('https://app.test/api/health'), env)).json();
    assert.equal(counter.calls, 21);
    assert.equal(health.rate_limit.credential_mode, 'authenticated');
    assert.equal(health.rate_limit.resources.core.remaining, 50);
  } finally { globalThis.fetch = original; }
});

test('missing, invalid, or future repository ages cannot generate a score', () => {
  for (const created_at of [null, 'not-a-date', '2027-01-01T00:00:00Z']) assert.throws(() => extractFeatures([commit(2)], { created_at }, now), error => error.code === 'insufficient_data');
});

test('six-month chart collection separates issues from PRs and counts merges', async () => {
  const github = new GitHub(null, async value => {
    const url = new URL(value), route = url.pathname;
    let result;
    if (route === '/repos/owner/project') result = repository;
    else if (route === '/search/issues') result = { total_count: 2, incomplete_results: false };
    else if (route.endsWith('/commits')) result = [commit(2), commit(40)];
    else if (route.endsWith('/issues')) result = [{ created_at: '2026-10-01T00:00:00Z', closed_at: '2026-10-05T00:00:00Z' }, { created_at: '2026-10-01T00:00:00Z', pull_request: {} }];
    else if (route.endsWith('/pulls')) result = [{ created_at: '2026-10-01T00:00:00Z', merged_at: '2026-10-05T00:00:00Z' }];
    else if (route.endsWith('/releases')) result = [];
    else throw new Error(route);
    return Response.json(result);
  });
  const result = await analyze('owner/project', github, now);
  assert.equal(result.coverage.prediction_complete, true);
  assert.equal(result.activity.commits.length, 6);
  assert.equal(result.activity.issues.at(-1).opened, 1);
  assert.equal(result.activity.pull_requests.at(-1).merged, 1);
  assert.ok(result.prediction.probability >= 0 && result.prediction.probability <= 1);
  assert.match(result.prediction.explanation, /183 days/);
});

test('truncated history withholds scores and leaves unknown chart values blank', async () => {
  const github = {
    get: async route => [route.startsWith('/repos/') ? repository : { total_count: 0 }, new Headers()],
    pages: async (_name, resource) => ({ items: resource === 'commits' ? [commit(2), commit(120)] : [], truncated: resource === 'commits', pages: 5, limit: 500 }),
  };
  const result = await analyze('owner/project', github, now);
  assert.equal(result.prediction, null);
  assert.equal(result.coverage.prediction_complete, false);
  assert.equal(result.activity.commits.at(-1).commits, null);
  assert.match(result.prediction_error, /collection cap/);
});

test('GraphQL commit collection collects commits and removes pagination-cap failure when 90d is complete', async () => {
  const store = createCacheStore(() => now);
  const graphqlData = {
    repository: {
      defaultBranchRef: {
        target: {
          history90d: { totalCount: 2 },
          history6m: {
            totalCount: 650,
            pageInfo: { hasNextPage: false, endCursor: null },
            nodes: [
              { oid: 'sha-1', committedDate: new Date(now - 2 * 86400000).toISOString(), author: { name: 'Alice', email: 'alice@example.com', user: { login: 'alice' } } },
              { oid: 'sha-2', committedDate: new Date(now - 15 * 86400000).toISOString(), author: { name: 'Bob', email: 'bob@example.com', user: { login: 'bob' } } },
            ],
          },
        },
      },
    },
  };
  const github = new GitHub('test-token', async (value, options) => {
    const url = new URL(value);
    if (url.pathname === '/graphql') {
      return Response.json({ data: graphqlData });
    }
    if (url.pathname === '/repos/owner/project') return Response.json(repository);
    if (url.pathname === '/search/issues') return Response.json({ total_count: 2, incomplete_results: false });
    if (url.pathname.endsWith('/issues')) return Response.json([]);
    if (url.pathname.endsWith('/pulls')) return Response.json([]);
    if (url.pathname.endsWith('/releases')) return Response.json([]);
    return Response.json([]);
  }, store);
  const result = await analyze('owner/project', github, now);
  assert.ok(result.prediction !== null);
  assert.equal(result.coverage.prediction_complete, true);
  assert.equal(result.features.commits_30d, 2);
  assert.equal(result.features.contributors_90d, 2);
});

test('GraphQL commit collection falls back to prediction: null when 90d exceeds collection cap', async () => {
  const store = createCacheStore(() => now);
  const graphqlData = {
    repository: {
      defaultBranchRef: {
        target: {
          history90d: { totalCount: 1500 },
          history6m: {
            totalCount: 2500,
            pageInfo: { hasNextPage: false, endCursor: null },
            nodes: [
              { oid: 'sha-1', committedDate: new Date(now - 2 * 86400000).toISOString(), author: { name: 'Alice', email: 'alice@example.com', user: { login: 'alice' } } },
            ],
          },
        },
      },
    },
  };
  const github = new GitHub('test-token', async value => {
    const url = new URL(value);
    if (url.pathname === '/graphql') return Response.json({ data: graphqlData });
    if (url.pathname === '/repos/owner/project') return Response.json(repository);
    if (url.pathname === '/search/issues') return Response.json({ total_count: 0 });
    return Response.json([]);
  }, store);
  const result = await analyze('owner/project', github, now);
  assert.equal(result.prediction, null);
  assert.equal(result.coverage.prediction_complete, false);
  assert.match(result.prediction_error, /collection cap/);
});

test('GraphQL failure falls back to REST commits collection', async () => {
  const store = createCacheStore(() => now);
  const github = new GitHub('test-token', async value => {
    const url = new URL(value);
    if (url.pathname === '/graphql') return Response.json({ errors: [{ message: 'GraphQL error' }] }, { status: 200 });
    if (url.pathname === '/repos/owner/project') return Response.json(repository);
    if (url.pathname === '/repos/owner/project/commits') return Response.json([commit(2), commit(10)]);
    if (url.pathname === '/search/issues') return Response.json({ total_count: 0 });
    return Response.json([]);
  }, store);
  const result = await analyze('owner/project', github, now);
  assert.ok(result.prediction !== null);
  assert.equal(result.features.commits_30d, 2);
});

test('API method, body, count, and URL errors return readable JSON', async () => {
  assert.equal((await handleApi(new Request('https://app.test/api/analyze'))).status, 405);
  assert.equal((await handleApi(new Request('https://app.test/api/compare', { method: 'POST', body: '{"urls": ["owner/project"]}' }))).status, 400);
  assert.equal((await handleApi(new Request('https://app.test/api/analyze', { method: 'POST', body: 'invalid' }))).status, 400);
  assert.equal((await handleApi(new Request('https://app.test/api/analyze', { method: 'POST', body: '{"url":"https://evil.com/a/b"}' }))).status, 400);
  const response = await handleApi(new Request('https://app.test/api/health'));
  assert.equal(response.status, 200);
  assert.equal((await response.json()).status, 'ok');
});

test('JavaScript features and actual forest probabilities match shared Python and sklearn', () => {
  const bundledPython = path.join(process.env.USERPROFILE ?? '', '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe');
  const python = process.env.PYTHON_BIN || (existsSync(bundledPython) ? bundledPython : 'python');
  const result = spawnSync(python, ['tests/backend_parity.py'], { cwd: root, encoding: 'utf8', maxBuffer: 10 * 1024 * 1024 });
  assert.equal(result.status, 0, result.stderr);
  const data = JSON.parse(result.stdout);
  const fixture = data.feature_fixture;
  assert.deepEqual(extractFeatures(fixture.commits, fixture.repository, Date.parse(fixture.now)), data.feature_expected);
  const errors = data.features.map((features, index) => Math.abs(predictForest(features) - data.expected[index]));
  assert.ok(Math.max(...errors) < 1e-12, `Maximum probability error ${Math.max(...errors)}`);
  assert.ok(errors.length > 100);
});
