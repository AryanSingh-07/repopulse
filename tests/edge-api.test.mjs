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
const { parseRepository, extractFeatures, predictForest, GitHub, ApiError, analyze, handleApi } = api;
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
