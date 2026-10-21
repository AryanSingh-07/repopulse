import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { createRequire } from 'node:module';
import { readFile, readdir } from 'node:fs/promises';
import { join } from 'node:path';
import { buildVercel, outputDirectory } from '../scripts/build-vercel.mjs';

await buildVercel();
const require = createRequire(import.meta.url);
const handler = require(join(outputDirectory, 'functions/api.func/index.cjs'));
const info = JSON.parse(await readFile(new URL('../model/model-info.json', import.meta.url), 'utf8'));

async function withServer(operation) {
  assert.equal(typeof handler, 'function', 'Build Output Nodejs launcher must receive a req/res function');
  const server = http.createServer(handler);
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const request = (path, { method = 'GET', body, headers = {} } = {}) => new Promise((resolve, reject) => {
    const client = http.request({ host: '127.0.0.1', port: server.address().port, path, method, headers: { 'Content-Type': 'application/json', ...headers } }, response => {
      const chunks = [];
      response.on('data', chunk => chunks.push(chunk));
      response.on('end', () => resolve({ status: response.statusCode, headers: response.headers, body: JSON.parse(Buffer.concat(chunks).toString('utf8')) }));
    });
    client.on('error', reject);
    client.end(body);
  });
  try { await operation(request); }
  finally { await new Promise(resolve => server.close(resolve)); }
}

test('Vercel output has a Node22 function, real forest, static frontend, and safe API routing', async () => {
  const config = JSON.parse(await readFile(join(outputDirectory, 'config.json'), 'utf8'));
  const runtime = JSON.parse(await readFile(join(outputDirectory, 'functions/api.func/.vc-config.json'), 'utf8'));
  assert.equal(config.version, 3);
  assert.equal(runtime.runtime, 'nodejs22.x');
  assert.equal(runtime.launcherType, 'Nodejs');
  assert.equal(runtime.shouldAddHelpers, false);
  assert.equal(runtime.handler, 'index.cjs');
  const apiRule = config.routes.find(route => route.dest?.startsWith('/api?'));
  assert.ok(new RegExp(apiRule.src).test('/api/analyze'));
  assert.equal('/api/health'.replace(new RegExp(apiRule.src), apiRule.dest), '/api?__repopulse_path=health');
  const fallback = config.routes.at(-1);
  assert.ok(new RegExp(`^${fallback.src}$`).test('/compare'));
  assert.ok(!new RegExp(`^${fallback.src}$`).test('/api/unknown'));
  assert.ok(!new RegExp(`^${fallback.src}$`).test('/missing.png'));
  const bundle = await readFile(join(outputDirectory, 'functions/api.func/index.cjs'), 'utf8');
  assert.ok(bundle.includes('children_left'));
  assert.ok(bundle.includes(info.version));
  assert.ok(bundle.includes('process.env.GITHUB_TOKEN'));
  assert.ok(!bundle.includes('api.openai.com'));
  const files = await readdir(join(outputDirectory, 'static/assets'));
  assert.ok(files.some(file => file.endsWith('.js')));
  assert.match(await readFile(join(outputDirectory, 'static/index.html'), 'utf8'), /<div id="root"/);
});

test('real Node HTTP requests reach the existing API and trained forest without any GitHub network calls', async () => {
  const originalFetch = globalThis.fetch, originalToken = process.env.GITHUB_TOKEN;
  process.env.GITHUB_TOKEN = 'vercel-integration-test-token';
  let calls = 0;
  const now = Date.now();
  globalThis.fetch = async (value, options) => {
    calls++;
    const url = new URL(value), path = url.pathname;
    assert.equal(url.origin, 'https://api.github.com');
    assert.equal(options.headers.Authorization, 'Bearer vercel-integration-test-token');
    let data;
    if (path === '/graphql') data = { data: { repository: { defaultBranchRef: { target: { history90d: { totalCount: 2 }, history6m: { totalCount: 2, pageInfo: { hasNextPage: false, endCursor: null }, nodes: [2, 40].map(days => ({ oid: `commit-${days}`, committedDate: new Date(now - days * 86400000).toISOString(), author: { name: 'maintainer', email: 'maintainer@example.com', user: { login: 'maintainer' } } })) } } } } } };
    else if (path === '/repos/vercel-test/project') data = { full_name: 'vercel-test/project', html_url: 'https://github.com/vercel-test/project', created_at: '2020-01-01T00:00:00Z', private: false, language: 'JavaScript', stargazers_count: 12, forks_count: 3 };
    else if (path.endsWith('/commits')) data = [2, 40].map(days => ({ sha: `commit-${days}`, author: { login: 'maintainer' }, commit: { committer: { date: new Date(now - days * 86400000).toISOString() }, author: { email: 'maintainer@example.com' } } }));
    else if (path === '/search/issues') data = { total_count: 0, incomplete_results: false };
    else if (/\/(issues|pulls|releases)$/.test(path)) data = [];
    else throw new Error('Unexpected test GitHub endpoint');
    return Response.json(data);
  };
  try {
    await withServer(async request => {
      const health = await request('/api/health');
      assert.equal(health.status, 200);
      assert.equal(health.body.runtime, 'nodejs');
      assert.equal(health.body.hosting, 'vercel');
      assert.equal(health.body.model_version, info.version);
      assert.equal(calls, 0);
      assert.equal((await request('/api?__repopulse_path=model-info')).body.horizon_days, 183);
      const result = await request('/api?__repopulse_path=analyze', { method: 'POST', body: JSON.stringify({ url: 'vercel-test/project' }) });
      assert.equal(result.status, 200);
      assert.equal(result.body.prediction.model_version, info.version);
      assert.equal(result.body.features.commits_30d, 1);
      assert.equal(result.body.features.commits_90d, 2);
      assert.ok(result.body.prediction.probability >= 0 && result.body.prediction.probability <= 1);
      assert.equal(calls, 7);
      const cached = await request('/api/analyze', { method: 'POST', body: JSON.stringify({ url: 'vercel-test/project' }) });
      assert.equal(cached.status, 200);
      assert.equal(cached.body.coverage.cached, true);
      assert.equal(calls, 7);
      assert.equal((await request('/api/analyze', { method: 'POST', body: 'invalid-json' })).status, 400);
      assert.equal((await request('/api/analyze', { method: 'POST', body: 'x'.repeat(4097) })).status, 413);
      assert.equal((await request('/api/missing')).status, 404);
      assert.equal(calls, 7);
    });
  } finally {
    globalThis.fetch = originalFetch;
    if (originalToken === undefined) delete process.env.GITHUB_TOKEN; else process.env.GITHUB_TOKEN = originalToken;
  }
});

test('Node adapter preserves rate-limit status, reset metadata, and Retry-After headers', async () => {
  const originalFetch = globalThis.fetch, originalToken = process.env.GITHUB_TOKEN;
  process.env.GITHUB_TOKEN = 'vercel-rate-limit-test-token';
  let calls = 0;
  globalThis.fetch = async () => {
    calls++;
    return Response.json({}, { status: 403, headers: { 'x-ratelimit-resource': 'core', 'x-ratelimit-remaining': '0', 'x-ratelimit-reset': String(Math.floor(Date.now() / 1000) + 120) } });
  };
  try {
    await withServer(async request => {
      const result = await request('/api/analyze', { method: 'POST', body: JSON.stringify({ url: 'vercel-test/rate-limited' }) });
      assert.equal(result.status, 429);
      assert.equal(result.body.error.code, 'github_rate_limit');
      assert.ok(Number(result.headers['retry-after']) > 0);
      assert.ok(Date.parse(result.body.error.retry_at) > Date.now());
      assert.ok(result.body.error.reset_at);
      const second = await request('/api/analyze', { method: 'POST', body: JSON.stringify({ url: 'vercel-test/another' }) });
      assert.equal(second.status, 429);
      assert.equal(calls, 1);
    });
  } finally {
    globalThis.fetch = originalFetch;
    if (originalToken === undefined) delete process.env.GITHUB_TOKEN; else process.env.GITHUB_TOKEN = originalToken;
  }
});
