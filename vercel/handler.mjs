import { handleApi } from '../worker/api.js';

const MAX_BODY_BYTES = 4096;

function failure(res, status, code, message) {
  res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff' });
  res.end(JSON.stringify({ error: { code, message } }));
}

// Vercel's Nodejs launcher supplies standard Node IncomingMessage/ServerResponse
// objects. The existing API remains the only collection and inference code.
export default async function handler(req, res) {
  try {
    const url = new URL(req.url || '/', 'https://repopulse.internal');
    // Build Output routing can expose the rewritten URL. Preserve the original
    // endpoint through an internal capture, while also accepting original URLs.
    if (url.pathname === '/api' && url.searchParams.has('__repopulse_path')) {
      url.pathname = `/api/${url.searchParams.get('__repopulse_path') || ''}`;
      url.searchParams.delete('__repopulse_path');
    }
    const method = req.method || 'GET';
    const chunks = [];
    let size = 0;
    if (!['GET', 'HEAD'].includes(method)) {
      for await (const chunk of req) {
        const bytes = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
        size += bytes.length;
        if (size <= MAX_BODY_BYTES) chunks.push(bytes);
      }
      if (size > MAX_BODY_BYTES) return failure(res, 413, 'invalid_request', 'Request body is too large.');
    }
    const init = { method, headers: { 'Content-Type': req.headers['content-type'] || 'application/json' } };
    if (!['GET', 'HEAD'].includes(method)) init.body = Buffer.concat(chunks);
    let response = await handleApi(new Request(url, init), { GITHUB_TOKEN: process.env.GITHUB_TOKEN });
    if (url.pathname === '/api/health' && response.ok) {
      response = Response.json({ ...(await response.json()), runtime: 'nodejs', hosting: 'vercel', node_version: process.versions.node }, { status: response.status, headers: response.headers });
    }
    res.writeHead(response.status, Object.fromEntries(response.headers));
    res.end(Buffer.from(await response.arrayBuffer()));
  } catch (error) {
    console.error('Vercel adapter failure', String(error?.name || 'Error').replace(/[^A-Za-z0-9_]/g, '').slice(0, 64));
    if (!res.headersSent) failure(res, 500, 'internal_error', 'Analysis could not be completed. Try again shortly.');
    else res.end();
  }
}
