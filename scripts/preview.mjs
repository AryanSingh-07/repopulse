import http from 'node:http';
import { readFile, stat } from 'node:fs/promises';
const workerUrl = new URL('../dist/server/index.js', import.meta.url);
let loadedAt = (await stat(workerUrl)).mtimeMs;
let worker = (await import(workerUrl)).default;
try {
  const local = await readFile(new URL('../.env', import.meta.url), 'utf8');
  for (const line of local.split(/\r?\n/)) {
    const match = line.match(/^([A-Z_]+)=(.*)$/);
    if (match && !process.env[match[1]]) process.env[match[1]] = match[2].replace(/^['"]|['"]$/g, '');
  }
} catch {}
const port = Number(process.env.PORT || 4173);
http.createServer(async (req, res) => {
  try {
    const modified = (await stat(workerUrl)).mtimeMs;
    if (modified !== loadedAt) {
      worker = (await import(`${workerUrl.href}?build=${modified}`)).default;
      loadedAt = modified;
    }
    const buffers = [];
    for await (const chunk of req) buffers.push(chunk);
    const init = { method: req.method, headers: req.headers };
    if (!['GET', 'HEAD'].includes(req.method)) init.body = Buffer.concat(buffers);
    const response = await worker.fetch(new Request(`http://localhost:${port}${req.url}`, init), process.env, { waitUntil: p => p.catch(() => {}) });
    res.writeHead(response.status, Object.fromEntries(response.headers));
    res.end(Buffer.from(await response.arrayBuffer()));
  } catch (err) {
    console.error(err.message);
    res.writeHead(500, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { message: 'Preview server failed. Check the terminal.' } }));
  }
}).listen(port, '127.0.0.1', () => console.log(`RepoPulse preview: http://localhost:${port}`));
