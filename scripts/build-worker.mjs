import { build } from 'esbuild';
import { readdir, readFile, mkdir, writeFile, copyFile } from 'node:fs/promises';
import { join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = new URL('../', import.meta.url);
const directory = fileURLToPath(new URL('frontend/dist/', root));
const assets = {};
const types = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.svg': 'image/svg+xml', '.woff2': 'font/woff2', '.ico': 'image/x-icon', '.png': 'image/png' };
async function gather(dir) {
  for (const item of await readdir(dir, { withFileTypes: true })) {
    const path = join(dir, item.name);
    if (item.isDirectory()) await gather(path);
    else {
      const ext = item.name.slice(item.name.lastIndexOf('.'));
      assets['/' + relative(directory, path).replaceAll('\\', '/')] = { type: types[ext] || 'application/octet-stream', body: (await readFile(path)).toString('base64') };
    }
  }
}
await gather(directory);
await mkdir(new URL('.generated/', root), { recursive: true });
await writeFile(new URL('.generated/assets.js', root), `export default ${JSON.stringify(assets)};\n`);
await mkdir(new URL('dist/server/', root), { recursive: true });
await mkdir(new URL('dist/.openai/', root), { recursive: true });
await build({ entryPoints: [fileURLToPath(new URL('worker/index.js', root))], outfile: fileURLToPath(new URL('dist/server/index.js', root)), bundle: true, format: 'esm', platform: 'browser', target: 'es2022', minify: true });
await copyFile(new URL('.openai/hosting.json', root), new URL('dist/.openai/hosting.json', root));
console.log(`Built RepoPulse Worker with ${Object.keys(assets).length} embedded frontend assets.`);
