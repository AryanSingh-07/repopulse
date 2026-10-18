import { build } from 'esbuild';
import { cp, lstat, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { dirname, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
export const outputDirectory = join(root, '.vercel', 'output');

export async function buildVercel() {
  const frontend = join(root, 'frontend', 'dist');
  await readFile(join(frontend, 'index.html'));
  const forest = JSON.parse(await readFile(join(root, 'model', 'forest.json'), 'utf8'));
  const info = JSON.parse(await readFile(join(root, 'model', 'model-info.json'), 'utf8'));
  if (!forest.trees?.length || forest.feature_names?.length !== 7 || forest.model_version !== info.version) throw new Error('A matching trained forest and model card are required.');

  // Remove only this build's known output folder, preserving CLI project links.
  if (!outputDirectory.startsWith(root + sep) || relative(root, outputDirectory) !== join('.vercel', 'output')) throw new Error('Unsafe Vercel output path.');
  for (const target of [join(root, '.vercel'), outputDirectory]) {
    const entry = await lstat(target).catch(error => { if (error.code !== 'ENOENT') throw error; return null; });
    if (entry?.isSymbolicLink()) throw new Error('Vercel output paths must not be symbolic links.');
  }
  await rm(outputDirectory, { recursive: true, force: true });
  const functionDirectory = join(outputDirectory, 'functions', 'api.func');
  await mkdir(functionDirectory, { recursive: true });
  await cp(frontend, join(outputDirectory, 'static'), { recursive: true });
  await build({
    entryPoints: [join(root, 'vercel', 'handler.mjs')], outfile: join(functionDirectory, 'index.cjs'),
    bundle: true, platform: 'node', target: 'node22', format: 'cjs', minify: true,
    // The Nodejs launcher expects module.exports to be the HTTP handler.
    footer: { js: 'module.exports = module.exports.default;' },
  });
  await writeFile(join(functionDirectory, '.vc-config.json'), JSON.stringify({ runtime: 'nodejs22.x', handler: 'index.cjs', launcherType: 'Nodejs', maxDuration: 60, shouldAddHelpers: false }, null, 2) + '\n');
  await writeFile(join(outputDirectory, 'config.json'), JSON.stringify({
    version: 3,
    routes: [
      { src: '/(.*)', headers: { 'X-Content-Type-Options': 'nosniff', 'Referrer-Policy': 'strict-origin-when-cross-origin', 'Permissions-Policy': 'camera=(), microphone=(), geolocation=()' }, continue: true },
      { src: '^/api(?:/(.*))?$', dest: '/api?__repopulse_path=$1' },
      { src: '/assets/(.*)', headers: { 'Cache-Control': 'public, max-age=31536000, immutable' }, continue: true },
      { handle: 'filesystem' },
      { src: '^/((?!api(?:/|$))[^.]*)$', dest: '/index.html' },
    ],
  }, null, 2) + '\n');
  console.log(`Built Vercel Node.js22 function with ${forest.trees.length} trained trees and static frontend.`);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await buildVercel();
