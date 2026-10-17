import assets from '../.generated/assets.js';
import { handleApi } from './api.js';

export default {
  async fetch(request, env = {}, ctx = {}) {
    const url = new URL(request.url);
    if (url.pathname.startsWith('/api/')) return handleApi(request, env, ctx);
    if (!['GET', 'HEAD'].includes(request.method)) return new Response('Method not allowed', { status: 405 });
    const asset = assets[url.pathname] || (!url.pathname.split('/').at(-1).includes('.') ? assets['/index.html'] : null);
    if (!asset) return new Response('Not found', { status: 404 });
    const bytes = Uint8Array.from(atob(asset.body), c => c.charCodeAt(0));
    return new Response(request.method === 'HEAD' ? null : bytes, { headers: {
      'Content-Type': asset.type,
      'Cache-Control': url.pathname.startsWith('/assets/') ? 'public, max-age=31536000, immutable' : 'no-cache',
      'X-Content-Type-Options': 'nosniff',
      'Referrer-Policy': 'strict-origin-when-cross-origin',
      'Permissions-Policy': 'camera=(), microphone=(), geolocation=()',
    } });
  },
};
