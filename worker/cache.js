// Process-local, bounded caches. Only successful public GitHub observations enter
// them. Expired observations are removed, never served as stale fallback data.
export const CACHE_TTL_MS = 10 * 60 * 1000;
const MAX_ENTRIES = 256;
const MAX_BYTES = 12 * 1024 * 1024;

export function createCacheStore(clock = () => Date.now()) {
  return { clock, entries: new Map(), pending: new Map(), quotas: new Map(), cooldowns: new Map(), bytes: 0 };
}

export const sharedCache = createCacheStore();

export async function credentialNamespace(token) {
  if (!token) return 'anonymous';
  const bytes = await globalThis.crypto.subtle.digest('SHA-256', new TextEncoder().encode(token));
  return `authenticated:${Array.from(new Uint8Array(bytes), byte => byte.toString(16).padStart(2, '0')).join('')}`;
}

function remove(store, key) {
  const old = store.entries.get(key);
  if (old) { store.bytes -= old.bytes; store.entries.delete(key); }
}

export function readCache(store, key) {
  const entry = store.entries.get(key);
  if (!entry) return null;
  if (entry.expiresAt <= store.clock()) { remove(store, key); return null; }
  store.entries.delete(key); store.entries.set(key, entry);
  return { ...entry, value: structuredClone(entry.value) };
}

export function writeCache(store, key, value, ttl = CACHE_TTL_MS) {
  const bytes = JSON.stringify(value).length * 2;
  if (bytes > MAX_BYTES) return;
  for (const [oldKey, entry] of store.entries) if (entry.expiresAt <= store.clock()) remove(store, oldKey);
  remove(store, key);
  while (store.entries.size >= MAX_ENTRIES || store.bytes + bytes > MAX_BYTES) remove(store, store.entries.keys().next().value);
  const observedAt = store.clock();
  store.entries.set(key, { value: structuredClone(value), observedAt, expiresAt: observedAt + ttl, bytes });
  store.bytes += bytes;
}

export async function deduplicate(store, key, operation) {
  if (store.pending.has(key)) return structuredClone(await store.pending.get(key));
  const pending = operation();
  store.pending.set(key, pending);
  try { return await pending; }
  finally { if (store.pending.get(key) === pending) store.pending.delete(key); }
}

const time = value => new Date(value).toISOString();
const keyFor = (namespace, resource) => `${namespace}|${resource}`;
const numberHeader = (headers, key) => {
  const raw = headers.get(key), value = raw === null ? NaN : Number(raw);
  return Number.isFinite(value) && value >= 0 ? value : null;
};

export function observeQuota(store, namespace, fallbackResource, headers) {
  const resource = headers.get('x-ratelimit-resource') || fallbackResource;
  const remaining = numberHeader(headers, 'x-ratelimit-remaining');
  if (remaining === null) return null;
  const reset = numberHeader(headers, 'x-ratelimit-reset');
  const snapshot = {
    resource, limit: numberHeader(headers, 'x-ratelimit-limit'), remaining,
    used: numberHeader(headers, 'x-ratelimit-used'),
    reset_at: reset ? time(reset * 1000) : null, observed_at: time(store.clock()),
  };
  const key = keyFor(namespace, resource), previous = store.quotas.get(key);
  if (previous?.reset_at === snapshot.reset_at && previous.remaining < remaining) snapshot.remaining = previous.remaining;
  store.quotas.set(key, snapshot);
  if (snapshot.remaining === 0) setCooldown(store, namespace, resource, reset && reset * 1000 > store.clock() ? reset * 1000 : store.clock() + 60000);
  return snapshot;
}

export function setCooldown(store, namespace, resource, until) {
  const key = keyFor(namespace, resource);
  store.cooldowns.set(key, Math.max(store.cooldowns.get(key) || 0, until));
}

export function cooldownUntil(store, namespace, resource) {
  let latest = 0;
  for (const scope of ['*', resource]) {
    const key = keyFor(namespace, scope), until = store.cooldowns.get(key) || 0;
    if (until <= store.clock()) store.cooldowns.delete(key); else latest = Math.max(latest, until);
  }
  return latest;
}

export function rateLimitStatus(store, namespace, resource) {
  const resources = {};
  for (const [key, snapshot] of store.quotas) if (key.startsWith(`${namespace}|`)) resources[snapshot.resource] = { ...snapshot };
  const cooldown = resource ? cooldownUntil(store, namespace, resource) : Math.max(cooldownUntil(store, namespace, 'core'), cooldownUntil(store, namespace, 'search'));
  const result = {
    credential_mode: namespace === 'anonymous' ? 'anonymous' : 'authenticated',
    cooldown_until: cooldown ? time(cooldown) : null,
    ...(resource ? (resources[resource] || { resource }) : { resources }),
  };
  return result;
}

export function retryTime(headers, now) {
  const raw = headers.get('retry-after');
  const reset = Number(headers.get('x-ratelimit-reset')) * 1000;
  const retry = raw && /^\d+$/.test(raw.trim()) ? now + Number(raw.trim()) * 1000 : raw ? Date.parse(raw) : 0;
  const primary = headers.get('x-ratelimit-remaining') === '0' && reset > now ? reset : 0;
  return Math.max(Number.isFinite(retry) && retry > now ? retry : 0, primary) || now + 60000;
}
