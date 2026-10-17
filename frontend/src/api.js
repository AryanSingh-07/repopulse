const CACHE_TTL = 10 * 60 * 1000;
const CACHE_KEY = 'repopulse-analysis-cache-v1';
const COOLDOWN_KEY = 'repopulse-github-cooldown-v1';
const cachedAnalyses = new Map();
const cooldownListeners = new Set();
let cooldown = null;

function readSession(key) { try { return JSON.parse(globalThis.sessionStorage?.getItem(key) || 'null'); } catch { return null; } }
function writeSession(key, value) { try { if(value==null)globalThis.sessionStorage?.removeItem(key);else globalThis.sessionStorage?.setItem(key, JSON.stringify(value)); } catch { /* In-memory cache works when session storage is unavailable. */ } }
for (const [key, entry] of Object.entries(readSession(CACHE_KEY)||{})) if (entry?.expiresAt > Date.now()) cachedAnalyses.set(key, entry);
cooldown = readSession(COOLDOWN_KEY);

export function normalizeRepository(input) {
  const value = String(input || '').trim();
  const match = value.match(/^(?:https?:\/\/(?:www\.)?github\.com\/)?([a-zA-Z0-9](?:[a-zA-Z0-9-]{0,38}))\/([a-zA-Z0-9_.-]+)(?:\.git)?\/?$/);
  if (!match || match[2] === '.' || match[2] === '..') throw new Error('Enter a public GitHub repository, like facebook/react or its GitHub URL.');
  return `${match[1]}/${match[2].replace(/\.git$/, '')}`;
}
function cacheKey(repository) { return normalizeRepository(repository).toLowerCase(); }
function cacheableAnalysis(data) {return data?.coverage?.cacheable!==false&&!Object.values(data?.coverage?.collections||{}).some(collection=>collection?.error)&&data?.repository?.open_issues!==null&&data?.repository?.open_pull_requests!==null;}
function persistCache() {
  for(const [key,entry] of cachedAnalyses)if(entry.expiresAt<=Date.now())cachedAnalyses.delete(key);
  while(cachedAnalyses.size>12)cachedAnalyses.delete(cachedAnalyses.keys().next().value);
  writeSession(CACHE_KEY,Object.fromEntries(cachedAnalyses));
}
function getCached(repository) {
  const key = cacheKey(repository); const entry = cachedAnalyses.get(key);
  if (!entry) return null;
  if (entry.expiresAt <= Date.now()||!cacheableAnalysis(entry.data)) { cachedAnalyses.delete(key); persistCache(); return null; }
  return {...entry.data, _clientCache: {hit:true,cached_at:new Date(entry.storedAt).toISOString(),expires_at:new Date(entry.expiresAt).toISOString()}};
}
export function hasCachedAnalysis(repository) { try { return !!getCached(repository); } catch { return false; } }
function saveAnalysis(repository, data) {
  if (!data?.repository?.full_name||!cacheableAnalysis(data)) return;
  const entry = {data, storedAt:Date.now(), expiresAt:Date.now()+CACHE_TTL};
  const serverExpiry = Date.parse(data.coverage?.cache_expires_at||'');
  if (Number.isFinite(serverExpiry)) entry.expiresAt = Math.min(entry.expiresAt, serverExpiry);
  cachedAnalyses.set(cacheKey(repository), entry); cachedAnalyses.set(cacheKey(data.repository.full_name), entry); persistCache();
}
export function getRateLimit() {
  if (!cooldown?.retryAtMs || cooldown.retryAtMs<=Date.now()) { if(cooldown){cooldown=null;writeSession(COOLDOWN_KEY,null);} return null; }
  return {...cooldown, remainingSeconds:Math.max(0,Math.ceil((cooldown.retryAtMs-Date.now())/1000))};
}
export function subscribeRateLimit(listener) {cooldownListeners.add(listener);return()=>cooldownListeners.delete(listener);}
function setCooldown(error) {cooldown={message:error.message,code:error.code,retryAt:error.retryAt,retryAtMs:error.retryAtMs,rateLimited:true,credentialMode:error.credentialMode};writeSession(COOLDOWN_KEY,cooldown);for(const listener of cooldownListeners)listener();}
function assertAllowance() {const limit=getRateLimit();if(limit){const error=Object.assign(new Error(limit.message),limit);throw error;}}
function timestamp(value) {if(value==null||value==='')return NaN; if(typeof value==='number')return value>1e12?value:value*1000;return Date.parse(value);}

export async function request(path, body, { signal } = {}) {
  const response = await fetch(`/api${path}`, { method: body ? 'POST' : 'GET', headers: body ? { 'Content-Type': 'application/json' } : undefined, body: body ? JSON.stringify(body) : undefined, signal });
  let data;
  try { data = await response.json(); } catch { throw new Error('The service returned an unreadable response. Please try again.'); }
  if (!response.ok) {
    const detail=data?.error||{}; const error = new Error(detail.message || data?.message || 'Unable to load this repository. Please try again.');
    error.code=detail.code;error.status=response.status;error.rateLimited=response.status===429||/rate.?limit/i.test(detail.code||'');
    const headerRetry=response.headers?.get?.('retry-after');
    error.retryAfter=Number(detail.retry_after??headerRetry)||undefined;
    const reset=detail.retry_at||detail.reset_at||detail.resetAt||detail.rate_limit?.cooldown_until||detail.rate_limit?.reset_at;
    error.retryAtMs=timestamp(reset);
    if(!Number.isFinite(error.retryAtMs)&&error.retryAfter>0)error.retryAtMs=Date.now()+error.retryAfter*1000;
    if(!Number.isFinite(error.retryAtMs)&&headerRetry)error.retryAtMs=timestamp(headerRetry);
    error.retryAt=Number.isFinite(error.retryAtMs)?new Date(error.retryAtMs).toISOString():null;
    error.credentialMode=detail.rate_limit?.credential_mode;
    if(error.rateLimited&&error.retryAtMs>Date.now())setCooldown(error);
    throw error;
  }
  return data;
}
export function analyzeRepository(input, options) {
  const repository=normalizeRepository(input);const cached=getCached(repository);if(cached)return Promise.resolve(cached);
  assertAllowance();return request('/analyze',{url:repository},options).then(data=>{saveAnalysis(repository,data);return data;});
}
export function compareRepositories(first, second, options) {
  const urls=[normalizeRepository(first),normalizeRepository(second)];
  if(urls[0].toLowerCase()===urls[1].toLowerCase())throw new Error('Choose two different repositories to compare.');
  const cached=urls.map(getCached);
  if(cached.every(Boolean))return Promise.resolve({repositories:cached});
  assertAllowance();
  if(cached.some(Boolean))return Promise.all(urls.map((url,index)=>cached[index]||analyzeRepository(url,options))).then(repositories=>({repositories}));
  return request('/compare',{urls},options).then(data=>{data.repositories?.forEach((analysis,index)=>saveAnalysis(urls[index],analysis));return data;});
}
export function loadModelInfo(options) { return request('/model-info', null, options); }
export async function checkConnection(options) {
  const health=await request('/health',null,options);
  if(health.github_token_configured===true&&cooldown?.credentialMode==='anonymous'){
    cooldown=null;writeSession(COOLDOWN_KEY,null);for(const listener of cooldownListeners)listener();
  }
  return health;
}
export function clearClientState() {cachedAnalyses.clear();cooldown=null;writeSession(CACHE_KEY,null);writeSession(COOLDOWN_KEY,null);for(const listener of cooldownListeners)listener();}
