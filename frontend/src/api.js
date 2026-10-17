export function normalizeRepository(input) {
  const value = String(input || '').trim();
  const match = value.match(/^(?:https?:\/\/(?:www\.)?github\.com\/)?([a-zA-Z0-9](?:[a-zA-Z0-9-]{0,38}))\/([a-zA-Z0-9_.-]+)(?:\.git)?\/?$/);
  if (!match || match[2] === '.' || match[2] === '..') throw new Error('Enter a public GitHub repository, like facebook/react or its GitHub URL.');
  return `${match[1]}/${match[2].replace(/\.git$/, '')}`;
}
export async function request(path, body, { signal } = {}) {
  const response = await fetch(`/api${path}`, { method: body ? 'POST' : 'GET', headers: body ? { 'Content-Type': 'application/json' } : undefined, body: body ? JSON.stringify(body) : undefined, signal });
  let data;
  try { data = await response.json(); } catch { throw new Error('The service returned an unreadable response. Please try again.'); }
  if (!response.ok) { const error = new Error(data?.error?.message || data?.message || 'Unable to load this repository. Please try again.'); error.code = data?.error?.code; error.retryAfter = data?.error?.retry_after; throw error; }
  return data;
}
export function analyzeRepository(input, options) { return request('/analyze', { url: normalizeRepository(input) }, options); }
export function compareRepositories(first, second, options) {
  const urls = [normalizeRepository(first), normalizeRepository(second)];
  if (urls[0].toLowerCase() === urls[1].toLowerCase()) throw new Error('Choose two different repositories to compare.');
  return request('/compare', { urls }, options);
}
export function loadModelInfo(options) { return request('/model-info', null, options); }
