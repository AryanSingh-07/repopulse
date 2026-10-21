/**
 * Extract repository parameter from URL search or hash location.
 */
export function getRepoFromUrl(locationSearch, windowLocation) {
  if (locationSearch) {
    const params = new URLSearchParams(locationSearch);
    const repo = params.get('repo');
    if (repo && repo.trim()) return repo.trim();
  }
  if (windowLocation) {
    if (windowLocation.search) {
      const params = new URLSearchParams(windowLocation.search);
      const repo = params.get('repo');
      if (repo && repo.trim()) return repo.trim();
    }
    if (windowLocation.hash && windowLocation.hash.includes('?')) {
      const q = windowLocation.hash.slice(windowLocation.hash.indexOf('?'));
      const params = new URLSearchParams(q);
      const repo = params.get('repo');
      if (repo && repo.trim()) return repo.trim();
    }
  }
  return null;
}
