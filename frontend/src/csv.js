/**
 * Client-side CSV generation for repository analysis results.
 */

function escapeCsvCell(cell) {
  if (cell == null) return '';
  const str = String(cell);
  return /[",\r\n]/.test(str) ? `"${str.replace(/"/g, '""')}"` : str;
}

export function generateAnalysisCsv(data) {
  if (!data || typeof data !== 'object') return '';
  const r = data.repository || {};
  const p = data.prediction || {};
  const f = data.features || {};

  const rows = [
    ['section', 'metric', 'value'],
    ['repository', 'full_name', r.full_name || ''],
    ['repository', 'url', r.url || (r.full_name ? `https://github.com/${r.full_name}` : '')],
    ['repository', 'created_at', r.created_at || ''],
    ['repository', 'age_days', r.age_days ?? ''],
    ['repository', 'language', r.language || ''],
    ['repository', 'stars', r.stars ?? ''],
    ['repository', 'forks', r.forks ?? ''],
    ['repository', 'open_issues', r.open_issues ?? ''],
    ['repository', 'open_pull_requests', r.open_pull_requests ?? ''],
    ['repository', 'last_commit_at', r.last_commit_at || ''],
    ['repository', 'archived', r.archived ? 'true' : 'false'],
    ['prediction', 'probability', p.probability ?? ''],
    ['prediction', 'risk', p.risk || ''],
    ['prediction', 'model_version', p.model_version || ''],
    ['prediction', 'predicted_at', p.predicted_at || ''],
    ['prediction', 'explanation', p.explanation || ''],
    ['feature', 'commits_30d', f.commits_30d ?? ''],
    ['feature', 'commits_90d', f.commits_90d ?? ''],
    ['feature', 'commit_trend', f.commit_trend ?? ''],
    ['feature', 'contributors_90d', f.contributors_90d ?? ''],
    ['feature', 'contributor_trend', f.contributor_trend ?? ''],
    ['feature', 'days_since_last_commit', f.days_since_last_commit ?? ''],
    ['feature', 'repository_age_days', f.repository_age_days ?? ''],
  ];

  if (p.contributions && Array.isArray(p.contributions)) {
    p.contributions.forEach((c, idx) => {
      rows.push(['contribution', `top_${idx + 1}_feature`, c.feature || '']);
      rows.push(['contribution', `top_${idx + 1}_direction`, c.direction || '']);
      rows.push(['contribution', `top_${idx + 1}_sentence`, c.sentence || '']);
      rows.push(['contribution', `top_${idx + 1}_contribution`, c.contribution ?? '']);
    });
  }

  const months = (data.activity?.commits || []).map(c => c.month);
  if (months.length > 0) {
    const commitsByMonth = Object.fromEntries((data.activity?.commits || []).map(c => [c.month, c]));
    const issuesByMonth = Object.fromEntries((data.activity?.issues || []).map(c => [c.month, c]));
    const pullsByMonth = Object.fromEntries((data.activity?.pull_requests || []).map(c => [c.month, c]));
    const releasesByMonth = Object.fromEntries((data.activity?.releases || []).map(c => [c.month, c]));

    for (const m of months) {
      rows.push(['activity', `${m}_commits`, commitsByMonth[m]?.commits ?? '']);
      rows.push(['activity', `${m}_contributors`, commitsByMonth[m]?.contributors ?? '']);
      rows.push(['activity', `${m}_issues_opened`, issuesByMonth[m]?.opened ?? '']);
      rows.push(['activity', `${m}_issues_closed`, issuesByMonth[m]?.closed ?? '']);
      rows.push(['activity', `${m}_pulls_opened`, pullsByMonth[m]?.opened ?? '']);
      rows.push(['activity', `${m}_pulls_merged`, pullsByMonth[m]?.merged ?? '']);
      rows.push(['activity', `${m}_releases`, releasesByMonth[m]?.releases ?? '']);
    }
  }

  return rows.map(row => row.map(escapeCsvCell).join(',')).join('\r\n');
}

export function downloadAnalysisCsv(data) {
  const csvContent = generateAnalysisCsv(data);
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  const fullName = data?.repository?.full_name || 'repository';
  a.download = `${fullName.replace('/', '-')}-analysis.csv`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
