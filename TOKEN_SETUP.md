# Restore GitHub API access

If the app reports an anonymous GitHub allowance or `/api/health` reports `github_token_configured: false`, the serving environment needs a server-side `GITHUB_TOKEN` for a higher allowance. Each Vercel environment, Sites deployment, and local server has independent configuration. Anonymous GitHub REST requests have a shared server-IP allowance of 60 requests per hour. Token-authenticated user requests normally have a 5,000-per-hour allowance; search and secondary limits still apply.

## Create a token for public reads

1. Open [GitHub's fine-grained token form](https://github.com/settings/personal-access-tokens/new?name=RepoPulse&description=Read%20public%20repository%20activity&expires_in=30).
2. Use the name **RepoPulse** and an expiration appropriate for the project, such as 30 days.
3. Keep access limited to public repositories and do not grant write or account permissions. GitHub's fine-grained tokens include read-only access to public repositories.
4. Generate the token yourself. Treat it as a password.

## Configure the hosted app

The selected hosting environment needs `GITHUB_TOKEN` as a server-side secret. Adding it to a local `.env` alone does not update a hosted deployment. Never put the token in frontend source, a `VITE_*` variable, a build artifact, or a Git commit. Retraining the model is unnecessary when configuring credentials.

- **Vercel:** add `GITHUB_TOKEN` to the intended project's **Production** environment, then deploy or redeploy the application. Add it separately to Preview if those deployments need authenticated access. The Node.js 22 API function reads the server environment at runtime. For the standalone `repopulse` repository, leave Root Directory at the repository root (`.`); the configured install command is `npm ci && npm --prefix frontend ci` and build command is `npm run build:vercel`.
- **Sites:** save `GITHUB_TOKEN` as a secret for the existing Site, then deploy the saved Site version to apply the new environment revision.
- **Local serving:** put `GITHUB_TOKEN=...` in the ignored `repopulse/.env` for the local preview, or provide it in the Python server's environment. Restart the local service after changing its environment.

If configuring with Codex, put the token only in the ignored local `repopulse/.env` file as `GITHUB_TOKEN=...`, say that it is saved, and identify the intended hosting project and environment. That value can then be transferred to the authorized server secret without putting it into source or frontend code. Do not paste the token into chat.

Check `/api/health` on the deployment being used; it reports `github_token_configured` as a boolean only. A valid token is verified by a real repository request; the presence of the environment key alone does not establish valid credentials. The interface's **Check connection** action can clear a previously recorded anonymous cooldown after server authentication is configured. Authenticated and unknown cooldowns keep their reported reset times.

## While waiting for access

The interface displays GitHub's reported retry time. Wait until then instead of repeatedly clicking Analyze. The app reuses recent real observations when available and labels their observation time; a cache cannot supply results that have never been fetched.

Successful cached public responses and analyses are kept for up to ten minutes. In-memory caches are bounded, scoped to the credential, and reset when the serving instance restarts. They use no database and do not provide a guaranteed shared cache across all hosting instances.

References: [GitHub REST rate limits](https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api), [personal access tokens](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).
