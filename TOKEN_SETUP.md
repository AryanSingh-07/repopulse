# Restore GitHub API access

The app is deployed, but no `GITHUB_TOKEN` is configured in its hosted environment. Anonymous GitHub REST requests have a shared server-IP allowance of 60 requests per hour. Token-authenticated user requests normally have a 5,000-per-hour allowance; search and secondary limits still apply.

## Create a token for public reads

1. Open [GitHub's fine-grained token form](https://github.com/settings/personal-access-tokens/new?name=RepoPulse&description=Read%20public%20repository%20activity&expires_in=30).
2. Use the name **RepoPulse** and an expiration appropriate for the project, such as 30 days.
3. Keep access limited to public repositories and do not grant write or account permissions. GitHub's fine-grained tokens include read-only access to public repositories.
4. Generate the token yourself. Treat it as a password.

## Configure the hosted app

The production environment needs `GITHUB_TOKEN` marked as a secret. Adding it to a local `.env` alone does not update the deployed Site. After the hosted secret is saved, deploy the existing saved Site version to apply the new environment revision; retraining the model is unnecessary.

If configuring with Codex, put the token only in the ignored local `repopulse/.env` file as `GITHUB_TOKEN=...` and say that it is saved. Codex can then transfer that value to the hosted secret without putting it into source or frontend code. Do not paste the token into chat.

The application health endpoint reports `github_token_configured` as a boolean only. A valid token is verified by a real repository request; the presence of the environment key alone does not establish valid credentials.

## While waiting for access

The interface displays GitHub's reported retry time. Wait until then instead of repeatedly clicking Analyze. The app reuses recent real observations when available and labels their observation time; a cache cannot supply results that have never been fetched.

Successful cached public responses and analyses are kept for up to ten minutes. In-memory caches are bounded, scoped to the credential, and reset when the serving instance restarts. They use no database and do not provide a guaranteed shared cache across all hosting instances.

References: [GitHub REST rate limits](https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api), [personal access tokens](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).
