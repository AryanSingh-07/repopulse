# RepoPulse architecture

RepoPulse keeps descriptive GitHub observations separate from an exploratory machine-learning prediction. It does not use a database.

```text
React dashboard → same-origin /api → GitHub REST API
                            ↓
                  shared feature definitions
                            ↓
                  trained Random Forest

Public Git default-branch histories → historical windows → time/repository split
                                                       ↓
                                                actual evaluation
                                                       ↓
                                         Joblib + portable forest JSON
```

## Three serving paths

The Vercel build uses Node.js 22 and Build Output API v3. `npm run build:vercel` builds the React assets into `.vercel/output/static/` and bundles a Node.js function into `.vercel/output/functions/api.func/`. `vercel/handler.mjs` adapts Node HTTP requests to the existing `worker/api.js` collection and inference service. Build Output routes send same-origin `/api` requests to that function and serve the static frontend separately. `vercel.json` sets `npm ci && npm --prefix frontend ci` as the install command. `npm run test:vercel` verifies the adapter and generated output.

The existing Sites deployment is a Cloudflare Worker containing the built React application, the GitHub collection service, the actual tree arrays exported from scikit-learn, and the model card. `npm run build` produces this Worker artifact. Both JavaScript serving paths follow each trained tree and average its leaf probabilities; neither substitutes a handwritten scoring formula. Model artifacts are bundled into each deployment, so inference needs no external Python server.

For local Python development, `backend.app:app` exposes the FastAPI service. The frontend development server proxies `/api` to it. Python training saves both the Joblib model and portable JSON; the portable format is also used by the Python API to keep serving behavior consistent. Tests compare exported inference against scikit-learn.

## Evidence and limitations

Training history is the ancestry reachable from each repository's collected default-branch HEAD. The operational target is **no default-branch commits during the next 183 days**. This includes commits brought into the default branch by merges, but does not measure activity on unmerged branches or merged PRs that do not leave a new default-branch commit. This narrower scope is visible in the methodology.

Seven model inputs measure 30/90-day commit activity, contributor participation, days since the last commit, and repository age. Issues, pull requests, and releases are descriptive context, not additional model inputs. No present-day star counts, archive status, or future events enter historical features.

GitHub history requests have explicit pagination caps. Incomplete chart bins are blank rather than reported as zero. If the last 90 days of commits are incomplete, inference is withheld. Request errors, empty repositories, unavailable models, and API limits return explicit states.

The training corpus is a convenience sample of small public libraries, not a representative GitHub sample. Evaluation holds out repository groups and uses later dates; repeated windows from the same repository remain correlated. Scores are uncalibrated exploratory model estimates, not certified confidence or a statement that maintainers have abandoned a project.

## Storage and credentials

Sanitized historical commit records, derived CSVs, provenance, model artifacts, and evaluation output are ordinary files. Contributor emails in the retained training corpus are hashed. No repository source blobs are retained in the dataset. Browser session storage remembers recent repository names, up to twelve successful public analyses for at most ten minutes, and known rate-limit reset metadata. Cached results retain their observation timestamps; incomplete optional upstream failures are not saved as complete analyses.

Server response caches are bounded, scoped to the configured credential, and held in memory for up to ten minutes. They are not durable or guaranteed to be shared across hosting instances. API credentials remain server-side in `GITHUB_TOKEN`; they are never included in the React bundle, browser storage, or tracked `.env` files. Vercel Production and Preview environments, Sites secrets, and a local server's environment are configured independently. Changing a local `.env` does not update either hosted deployment; deploy or redeploy the selected hosting environment after setting its secret.

## References

- [GitHub REST API rate limits](https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api)
- [Scikit-learn evaluation and cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
