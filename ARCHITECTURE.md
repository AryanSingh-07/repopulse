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

## Two equivalent serving paths

The hosted app is a Cloudflare Worker containing the built React application, the GitHub collection service, the actual tree arrays exported from scikit-learn, and the model card. The Worker follows each trained tree and averages its leaf probabilities; it does not substitute a handwritten scoring formula. Model artifacts are bundled into the deployment, so inference needs no external Python server.

For local Python development, `backend.app:app` exposes the FastAPI service. The frontend development server proxies `/api` to it. Python training saves both the Joblib model and portable JSON; the portable format is also used by the Python API to keep serving behavior consistent. Tests compare exported inference against scikit-learn.

## Evidence and limitations

Training history is the ancestry reachable from each repository's collected default-branch HEAD. The operational target is **no default-branch commits during the next 183 days**. This includes commits brought into the default branch by merges, but does not measure activity on unmerged branches or merged PRs that do not leave a new default-branch commit. This narrower scope is visible in the methodology.

Seven model inputs measure 30/90-day commit activity, contributor participation, days since the last commit, and repository age. Issues, pull requests, and releases are descriptive context, not additional model inputs. No present-day star counts, archive status, or future events enter historical features.

GitHub history requests have explicit pagination caps. Incomplete chart bins are blank rather than reported as zero. If the last 90 days of commits are incomplete, inference is withheld. Request errors, empty repositories, unavailable models, and API limits return explicit states.

The training corpus is a convenience sample of small public libraries, not a representative GitHub sample. Evaluation holds out repository groups and uses later dates; repeated windows from the same repository remain correlated. Scores are uncalibrated exploratory model estimates, not certified confidence or a statement that maintainers have abandoned a project.

## Storage and credentials

Sanitized historical commit records, derived CSVs, provenance, model artifacts, and evaluation output are ordinary files. Contributor emails in the retained training corpus are hashed. No repository source blobs are retained in the dataset. Browser session storage remembers recent repository names only. API credentials remain server-side in `GITHUB_TOKEN`; they are never included in the React bundle or tracked `.env` files.

## References

- [GitHub REST API rate limits](https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api)
- [Scikit-learn evaluation and cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
