# RepoPulse — Repository Intelligence

[![CI](https://github.com/AryanSingh-07/repopulse/actions/workflows/ci.yml/badge.svg)](https://github.com/AryanSingh-07/repopulse/actions/workflows/ci.yml)

A distinctive navy-and-peach dashboard for understanding public GitHub activity and exploring six-month inactivity risk. React, FastAPI, GitHub REST, and an actually trained scikit-learn Random Forest. No Streamlit and no database.

Live application: [RepoPulse on Vercel](https://repopulse-rose.vercel.app/).

The application supports a Vercel Node.js 22 deployment, the existing Sites Cloudflare Worker deployment, and a Python API for local development. Both JavaScript deployment builds use a portable export of the same trained forest. See [ARCHITECTURE.md](ARCHITECTURE.md) for the serving paths and evidence boundaries.

## Run the complete app locally (Worker preview)

Prerequisites: Node.js 22.x, npm, Python 3.11+ for the Python service/training, and Git for historical collection.

```powershell
cd repopulse
npm install
npm --prefix frontend install
npm run build
npm run validate
npm run preview
```

Open `http://localhost:4173`. This serves both the React app and the real GitHub API/inference endpoints. The saved trained model is included; retraining is optional.

Copy `.env.example` to `.env` if you want to supply `GITHUB_TOKEN` for local serving. This is optional for public repositories. Without it, GitHub's unauthenticated quota is small and shared by outbound IP. Use a token with the least access needed for public repository reads; never place it in `frontend/` or a `VITE_*` variable. Configure hosted credentials separately as a server environment variable in the relevant Vercel project and environment or as a Sites secret.

If you see "GitHub's API rate limit was reached", follow [TOKEN_SETUP.md](TOKEN_SETUP.md). A local `.env` does not configure the hosted secret. The app caches recent real observations and displays the reported retry time, but a token is needed for a larger server allowance.

## Develop with FastAPI

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.app:app --reload --port 8000
```

In a second terminal, run `npm --prefix frontend run dev` and open the Vite Local URL. Vite proxies `/api` to FastAPI on port 8000. Interactive Python API documentation is at `http://localhost:8000/docs`.

## What the model means

The target is **no default-branch commits in the 183 days after a historical prediction date**. A commit brought into that branch by a merged PR is counted. Activity on unmerged branches, or merges confined to another branch, is outside this target. An inactive outcome does not establish permanent abandonment; mature projects may be complete and healthy with little activity.

Inputs are identical between training and serving:

| Feature | Definition |
| --- | --- |
| `commits_30d` | Default-branch commits in the preceding 30 days |
| `commits_90d` | Default-branch commits in the preceding 90 days |
| `commit_trend` | Recent 30-day commits / (preceding 60-day commits / 2 + 1) − 1 |
| `contributors_90d` | Distinct commit author identities in 90 days |
| `contributor_trend` | Recent 30-day authors / (preceding 60-day authors + 1) − 1 |
| `days_since_last_commit` | UTC days since the latest observed commit |
| `repository_age_days` | UTC days since GitHub repository creation |

Issues, pull requests, stars, forks, and releases provide descriptive context. They are not additional inputs to the fitted model. Importances shown beside inputs are global forest importances, not causal or per-repository explanations.

Probabilities are **uncalibrated exploratory estimates**. Low (<30%), medium (30–<65%), and high (≥65%) bands are display conventions. They are not optimized intervention thresholds or validated confidence levels. The model card gives actual held-out metrics and the baseline comparison.

### Executed evaluation

The saved model was trained on 864 historical windows from 35 repositories and evaluated on 144 later windows from 12 different repositories. Results at a classification threshold of 0.5:

| Metric | Random Forest | Prior baseline |
| --- | ---: | ---: |
| ROC-AUC | 0.800688 | 0.500000 |
| Inactive-class precision | 0.492958 | 0 |
| Inactive-class recall | 0.760870 | 0 |
| Inactive-class F1 | 0.598291 | 0 |
| Accuracy | 0.673611 | 0.680556 |

The model detects more inactive outcomes and ranks them better, but its accuracy is slightly below the majority baseline because it also produces false positives. The confusion matrix is `[[62, 36], [11, 35]]`, with observed active/inactive rows and predicted active/inactive columns. These are descriptive results on this small selected corpus, not a guarantee of performance on arbitrary repositories. The exported forest matched scikit-learn predictions within `4.44e-16` on the holdout.

## Reproduce the data and model

The retained training sources are 48 real public repositories, with sanitized full Git default-branch histories, source HEADs, collection timestamps, and actual GitHub creation dates. Outcomes are calculated from commits; no outcomes are manually assigned. Author email identities are hashed in retained records.

```powershell
python -m pip install -r backend/ml/requirements.txt
# Optional: fetch or refresh the real histories; uses public network and Git.
python scripts/collect_history.py --refresh --workers 4
# Rebuild CSV partitions, fit, evaluate, export, and render the EDA figure.
python scripts/train_model.py
```

The collection is cached as ordinary JSON files; `data/.git-cache/` is excluded from source control. Training can run directly against the included history JSON without making network requests.

The split assigns entire repositories using a deterministic hash before examining outcomes. Training prediction dates end in 2022; held-out prediction dates begin in July 2023, after all training outcome windows finish. Complete future windows are required; censored examples are excluded. Historical windows from one repository are correlated and may overlap. The corpus is a convenience sample of libraries, and metrics should not be generalized to all of GitHub.

Artifacts:

- `model/repopulse.joblib`: fitted scikit-learn forest and metadata.
- `model/forest.json`: actual portable decision trees used at inference.
- `model/model-info.json`: actual evaluation, thresholds, feature definitions, and limitations.
- `model/evaluation.png`: class balance and held-out confusion matrix.
- `data/historical_windows.csv`, `train.csv`, `test.csv`: derived observation windows and outcomes.
- `data/provenance.json`, `training_provenance.json`: source commits, collection times, partitions, and hashes.

## API

| Method | Endpoint | Body / result |
| --- | --- | --- |
| GET | `/api/health` | Model availability and runtime |
| GET | `/api/model-info` | Actual model card and evaluation |
| POST | `/api/analyze` | `{"url":"pallets/click"}` → observations, charts, estimate, coverage |
| POST | `/api/predict` | Same validated collection and prediction workflow |
| POST | `/api/compare` | `{"urls":["pallets/click","pypa/packaging"]}` → two analyses |

Only public `github.com` repository roots are accepted. Credentials stay server-side. Requests to other hosts and repository subpaths are rejected. An unavailable or empty history does not get a fabricated score. When a pagination cap prevents calculating the 90-day inputs, the response contains `prediction: null` and a coverage explanation. Missing chart values are gaps, while observed zero activity is represented by zero.

## Verification

```powershell
npm --prefix frontend test
node --test tests/edge*.test.mjs
python -m pip install -r requirements-dev.txt
python -m pytest tests/backend_service_test.py -q
python -m unittest backend.ml.test_features -v
npm run build
npm run validate
npm run test:vercel
```

Tests cover URL validation, API errors, time-window boundaries, future-data exclusion, censored outcomes, feature compatibility, exported inference, and request/response behavior. Product data always comes from GitHub; mocked HTTP responses are confined to offline tests.

## Deployment

### Vercel

For the standalone `repopulse` GitHub repository, leave the Vercel project's Root Directory at the repository root (`.`) and use Node.js **22.x**. The checked-in `vercel.json` specifies the install command `npm ci && npm --prefix frontend ci` and build command `npm run build:vercel`.

```powershell
cd repopulse
npm ci
npm --prefix frontend ci
npm run build:vercel
npm run test:vercel
```

`scripts/build-vercel.mjs` produces **Build Output API v3** files under `.vercel/output/`: static React assets and a Node.js 22 function serving `/api`. The function bundles the trained forest and reuses the existing GitHub collection and inference implementation; a separate Python server is unnecessary.

The current production deployment was published from the Vercel CLI. From an authenticated checkout, use `npx vercel link` to select the intended project, then `npx vercel deploy --prod` to publish a new build. Continuous deployment on Git pushes requires separately connecting the GitHub repository in Vercel; it is not configured by this one-time deployment.

Add `GITHUB_TOKEN` to the project's **Production server environment** for an authenticated GitHub allowance, then deploy or redeploy to apply it. Preview deployments have their own environment configuration. Do not use a `VITE_*` variable for the token. See [TOKEN_SETUP.md](TOKEN_SETUP.md) for credential handling and verification.

### Existing Sites deployment

`.openai/hosting.json` records the existing Site identity. The Sites source workflow packages `dist/server/index.js` and deployment metadata from the exact pushed source commit. `npm run build` invokes `scripts/build-worker.mjs`, which embeds the React build and trained model in the Worker. This serving path also avoids a separate hosted Python service while retaining the same fitted model. The application uses no database.

To redeploy, rebuild and validate, then use the Sites publishing workflow for the existing project. Never create another Site merely to update this one. The deployment defaults to private access for the owner.
