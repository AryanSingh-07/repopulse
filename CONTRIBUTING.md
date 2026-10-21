# Contributing to RepoPulse

Thank you for your interest in contributing to RepoPulse! RepoPulse is an open-source repository intelligence dashboard built with React, FastAPI, Cloudflare Worker, Vercel Functions, and a portable scikit-learn Random Forest model.

## Code of Conduct

Please be respectful, collaborative, and constructive when reporting issues, discussing proposals, or reviewing pull requests.

## Getting Started

### Prerequisites

- **Node.js**: 22.x LTS and npm
- **Python**: 3.11+
- **Git**

### Local Setup

1. Fork and clone the repository.
2. Install dependencies:
   ```bash
   npm install
   npm --prefix frontend install
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
   pip install -r backend/requirements.txt -r backend/ml/requirements.txt -r requirements-dev.txt
   ```
3. Build the assets and validate the Worker artifact:
   ```bash
   npm run build
   npm run validate
   npm run preview
   ```

## Development Guidelines

- **Branching**: Create focused topic branches off `main` (e.g., `feature/description` or `fix/issue-number`).
- **Commits**: Write clear, descriptive commit messages.
- **Secrets & Credentials**: Never commit `.env`, API tokens, or secrets. Never expose `GITHUB_TOKEN` to the frontend or via `VITE_*` variables.
- **Inference Parity**: When updating model artifacts or feature extraction, ensure JS and Python inference stay in parity (exported `forest.json` must match scikit-learn predictions within `1e-9`).
- **UI Theme**: Preserve the distinctive navy and peach UI theme (`#181A2F`, `#242E49`, `#FDAA81`, etc.).
- **Data Integrity**: Never fabricate data or metrics; report real evaluation numbers only.

## Running Tests

Before submitting a pull request, verify all test suites pass:

```bash
# Frontend unit tests
npm --prefix frontend test

# Edge worker API tests
node --test tests/edge*.test.mjs

# Backend FastAPI service tests
python -m pytest tests/backend_service_test.py -q

# Machine learning feature extraction & parity tests
python -m unittest backend.ml.test_features -v

# Worker build and artifact validation
npm run build
npm run validate

# Vercel adapter tests
npm run test:vercel
```

## Pull Request Process

1. Ensure all tests and validation checks pass locally.
2. Ensure new features or bug fixes include corresponding unit/integration tests.
3. Open a pull request against `main` describing the changes, motivation, and verification steps.
