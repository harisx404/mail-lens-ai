# Deployment

## Current deployment target

Mail-Lens AI is a Streamlit application. It is designed to run locally or on a small VM. Full deployment on Streamlit Community Cloud is feasible if model artifact sizes stay within the platform's repository size limit.

## Local deployment

The simplest deployment is running directly:

```bash
streamlit run app/main.py --server.port 8501 --server.headless true
```

For background execution on Linux/macOS:
```bash
nohup streamlit run app/main.py --server.port 8501 --server.headless true &
```

## Streamlit Community Cloud

**Status:** Deployment-compatible in principle.

**Steps:**
1. Push the repository to GitHub (public or connected private)
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repository
3. Set the main file path to `app/main.py`
4. The model artifacts in `models/saved/` will be included (their total size is ~1.2MB, well within limits)

**Limitation:** The training datasets (`data/raw/`) are excluded from the repository (see `.gitignore`). The training script (`scripts/train.py`) cannot run on Streamlit Cloud because it needs to download external datasets and has a training time that exceeds serverless time limits. The pre-trained model in `models/saved/` must be committed to the repository for cloud deployment to work.

**Environment variables for Streamlit Cloud:**

Go to App Settings → Secrets and add:
```toml
MODELS_DIR = "models/saved"
LOG_LEVEL = "WARNING"
```

## Vercel

**Status: Not compatible.**

Vercel's Python runtime (Serverless Functions) has strict resource limits:
- 50MB unzipped function size — model artifacts alone are ~1.2MB but the full Python environment (scikit-learn, numpy, scipy, NLTK) exceeds this significantly
- 10-second function timeout — ML model loading + inference would likely exceed this on cold start
- No persistent filesystem — Streamlit's session state model requires a persistent process

Vercel is not a suitable deployment target for this application without significant architectural changes (e.g., migrating to a FastAPI backend hosted on a VPS or container service).

## Docker

For containerized deployment:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.headless=true"]
```

Build and run:
```bash
docker build -t mail-lens-ai .
docker run -p 8501:8501 mail-lens-ai
```

## Environment variables

| Variable | Required | Default | Notes |
|---|---|---|---|
| `MODELS_DIR` | No | `models/saved` | Path to model artifact directory |
| `DATA_RAW_DIR` | Training only | `data/raw` | Not needed for inference |
| `DATA_PROCESSED_DIR` | Training only | `data/processed` | Not needed for inference |
| `RANDOM_SEED` | No | `42` | Only affects training |
| `LOG_LEVEL` | No | `INFO` | Set to `WARNING` in production |
| `APP_PORT` | No | `8501` | Streamlit server port |

## Model artifact handling

The trained model artifacts in `models/saved/` are committed to the repository because:
- Total size is approximately 1.2MB (well under GitHub's 100MB file limit)
- Committing them makes the app immediately runnable without a training step
- This is appropriate for a prototype with a fixed, stable model

If the model is retrained and artifacts change significantly, commit the new artifacts and update `model_metadata.json`.

## Production checklist

If exposing the application beyond localhost:

- [ ] Put Streamlit behind a reverse proxy (nginx / Caddy) with HTTPS
- [ ] Add access controls (basic auth, IP allowlist, or OAuth)
- [ ] Set `LOG_LEVEL=WARNING` to reduce log verbosity
- [ ] Set `APP_DEBUG=false`
- [ ] Ensure model files are from trusted sources
- [ ] Review Streamlit's security settings in `.streamlit/config.toml`
