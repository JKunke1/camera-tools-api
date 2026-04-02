# Camera Tools Full Action Package

This package provides a deterministic FastAPI backend plus an OpenAPI schema for use with a Custom GPT Action.

Included endpoints:
- /health
- /estimate-record-time
- /estimate-file-size
- /estimate-render-time
- /camera-preset-lookup
- /crop-factor
- /field-of-view
- /depth-of-field
- /lens-equivalency

## Run locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Then expose it through a public HTTPS tunnel and replace the placeholder domain in `openapi.yaml`.
