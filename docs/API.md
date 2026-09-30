# Inference Engine API

Mail-Lens AI does not expose an HTTP API. The inference pipeline is consumed directly through Python.

## `MailLensInference`

Located in `src/inference/engine.py`.

### Instantiation

```python
from src.inference.engine import MailLensInference

# Uses default model directory (models/saved/)
engine = MailLensInference()

# Or specify a custom path
engine = MailLensInference(model_dir="path/to/models")
```

The constructor auto-loads the model if `best_model.joblib` is found in the target directory.

### `analyze(text, subject="", sender="")`

The primary analysis method.

**Parameters:**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `text` | `str` | Yes | Email body content |
| `subject` | `str` | No | Email subject line |
| `sender` | `str` | No | Sender email address |

**Returns:** `AnalysisResult` dataclass

**Raises:** `RuntimeError` if model not loaded

### `AnalysisResult` fields

| Field | Type | Description |
|---|---|---|
| `prediction` | `str` | `"LEGITIMATE"`, `"PHISHING"`, `"MALICIOUS"`, or `"UNKNOWN"` |
| `confidence` | `float` | Probability of predicted class (0.0–1.0) |
| `probabilities` | `dict` | `{"LEGITIMATE": float, "PHISHING": float, "MALICIOUS": float}` |
| `risk_level` | `str` | `"LOW"`, `"MEDIUM"`, `"HIGH"`, `"CRITICAL"` |
| `risk_score` | `float` | Composite risk score (0.0–1.0) |
| `risk_color` | `str` | CSS color string for the risk level |
| `detected_indicators` | `list[str]` | Human-readable security signal descriptions |
| `indicator_count` | `int` | Number of security indicators fired |
| `explanation` | `str` | Multi-line plain-text explanation |
| `recommendation` | `str` | Actionable security advice |
| `top_threat_tokens` | `list[dict]` | Top n-grams pushing toward threat class |
| `top_safe_tokens` | `list[dict]` | Top n-grams pushing toward legitimate class |
| `pipeline_trace` | `dict` | Preprocessing and feature diagnostics |
| `email_snippet` | `str` | First 100 chars of combined input |

**Token dict format** (for `top_threat_tokens` and `top_safe_tokens`):
```python
{
    "token": str,   # N-gram string
    "tfidf": float, # TF-IDF weight for this input
    "weight": float, # LinearSVC coefficient
    "impact": float  # tfidf * weight
}
```

### Example

```python
from src.inference.engine import MailLensInference

engine = MailLensInference()

result = engine.analyze(
    text="Your account has been suspended. Click here to verify: http://login-secure.xyz/restore",
    subject="URGENT: Account Suspended",
    sender="support@corporate-helpdesk.xyz",
)

print(result.prediction)    # "PHISHING"
print(result.confidence)    # e.g. 0.9842
print(result.risk_level)    # "HIGH"
print(result.risk_score)    # e.g. 0.781
print(result.detected_indicators)
# ["Urgency language detected", "Credential-seeking keywords", ...]

result_dict = result.to_dict()  # Convert to plain dict
```

### Batch processing

There is no built-in batch method. For batch inference:

```python
results = [
    engine.analyze(text=row["body"], subject=row["subject"])
    for _, row in df.iterrows()
]
```

### Edge cases

| Input | Behavior |
|---|---|
| Empty string | Returns `prediction="UNKNOWN"`, `risk_score=0.0` |
| Whitespace only | Same as empty — stripped before check |
| Very long text (>50k chars) | Accepted but may slow preprocessing |
| Non-Latin text | Processed; performance may be lower |
| HTML content | Tags stripped by preprocessor |
| SQL / shell injection strings | Treated as inert text tokens |
