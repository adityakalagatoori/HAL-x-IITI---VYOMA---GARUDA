# Security Review — Turbojet Digital Twin Dashboard

## Current attack surface: minimal
The app (`dashboard/app.py`) is a **read-only Streamlit visualization** over static local CSVs
(`data/*.csv`). It has:
- No user file upload
- No user text input parsed as code/SQL
- No `eval`/`exec`/`os.system`/`subprocess` calls
- No hardcoded credentials, API keys, or secrets
- No outbound network calls (fully offline once data is loaded)

Verified by direct source inspection — confirmed clean.

## Findings

| # | Item | Severity | Status |
|---|---|---|---|
| 1 | No auth on the dashboard | Low (data is not sensitive: synthetic PS dataset) | Accepted — do not put real HAL/proprietary data behind this without adding auth |
| 2 | `requirements.txt` not yet pinned | Low | **Action before deploy**: pin exact versions (`streamlit==1.50.0`, `plotly==<installed>`, `pandas==2.3.3`, etc.) to avoid supply-chain drift on redeploy |
| 3 | If deployed publicly (Streamlit Community Cloud / similar), default network binding is open | Low | Fine for a hackathon demo; do not reuse this exact deployment for anything with real operational engine data later |

## Forward-looking: if you add live upload for the finale demo
If the 24-hr finale round asks you to demo on **new/uploaded sensor CSVs live**, add before doing so:
- File size cap (e.g., reject >5MB) to avoid memory-exhaustion DoS from `pandas.read_csv`
- Column schema validation (reject files missing the required PS-2 sensor columns, don't silently coerce)
- Never use `pandas.read_pickle` or `eval`-based parsing on user-uploaded files — CSV via `pd.read_csv` only
- Sanitize the `EngineID`/`Cycle` selectors if they ever come from raw uploaded text rather than a dropdown (currently they don't — `st.selectbox` only, which is safe)

## Verdict
**Safe to run/demo as-is.** No exploitable vulnerabilities found because there is no user-controlled input path yet. Revisit this review only if you add file upload, auth, or a backend API before the finale.
