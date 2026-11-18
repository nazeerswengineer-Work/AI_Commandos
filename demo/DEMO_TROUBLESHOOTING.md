REQTRACE AI Demonstration Troubleshooting
Streamlit Launcher Access Denied
Use:

python -m streamlit run app.py
instead of:

streamlit run app.py
Ollama Is Slow
Check:

ollama ps
The validated development environment may show:

llama3.2:1b
PROCESSOR: 100% CPU
CPU-only narration can have noticeable latency.

This does not affect deterministic engineering analysis.

Recommended demo option:

Disable grounded LLM narration when demonstrating deterministic engineering capabilities.

Ollama Is Offline
REQTRACE AI should continue operating through deterministic output.

Expected production behavior:

DETERMINISTIC_FALLBACK
Core engineering analysis remains available.

Narration Fails Validation
This is an expected controlled behavior.

The system allows:

Initial generation
    ↓
Validation failure
    ↓
One repair
    ↓
Validation failure
    ↓
Deterministic fallback
Invalid generated narration is never delivered.

Unknown Observation
Example:

OBS-999
Expected:

NOT FOUND
NOT ASSESSED
REQTRACE AI must not substitute another observation.

Health Status Is DEGRADED
A DEGRADED status can still mean:

Core ready: True
Narration ready: False
The deterministic engineering pipeline remains operational.

Unexpected Core Failure
Run:

python health_check.py
Then:

python -m pytest -q
Then:

python evaluate_production_app.py
Do not continue the formal demo if the deterministic core is unavailable.