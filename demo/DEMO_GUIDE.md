REQTRACE AI Final Demonstration Guide
Demonstration Objective
Demonstrate that REQTRACE AI can:

retrieve engineering evidence;
preserve engineering identity;
generate deterministic engineering assessments;
distinguish assessment from closure readiness;
preserve traceability and revision conflicts;
expose physical provenance;
constrain LLM narration;
fail safely when narration is unavailable or invalid.
1. Pre-Demo Verification
Activate the environment:

.venv\Scripts\activate
Run system health:

python health_check.py
Run regression:

python -m pytest -q
Run production benchmark:

python evaluate_production_app.py
Expected production result:

Passed cases: 6
Failed cases: 0
Production acceptance rate: 1.000
2. Start REQTRACE AI
Recommended:

start_reqtrace.bat
Alternative:

python -m streamlit run app.py
Open the local Streamlit URL shown by the console.

3. Architecture Introduction
Explain:

REQTRACE AI is not a conventional RAG chatbot.

Architecture:

Query
  ↓
Hybrid Retrieval
  ↓
Explicit Identifier Integrity
  ↓
Deterministic Engineering Decision
  ↓
Grounding and Provenance Validation
  ↓
Safe Answer Contract
  ↓
Constrained Local LLM Narration
  ↓
Narration Validation
  ↓
Validated Answer or Deterministic Fallback
Core rule:

LLM = narrator

LLM != assessment authority
LLM != confidence authority
LLM != closure authority
LLM != traceability authority
4. UI Walkthrough
Show the five tabs:

Engineering Answer
Evidence Matrix
Decision Trace
Provenance
Safety
Explain that every tab is derived from the same unified production response.

5. Close the Demo
Summarize:

REQTRACE AI combines hybrid retrieval with deterministic engineering decision logic, evidence grounding, physical provenance, explicit-ID integrity, constrained narration, and deterministic fallback.

The engineering result remains available even if the LLM fails.