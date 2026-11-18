REQTRACE AI Known Limitations
1. LLM Performance
The verified development environment executes the local narration model primarily on CPU.

Natural-language narration may therefore have noticeable latency.

This limitation does not affect deterministic assessment correctness.

2. Narration Validation Failure
The local narration model may occasionally fail strict formatting, citation, or control-boundary requirements.

REQTRACE AI handles this condition through:

One constrained repair attempt.
Deterministic fallback.
Unvalidated narration is not delivered.

3. Evidence Corpus Dependency
REQTRACE AI can reason only from available indexed engineering evidence.

Missing source documents can lead to:

INCONCLUSIVE

or:

CLOSURE NOT ESTABLISHED

rather than a guessed result.

4. Metadata Dependency
Correct provenance and traceability depend on structured metadata quality.

Required metadata includes, where applicable:

Requirement ID
Observation ID
Test Case ID
Document revision
Page
Source file
Function
Physical line range
Test result
5. Explicit Unknown Identifiers
Explicit unknown identifiers are treated as hard failures.

Example:

OBS-999

returns:

NOT FOUND / NOT ASSESSED

REQTRACE AI intentionally does not return a semantically similar observation.

6. Human Authority
REQTRACE AI is a decision-support system.

REQTRACE AI does not provide final:

Engineering approval
Quality approval
Safety approval
Certification approval
Release authorization
Observation closure authorization
These responsibilities remain with authorized human personnel.

7. Demonstration Scope
The verified benchmark results apply to the defined capstone evidence corpus and test scenarios.

Additional operational datasets should undergo independent verification before production use.