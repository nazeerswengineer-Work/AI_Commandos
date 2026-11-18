REQTRACE AI
Verification and Validation Report
Project: REQTRACE AI
System Type: Evidence-Grounded Engineering Decision-Support System
Verification Scope: Hybrid retrieval, deterministic engineering assessment, traceability, grounding, citation provenance, closure readiness, constrained LLM narration, operational resilience, and production UI.

1. Purpose
This report documents the verification and validation activities performed for REQTRACE AI.

REQTRACE AI is designed to support engineering requirement traceability and observation analysis using retrieved engineering evidence.

The system intentionally separates:

Evidence retrieval
Engineering identity validation
Deterministic engineering assessment
Confidence classification
Closure-readiness evaluation
Citation and provenance validation
Natural-language narration
The LLM narration layer does not determine the engineering result.

2. Verification Objectives
The verification program demonstrates that REQTRACE AI:

retrieves relevant engineering evidence;
preserves explicit engineering identifiers;
does not replace missing explicit identifiers using semantic similarity;
produces deterministic observation assessments;
distinguishes assessment from closure readiness;
preserves requirement revision conflicts;
preserves explicit traceability;
validates citation provenance;
rejects unsupported engineering claims;
validates generated LLM narration;
falls back to deterministic output when narration is unsafe or unavailable;
preserves engineering results through the production UI.
3. Architectural Safety Principle
REQTRACE AI follows this authority model:

LLM = narrator

LLM != assessment authority
LLM != confidence authority
LLM != closure authority
LLM != traceability authority
LLM != certification authority
LLM != release authority

Engineering assessment is produced before LLM narration.

4. System Under Verification
The verified runtime pipeline is:

Engineering Evidence ↓ Evidence Ingestion ↓ Structured Evidence Store ↓ Hybrid Retrieval ↓ Explicit Identifier Integrity ↓ Final Evidence Context ↓ Deterministic Engineering Assessment ↓ Confidence Classification ↓ Optional Closure Evaluation ↓ Citation Construction ↓ Grounding Validation ↓ Safe Answer Contract ↓ Adversarial Validation ↓ Deterministic Answer Package ↓ Constrained LLM Narration ↓ Narration Validation ↓ Repair or Deterministic Fallback ↓ Production Response ↓ Streamlit Engineering UI

5. Verification Strategy
Verification was performed incrementally.

Major verification categories included:

Unit testing
Retrieval inspection
Deterministic scenario testing
Grounding testing
Provenance testing
Adversarial testing
Narration validation
Production end-to-end testing
Operational resilience testing
UI contract testing
Explicit identifier integrity testing
Performance testing
The complete regression suite is required to pass before a development phase is accepted.

6. Benchmark Evidence Scenarios
Six engineering scenarios were used throughout system verification.

OBS-023
Expected:

Assessment: SUPPORTED
Confidence: HIGH

Primary evidence chain:

REQ-017
OBS-023
process_data()
TC-102 FAIL

Purpose:

Verify support of an observation through requirement, implementation, and failing executed test evidence.

OBS-031
Expected:

Assessment: NOT SUPPORTED
Confidence: HIGH

Primary evidence includes contradictory technical behavior and:

TC-201 PASS

Purpose:

Verify that technical evidence can contradict an observation instead of automatically supporting every retrieved observation.

OBS-044
Expected:

Assessment: INCONCLUSIVE
Confidence: HIGH

Primary blocker:

TC-301 NOT EXECUTED

Purpose:

Verify that incomplete validation evidence does not become an unsupported positive or negative conclusion.

OBS-052
Expected:

Assessment: INCONCLUSIVE
Confidence: HIGH

Authoritative traceability:

OBS-052 → REQ-041

REQ-042 may appear as semantically relevant evidence but must not replace REQ-041 automatically.

Purpose:

Verify traceability preservation.

REQ-050 / OBS-061
Expected:

Assessment: INCONCLUSIVE
Confidence: HIGH

Evidence contains multiple requirement revisions.

Purpose:

Verify that REQTRACE AI preserves unresolved requirement revision conflicts rather than selecting one revision automatically.

OBS-070
Expected observation assessment:

SUPPORTED
HIGH

Expected closure result when closure evaluation is requested:

CLOSURE NOT ESTABLISHED

Purpose:

Verify separation between historical defect support and corrective-action closure readiness.

7. Hybrid Retrieval Verification
The retrieval architecture combines:

Exact engineering-ID retrieval
BM25 lexical retrieval
Dense semantic retrieval
Reciprocal Rank Fusion
Controlled traceability expansion
Final-context selection
Exact IDs provide identity precision.

BM25 provides lexical matching for engineering terminology, identifiers, filenames, functions, requirement language, and diagnostic terminology.

Dense retrieval provides semantic recall.

Reciprocal Rank Fusion combines retrieval channels without requiring the raw lexical and semantic scores to share the same scale.

Controlled traceability expansion adds related evidence while preserving source-recorded traceability.

8. Explicit Identifier Integrity Verification
A critical retrieval-integrity invariant was introduced:

Explicit identity has higher authority than semantic similarity.

For example:

Query: OBS-999

Required result:

Status: NOT FOUND
Assessment: NOT ASSESSED
Closure: NOT ASSESSED

The system must not substitute:

OBS-052

or any other semantically similar observation.

Equivalent protections apply to:

REQ-999
TC-999

The LLM narration layer is not invoked for a missing explicit identifier.

This prevents the system from producing a well-grounded answer about the wrong engineering entity.

9. Deterministic Assessment Verification
The assessment engine produces:

SUPPORTED
NOT SUPPORTED
INCONCLUSIVE

The result is derived from deterministic evidence conditions and assessment rules.

The LLM cannot change the assessment.

Verified scenarios demonstrate:

OBS-023 → SUPPORTED
OBS-031 → NOT SUPPORTED
OBS-044 → INCONCLUSIVE
OBS-052 → INCONCLUSIVE
REQ-050 → INCONCLUSIVE
OBS-070 → SUPPORTED

10. Confidence Verification
Confidence is generated by a deterministic classifier.

Confidence is not:

a model probability;
an LLM confidence estimate;
a statistical certainty claim.
The narrator is explicitly prohibited from converting deterministic confidence into an unsupported probability or percentage.

11. Closure-Readiness Verification
Closure readiness is evaluated separately from observation assessment.

This distinction is essential.

Example:

OBS-070

Assessment: SUPPORTED

Confidence: HIGH

Closure: CLOSURE NOT ESTABLISHED

Supporting evidence that an original defect existed does not prove that:

corrective implementation exists;
corrected behavior was tested;
post-correction verification passed;
engineering review accepted the correction;
an authorized person approved closure.
Closure intent may also be detected from natural-language queries such as:

"What is the closure status of OBS-070?"

"Is OBS-070 open or closed?"

"Can OBS-070 be closed?"

12. Citation and Provenance Verification
Citation validation requires source-specific provenance.

PDF evidence preserves:

document name;
revision where available;
one-based page number;
evidence ID;
engineering identifiers;
content hash where available.
Code evidence preserves:

source filename;
function where available;
physical line range;
evidence ID;
engineering identifiers;
content hash where available.
REQTRACE AI does not allow the narration layer to invent page numbers, physical source lines, revisions, identifiers, or test outcomes.

13. Grounding Verification
Grounding validation checks include:

final-context citation coverage;
deterministic evidence-ID coverage;
requirement grounding;
observation grounding;
actual Test evidence for test-result claims;
source-code provenance;
requirement-revision conflict grounding;
traceability preservation;
closure grounding;
engineering-ID allowlist enforcement.
A Test case ID mentioned by another source is not equivalent to actual Test evidence.

A filename mentioned inside an observation is not equivalent to actual source-code evidence.

14. Adversarial Verification
The system was tested against:

altered assessment;
altered confidence;
altered closure status;
fabricated citations;
unsupported engineering IDs;
added claims;
removed prohibitions;
revision selection;
traceability rewriting;
false closure statements;
false authorization statements;
prompt-like instructions embedded inside source evidence.
Prompt-like source text remains untrusted quoted evidence and cannot become narrator instructions.

15. Narration Verification
The local narration model is used only after deterministic engineering processing and grounding validation.

Narration validation checks include:

NV-001 Assessment preservation
NV-002 Confidence preservation
NV-003 Closure preservation
NV-004 Engineering-ID allowlist
NV-005 Citation allowlist
NV-006 Required citations
NV-007 Test-result grounding
NV-008 Code provenance
NV-009 Revision-conflict protection
NV-010 Traceability protection
NV-011 False closure protection
NV-012 Human-authority protection
NV-013 Evidence instruction isolation
NV-014 Unsupported numeric claims
NV-015 Required output sections
NV-016 Narrator-control leakage

Invalid narration cannot become the final answer.

16. Repair and Fallback Verification
Maximum LLM calls per request:

Initial generation: 1
Repair generation: 1
Maximum total: 2

Delivery states are:

VALIDATED_GENERATION
VALIDATED_REPAIR
DETERMINISTIC_FALLBACK

If both generated responses fail validation, or if the optional narration service times out, REQTRACE AI returns the deterministic grounded answer.

Therefore:

Model failure may reduce fluency.

Model failure cannot reduce engineering decision integrity.

17. Production Application Verification
Phase 15 production evaluation executed all six benchmark scenarios through the unified production pipeline.

Production benchmark:

Passed scenarios: 6 / 6
Failed scenarios: 0
Production acceptance rate: 1.000

The production response exposes:

Assessment
Confidence
Closure status
Engineering IDs
Supporting items
Blocking items
Citations
Provenance
Narration delivery state
Fallback state
Request timing
18. UI Verification
The production Streamlit interface exposes:

Assessment status
Confidence status
Closure status
Engineering Answer
Evidence Matrix
Decision Trace
Provenance Inspector
Safety Inspector
Runtime timing
System health
Automatic closure-intent feedback
The UI consumes only the unified production response.

The UI does not independently perform engineering assessment.

19. Operational Resilience Verification
Verified resilience behaviors include:

Ollama unavailable → deterministic pipeline remains usable

Narration timeout → deterministic fallback

Narration validation failure → one repair attempt

Repair failure → deterministic fallback

Empty query → controlled rejection

Unknown explicit engineering identifier → NOT FOUND

Grounding failure → hard production failure

Unsafe delivery state → hard production failure

20. Performance Verification
The narration model executes on CPU in the verified development system.

Phase 16A optimization reduced the narration prompt approximately from:

1965 tokens

to approximately:

1038 tokens

This represents roughly a 47% prompt-size reduction.

Production narration configuration was reduced to:

Context window: 4096
Maximum narration output: 350 tokens
Model keep-alive: 15 minutes

These optimizations preserved all deterministic and validation safety controls.

Hardware inference performance remains environment-dependent.

21. Regression Verification
The project regression suite passed before Phase 16 packaging.

The final acceptance report must record the current test total generated after the explicit-identifier integrity tests were added.

No phase should be accepted with a failing full regression suite.

22. Residual Risks
Residual risks include:

CPU-only LLM inference may produce high latency.

Small local language models may frequently fail strict narration formatting or citation requirements.

Semantic retrieval quality depends on the completeness and correctness of the indexed engineering corpus.

Structured metadata quality directly affects traceability and provenance quality.

Closure readiness is only as reliable as the corrective-implementation, verification, and review evidence available to the system.

REQTRACE AI provides engineering decision support and does not replace authorized human engineering judgment.

23. Verification Conclusion
REQTRACE AI satisfies the defined verification objectives for the validated demonstration corpus and benchmark scenarios.

The system demonstrates:

hybrid engineering evidence retrieval;
explicit engineering-identifier integrity;
deterministic engineering assessment;
deterministic confidence classification;
separate closure-readiness evaluation;
citation and physical provenance enforcement;
traceability preservation;
requirement-revision conflict preservation;
adversarial answer-package validation;
constrained local LLM narration;
post-generation narration validation;
bounded repair;
deterministic safety fallback;
operational resilience;
explainable production UI behavior.
The LLM is not an engineering decision authority.

The validated engineering result remains deterministic and evidence-grounded.