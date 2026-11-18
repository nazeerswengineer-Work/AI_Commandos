REQTRACE AI
Evidence-Grounded RAG Assistant for Engineering Requirement Traceability and Observation Analysis
1. Executive Summary

REQTRACE AI is an evidence-grounded engineering decision-support system designed to analyze requirements, software observations, source code, test evidence, engineering reviews, and traceability relationships.

Unlike a conventional Retrieval-Augmented Generation chatbot, REQTRACE AI does not allow the Large Language Model to determine the engineering conclusion.

The system separates:


Evidence retrieval
↓
Engineering identity validation
↓
Deterministic engineering assessment
↓
Confidence classification
↓
Closure-readiness assessment
↓
Grounding and provenance validation
↓
Constrained LLM narration
↓
Post-generation validation
↓
Validated answer or deterministic fallback

This architecture prioritizes traceability, reproducibility, explainability, and fail-safe behavior.

2. Problem Statement

Engineering observation analysis requires information distributed across multiple artifacts:

Functional Requirement Documents
Observation reports
Source code
Test reports
Engineering reviews
Traceability records

A conventional semantic-search or RAG solution can retrieve related information but introduces several engineering risks:

Semantically similar requirements may be confused.
Missing engineering IDs may be silently substituted.
Conflicting requirement revisions may be automatically resolved incorrectly.
A language model may infer unsupported engineering conclusions.
Test status may be claimed without actual test evidence.
Observation support may be confused with closure readiness.
Generated answers may lack auditable source provenance.

REQTRACE AI was designed specifically to control these failure modes.

3. Project Objectives

The project objectives are to:

Retrieve heterogeneous engineering evidence.
Combine lexical and semantic retrieval.
Prioritize exact engineering identifiers.
Preserve explicit traceability.
Prevent substitution of unknown explicit identifiers.
Generate deterministic engineering assessments.
Classify deterministic confidence.
Evaluate observation closure separately.
Preserve citations and physical provenance.
Validate engineering claims before narration.
Restrict the LLM to narration.
Validate generated narration.
Fall back safely when narration fails.
Expose decision evidence through an explainable engineering UI.
4. Engineering Evidence Model

The demonstration corpus contains multiple evidence classes:


FRD
Observation
Source Code
Test Report
Engineering Review

Evidence maintains structured metadata where applicable:


Evidence ID
Document type
Requirement ID
Observation ID
Test Case ID
Revision
Page
Filename
Function
Physical line range
Test result
Content hash

This metadata is used for traceability and provenance rather than relying solely on semantic interpretation of free text.

5. Hybrid Retrieval Architecture

REQTRACE AI uses several complementary retrieval mechanisms:

5.1 Exact-ID Retrieval

Queries containing explicit identifiers such as:


OBS-023
REQ-050
TC-401

receive identity-sensitive handling.

5.2 BM25 Retrieval

BM25 supports literal engineering vocabulary including:


requirements
identifiers
filenames
function names
test-case references
diagnostic terminology
5.3 Dense Semantic Retrieval

Dense retrieval provides semantic recall when relevant artifacts use different wording.

5.4 Reciprocal Rank Fusion

RRF combines lexical and semantic retrieval rankings without requiring raw BM25 and embedding similarity scores to operate on the same numerical scale.

5.5 Controlled Traceability Expansion

Related engineering evidence can be added through explicit trace relationships while preserving source-recorded identity.

6. Explicit Identifier Integrity

A critical safety property is:

Explicit engineering identity has higher authority than semantic similarity.

For example:


Query
OBS-999
↓
Exact identifier validation
↓
No authoritative OBS-999 evidence
↓
NOT FOUND
NOT ASSESSED
↓
STOP

The system does not substitute a semantically similar observation such as OBS-052.

Equivalent protection applies to:


REQ-999
TC-999

This prevents a particularly dangerous RAG failure:


Wrong entity retrieved
↓
Technically grounded reasoning
↓
Confident but user-incorrect answer
7. Deterministic Engineering Decision Engine

The engineering assessment is produced before LLM narration.

Possible outcomes are:


SUPPORTED
NOT SUPPORTED
INCONCLUSIVE

Decision rules consider evidence completeness, executed tests, contradictory technical evidence, traceability blockers, and requirement revision conflicts.

The language model cannot modify this result.

8. Confidence Classification

Confidence is deterministic and accompanies the engineering assessment.

It must not be interpreted as:


probability
statistical certainty
LLM confidence

The narration layer is explicitly prohibited from translating HIGH, MEDIUM, or LOW into an invented probability.

9. Closure-Readiness Engine

REQTRACE AI deliberately distinguishes:


Was the original observation supported?

from:


Is sufficient evidence available to close it?

For example:


OBS-070
 
Assessment:
SUPPORTED
 
Confidence:
HIGH
 
Closure:
CLOSURE NOT ESTABLISHED

Supporting the original defect does not automatically prove that:

a corrective implementation exists;
post-correction testing passed;
review evidence exists;
an authorized engineer approved closure.

Natural-language questions such as:


Is OBS-070 open or closed?
 
What is the closure status of OBS-070?
 
Can OBS-070 be closed?

automatically activate closure evaluation.

10. Citation and Provenance Architecture

REQTRACE AI preserves evidence-specific provenance.

PDF evidence

Document name
Revision
Page
Evidence identifier
Engineering identifiers
Source-code evidence

Filename
Function
Physical line range
Evidence identifier
Engineering identifiers
Test evidence

Test Case ID
Source report
Page
Result status
 

The LLM cannot invent provenance.

11. Grounding Validation

Before narration, deterministic claims are validated against citations.

Important rules include:


Observation claim
→ actual Observation evidence
 
PASS/FAIL claim
→ actual Test evidence
 
Code behavior
→ actual source-code provenance
 
Revision conflict
→ multiple requirement revisions
 
Closure claim
→ closure evidence
 
Engineering ID
→ validated allowlist
12. Safe Answer Contract

Grounded evidence is converted into a machine-readable contract containing:


Authoritative assessment
Confidence
Closure status
 
Engineering-ID allowlist
Citation allowlist
 
Supporting items
Blocking items
 
Allowed claims
Prohibited claims
 
Narrator instructions

The contract becomes the boundary between engineering reasoning and LLM narration.

13. Adversarial Safety

REQTRACE AI was tested against:

Modified assessment
Modified confidence
Modified closure
Fabricated citation
Unsupported engineering identifiers
Removed safety prohibitions
Added unauthorized claims
Traceability rewriting
Requirement revision selection
False closure statements
Prompt-like instructions embedded in evidence

Invalid packages are blocked before narration.

14. Constrained Local LLM Narration

The local narration layer uses Ollama with llama3.2:1b.

The authority model is:


LLM = narrator
 
LLM ≠ assessment authority
LLM ≠ confidence authority
LLM ≠ closure authority
LLM ≠ traceability authority
LLM ≠ engineering approval authority

Evidence supplied to narration is explicitly treated as untrusted quoted source material.

15. Narration Validation

Generated narration passes a separate validation gate.

Validation covers:


Assessment preservation
Confidence preservation
Closure preservation
Engineering-ID allowlist
Citation allowlist
Required citations
Test-result grounding
Code provenance
Revision-conflict preservation
Traceability preservation
False closure detection
Human-authority claims
Evidence-instruction leakage
Unsupported numbers
Required answer sections
Prompt/control leakage

Raw invalid model output cannot become the final answer.

16. Repair and Deterministic Fallback

At most two model calls are possible:


Initial generation
↓
Validation
┌────┴────┐
PASS FAIL
│ │
Deliver One Repair
↓
Validation
┌────┴────┐
PASS FAIL
│ │
Deliver Deterministic
Fallback

This provides a central reliability guarantee:

Model failure may reduce answer fluency but cannot reduce engineering decision integrity.

17. Production Application

The unified production pipeline exposes a stable application response containing:


Assessment
Confidence
Closure status
Final answer
 
Requirement IDs
Observation IDs
Test IDs
 
Citations
Supporting evidence
Blocking evidence
 
Narration status
Fallback status
Request timing

The Streamlit application consumes only this production response.

18. Explainable Engineering UI

The production UI provides five primary views:

Engineering Answer

Final validated narration or deterministic answer.

Evidence Matrix

Engineering evidence and traceability relationships.

Decision Trace

Deterministic rule and assessment progression.

Provenance

Source documents, pages, files, functions, physical code lines, and test results.

Safety

Grounding status, narration mode, repair, fallback, and timing.

19. Verification and Validation

The six core scenarios are:

Scenario	Assessment	Confidence	Key result
OBS-023	SUPPORTED	HIGH	Defect evidence supported
OBS-031	NOT SUPPORTED	HIGH	Contrary evidence preserved
OBS-044	INCONCLUSIVE	HIGH	Insufficient execution evidence
OBS-052	INCONCLUSIVE	HIGH	REQ-041 preserved
REQ-050	INCONCLUSIVE	HIGH	Revision conflict preserved
OBS-070	SUPPORTED	HIGH	CLOSURE NOT ESTABLISHED

Phase 15 production acceptance achieved:


Passed scenarios: 6
Failed scenarios: 0
Acceptance rate: 100%

Negative identifier validation also includes:


OBS-999 → NOT FOUND
REQ-999 → NOT FOUND
TC-999 → NOT FOUND
20. Performance Optimization

The verified development machine executes local narration on CPU.

Phase 16A reduced the approximate narration-prompt size from:


~1965 tokens

to:


~1038 tokens

representing approximately:


47% reduction

The optimized runtime configuration is:


Context window: 4096
Maximum output tokens: 350
Model keep-alive: 15 minutes

These optimizations did not weaken grounding or narration validation.

21. Operational Resilience

REQTRACE AI distinguishes required deterministic components from optional narration.

Examples:


Ollama unavailable
→ deterministic answer remains available
 
Ollama timeout
→ deterministic fallback
 
Invalid narration
→ repair or fallback
 
Unknown explicit ID
→ NOT FOUND
 
Grounding failure
→ hard engineering pipeline failure

System-health reporting exposes required and optional component status separately.

22. Key Contributions

The principal technical contributions demonstrated by REQTRACE AI are:

Hybrid retrieval tailored to engineering evidence.
Explicit-ID-first identity preservation.
Deterministic engineering assessment above the LLM layer.
Separate assessment and closure semantics.
Physical source provenance.
Claim-level grounding controls.
Traceability-preserving retrieval and reasoning.
Requirement-revision conflict preservation.
Constrained narration.
Post-generation safety validation.
Bounded repair with deterministic fallback.
Explainable production UI.
23. Known Limitations
Local LLM inference is slow on CPU-only hardware.
Small local language models may frequently fail strict narration validation.
System quality depends on source-evidence completeness.
Structured metadata quality directly affects provenance.
Closure assessment cannot compensate for missing corrective-action evidence.
Demonstration V&V applies to the verified corpus and benchmark scenarios.
Human engineering and certification authority remain outside REQTRACE AI.
24. Future Enhancements

Potential future work includes:

GPU-accelerated local inference
Larger validated engineering corpora
Additional document parsers
Automated ingestion monitoring
Requirement baselining
Revision-aware configuration management
Graph-based traceability visualization
Change-impact analysis
Role-based engineering workflows
Exportable assessment reports
Integration with ALM/requirements platforms
Expanded aerospace/automotive verification datasets

These enhancements should preserve the deterministic authority boundary.

25. Conclusion

REQTRACE AI demonstrates that retrieval-augmented engineering assistance does not require delegating engineering decisions to a language model.

The implemented architecture uses hybrid retrieval to locate evidence, explicit identifier integrity to preserve engineering identity, deterministic rule engines to establish engineering conclusions, grounding and provenance checks to validate those conclusions, and a constrained local LLM only to improve their presentation.

When LLM narration is invalid, unavailable, or slow, REQTRACE AI retains the validated deterministic engineering response.

This architecture produces a system that is more traceable, explainable, testable, fail-safe, and appropriate for controlled engineering decision-support workflows than a conventional retrieve-and-generate chatbot.