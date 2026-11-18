REQTRACE AI Demonstration Scenarios
Demo 1 - Supported Observation
Query:

OBS-023
Expected:

Assessment: SUPPORTED
Confidence: HIGH
Closure: NOT ASSESSED
Demonstrate:

REQ-017
OBS-023
process_data()
TC-102 FAIL
Show:

Evidence Matrix
Code provenance
Test result
Decision Trace
Key message:

REQTRACE AI does not conclude SUPPORTED from semantic similarity. The result comes from the deterministic evidence chain.

Demo 2 - Observation Not Supported
Query:

OBS-031
Expected:

Assessment: NOT SUPPORTED
Confidence: HIGH
Key evidence:

TC-201 PASS
Purpose:

Demonstrate that retrieval of an observation does not mean the system automatically agrees with the observation.

Demo 3 - Inconclusive Evidence
Query:

OBS-044
Expected:

Assessment: INCONCLUSIVE
Confidence: HIGH
Important evidence:

TC-301 NOT EXECUTED
Key message:

REQTRACE AI does not guess when decisive verification evidence is unavailable.

Demo 4 - Traceability Protection
Query:

OBS-052
Expected:

Assessment: INCONCLUSIVE
Authoritative traceability:

OBS-052 -> REQ-041
Important:

REQ-042 may be semantically related but must not replace REQ-041.

Key message:

Semantic relevance cannot rewrite source-recorded engineering traceability.

Demo 5 - Requirement Revision Conflict
Query:

REQ-050
Expected:

Assessment: INCONCLUSIVE
Demonstrate:

Multiple FRD revisions
Revision conflict
No automatic revision selection
Key message:

REQTRACE AI preserves unresolved configuration ambiguity rather than allowing the LLM to select a convenient revision.

Demo 6 - Closure Readiness
Query:

What is the closure status of OBS-070?
Expected:

Assessment: SUPPORTED
Confidence: HIGH
Closure Status: CLOSURE NOT ESTABLISHED
Demonstrate:

Automatic closure-intent detection
Supporting original-defect evidence
Missing corrective-action / post-correction evidence
Key message:

Defect supported
    !=
Observation ready for closure
Demo 7 - Explicit Identifier Integrity
Query:

OBS-999
Expected:

Status: NOT FOUND
Assessment: NOT ASSESSED
Must NOT happen:

OBS-999 -> OBS-052
Key message:

Explicit engineering identity has higher authority than semantic similarity.

No LLM narration should run for the missing identifier.

Demo 8 - Safe LLM Fallback
Enable grounded LLM narration.

Run:

OBS-023
Possible safe outcomes:

VALIDATED_GENERATION
or:

VALIDATED_REPAIR
or:

DETERMINISTIC_FALLBACK
If fallback occurs, explain:

The LLM response failed validation or exceeded the runtime limit. REQTRACE AI discarded the unsafe narration and returned the deterministic grounded engineering answer.

Key message:

Model failure affects fluency, not engineering integrity.