REQTRACE AI Test Matrix
Test Area	Verification Objective	Acceptance
Exact-ID Retrieval	Preserve explicitly requested entities	PASS
BM25 Retrieval	Retrieve lexical engineering matches	PASS
Dense Retrieval	Retrieve semantic engineering matches	PASS
RRF	Fuse lexical and semantic rankings	PASS
Traceability Expansion	Expand evidence without rewriting source traceability	PASS
Explicit ID Integrity	Unknown OBS/REQ/TC cannot be substituted	PASS
Evidence Completeness	Identify available/missing evidence	PASS
Conflict Detection	Detect revision and evidence conflicts	PASS
Assessment Engine	SUPPORTED / NOT SUPPORTED / INCONCLUSIVE	PASS
Confidence Engine	Deterministic confidence	PASS
Closure Engine	Separate closure-readiness decision	PASS
Citation Construction	Build validated provenance	PASS
Grounding	Ensure claims are evidence-backed	PASS
Test Grounding	Test result requires actual Test evidence	PASS
Code Grounding	Code claims require physical line provenance	PASS
Answer Contract	Preserve allowed/prohibited claim boundaries	PASS
Adversarial Validation	Reject package tampering	PASS
Narration Prompt	Constrain LLM authority	PASS
Narration Client	Bound Ollama generation	PASS
Narration Validation	Reject unsafe LLM output	PASS
Narration Repair	Permit one constrained repair	PASS
Deterministic Fallback	Fail closed when narration fails	PASS
Closure Intent	Detect natural-language closure questions	PASS
Production Pipeline	Unified safe production response	PASS
Explainability UI	Evidence / trace / provenance / safety views	PASS
Health Monitoring	Core vs optional narration health	PASS
Performance Metrics	Capture deterministic/narration/total timing	PASS
Packaging	Reproducible Windows startup structure	PASS
Production Benchmark	Six scenarios	6/6 PASS