1. Master architecture
┌─────────────────────────────────────┐
│       Engineering Evidence          │
│                                     │
│ FRD | Observation | Code | Test     │
│ Review / Traceability Evidence      │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│     Evidence Ingestion Layer        │
│                                     │
│ Parse → Normalize → Metadata        │
│ Evidence IDs → Content Hashes       │
│ Page / File / Function / Lines      │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│          Evidence Store             │
│                                     │
│ ChromaDB + Structured Metadata      │
│ Dense Embeddings + Evidence IDs     │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│       Hybrid Retrieval Layer        │
│                                     │
│ Exact-ID retrieval                  │
│ BM25 lexical retrieval              │
│ Dense semantic retrieval            │
│ Reciprocal Rank Fusion              │
│ Controlled traceability expansion   │
│ Final-context selection             │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│ Explicit Identifier Integrity Gate  │
│                                     │
│ OBS / REQ / TC explicitly requested │
│          │                          │
│     Exact ID exists?                │
│       │          │                  │
│      YES         NO                 │
│       │          └─→ NOT FOUND      │
│       │              No substitution│
└───────┬─────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│   Deterministic Engineering Layer   │
│                                     │
│ Evidence completeness               │
│ Conflict / condition detection      │
│ Observation assessment              │
│ SUPPORTED                           │
│ NOT SUPPORTED                       │
│ INCONCLUSIVE                        │
│ Deterministic confidence            │
│ Separate closure-readiness engine   │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│       Grounding Safety Layer        │
│                                     │
│ Citation construction               │
│ Provenance validation               │
│ Claim grounding                     │
│ Engineering-ID allowlist            │
│ Citation allowlist                  │
│ Safe answer contract                │
│ Adversarial package validation      │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│      Deterministic Answer           │
│                                     │
│ Always available as safe fallback   │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│        Narration Layer              │
│                                     │
│ Ollama / llama3.2:1b                │
│ Narration only                      │
│ No engineering decision authority   │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│      Narration Validation           │
│                                     │
│ Decision preservation               │
│ ID / citation validation            │
│ Test & code grounding               │
│ Traceability protection             │
│ Revision-conflict protection        │
│ Closure protection                  │
│ Control-leakage detection           │
└──────────────────┬──────────────────┘
                   │
           ┌───────┴────────┐
           │                │
          PASS             FAIL
           │                │
           ▼                ▼
     Validated LLM     One Repair
       Narration           │
                           ▼
                     Validate Again
                       │       │
                     PASS     FAIL
                       │       │
                       ▼       ▼
                    Valid    Deterministic
                    Repair     Fallback
                       │       │
                       └───┬───┘
                           ▼
┌─────────────────────────────────────┐
│       Production Pipeline           │
│                                     │
│ Unified ProductionResponse          │
│ Health / timing / resilience        │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│       Streamlit Engineering UI      │
│                                     │
│ Engineering Answer                  │
│ Evidence Matrix                     │
│ Decision Trace                      │
│ Provenance Inspector                │
│ Safety & Runtime                    │
└─────────────────────────────────────┘

2. Architecture principle

The core design statement for your capstone should be:

REQTRACE AI separates evidence retrieval, engineering decision-making, grounding validation, and natural-language narration into
independently controlled layers. The LLM explains validated engineering results but does not determine them.

That distinction is one of the strongest aspects of the project.

3. Retrieval architecture
                  User Query
                      │
                      ▼
             Query / ID Analysis
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
       Exact-ID      BM25        Dense
       Retrieval    Lexical     Semantic
          │           │           │
          └───────────┼───────────┘
                      ▼
                    RRF
          Reciprocal Rank Fusion
                      │
                      ▼
            Traceability Expansion
                      │
                      ▼
              Context Selection
                      │
                      ▼
          Final Evidence Context
		  
Design rationale:

Exact-ID retrieval protects engineering identity.

BM25 captures technical vocabulary, requirement IDs, filenames, functions, diagnostic terms, and literal engineering language.

Dense retrieval captures semantically related evidence that does not share exact terminology.

RRF combines lexical and semantic rankings without requiring incompatible score normalization.

Controlled traceability expansion expands relationships without allowing semantic similarity to rewrite authoritative traceability.

4. Explicit identifier integrity
Query
"Tell me about OBS-999"
          │
          ▼
Extract explicit IDs
          │
          ▼
      OBS-999
          │
          ▼
Final retrieved context
          │
          ▼
Structured metadata contains OBS-999?
          │
    ┌─────┴─────┐
    │           │
   YES          NO
    │           │
    ▼           ▼
Continue     STOP PIPELINE
analysis
                │
                ▼
             NOT FOUND
             NOT ASSESSED

             No OBS-052
             substitution

             No decision

             No LLM
			 
Architecture invariant:

Explicit identity > semantic similarity

If a requested engineering identifier does not exist, REQTRACE AI fails closed rather than analyzing the nearest semantic entity.

5. Deterministic decision architecture
Final Validated Context
          │
          ▼
Evidence Completeness
          │
          ▼
Conditions / Conflicts
          │
    ┌─────┼──────────────┐
    ▼     ▼              ▼
Tests   Revision     Traceability
       Conflict       Concern
    │     │              │
    └─────┼──────────────┘
          ▼
Deterministic Assessment Rules
          │
    ┌─────┼─────────────┐
    ▼     ▼             ▼
SUPPORTED NOT        INCONCLUSIVE
          SUPPORTED
          │
          ▼
Deterministic Confidence
          │
          ▼
Separate Closure Evaluation

The key point is:

Assessment ≠ Closure

For example:

Plain Text
OBS-070
 
Original defect:
SUPPORTED
 
Confidence:
HIGH
 
Closure:
CLOSURE NOT ESTABLISHED

A supported historical defect does not prove that corrective action has been verified.

6. Grounding and citation architecture
Deterministic Decision
        +
Final Evidence Context
          │
          ▼
Citation Construction
          │
          ▼
Provenance Validation
          │
          ▼
Claim Grounding
          │
          ▼
Engineering-ID Allowlist
          │
          ▼
Citation Allowlist
          │
          ▼
Allowed Claims
Prohibited Claims
          │
          ▼
Safe Answer Contract
          │
          ▼
Adversarial Validation
          │
          ▼
Deterministic Answer Package

Provenance model

For PDF evidence:

Document
Revision
Page
Evidence ID
Engineering IDs
Content hash

For source code:

File
Function
Physical line range
Evidence ID
Engineering IDs
Content hash

No page number, physical line, revision, test result, or identifier should be invented by narration.

7. LLM safety architecture
Validated Phase 13 Package
            │
            ▼
Constrained Narration Prompt
            │
            ▼
    llama3.2:1b
            │
            ▼
    UNTRUSTED OUTPUT
            │
            ▼
NV-001 ... NV-016 Validation
            │
      ┌─────┴─────┐
      │           │
    PASS          FAIL
      │           │
      ▼           ▼
  Delivered     Repair Prompt
                  │
                  ▼
              LLM attempt 2
                  │
                  ▼
               Validate
                │    │
              PASS  FAIL
                │    │
                ▼    ▼
             Deliver Deterministic
                      Fallback