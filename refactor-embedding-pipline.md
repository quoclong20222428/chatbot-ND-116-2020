You are working on the Vietnamese legal RAG chatbot project.

The current project uses `BAAI/bge-m3` with 1024-dimensional embeddings. The project must now be refactored so that developers can flexibly switch between multiple Hugging Face embedding models with minimal or no changes to retrieval, evaluation, and other application logic.

This is primarily a **software architecture and maintainability task**. The goal is not only to support different embedding models, but also to make it possible to run controlled retrieval experiments and compare the results produced by different models.

---

# 1. Supported Embedding Models

The system must support exactly these six embedding models initially:

```text
1. BAAI/bge-m3
2. darklethelong/vnlegal-lal
3. mainguyen9/vietlegal-harrier-0.6b
4. mainguyen9/vietlegal-e5
5. jinaai/jina-embeddings-v3
6. dxtech-asia/deepx-embedding-v1
```

The expected comparison dimension is:

```text
1024
```

However, do NOT blindly assume that every model exposes 1024-dimensional output through the same API.

Before implementation, inspect each model's Hugging Face model card and/or repository configuration and determine:

* Loading framework
* Required Python libraries
* Whether `trust_remote_code=True` is required
* Native embedding dimension
* Whether 1024-dimensional output is directly supported
* Whether Matryoshka/truncation is required to obtain 1024 dimensions
* Query encoding protocol
* Document/passage encoding protocol
* Normalization requirements
* Maximum supported sequence length
* Task/instruction requirements
* Whether the model supports Sentence Transformers
* Any model-specific inference requirements

Do not invent compatibility.

If a model requires a model-specific loader or inference protocol, isolate that behavior behind the embedding abstraction rather than forcing all models through the same implementation.

---

# 2. Main Architectural Goal

Refactor the current embedding implementation so that the rest of the system does not depend on a specific embedding model.

The desired dependency direction is:

```text
Configuration
      │
      ▼
Embedding Model Resolver
      │
      ▼
Embedding Interface
      │
      ├── BGE-M3 backend
      ├── VNLegal-LAL backend
      ├── VietLegal-Harrier backend
      ├── VietLegal-E5 backend
      ├── Jina v3 backend
      └── DeepX backend
      │
      ▼
Embedding vectors
      │
      ▼
PostgreSQL / pgvector
      │
      ▼
Retrieval
      │
      ▼
Evaluation
```

The following layers should NOT need to know which embedding model is currently selected:

* `retrieval.py`
* `test_retrieval.py`
* evaluation metrics
* ground-truth logic
* query definitions
* metadata-aware document representation
* retrieval scoring logic

Changing the model should be a configuration-level operation whenever the selected model is already supported.

---

# 3. Stable Embedding Interface

Create or refactor the embedding abstraction around a stable interface.

At minimum, the abstraction should conceptually support:

```python
embed_query(text)
embed_documents(texts)
get_dimension()
get_model_name()
```

The exact implementation can follow the project's existing coding style.

The important requirement is:

```text
retrieval.py
     ↓
EmbeddingModel interface
     ↓
selected backend
```

and NOT:

```text
retrieval.py
     ↓
if BGE...
if Jina...
if E5...
...
```

Model-specific behavior must remain inside the embedding layer.

---

# 4. Model Backend Strategy

Do not assume that every Hugging Face embedding model uses the same library.

Use a layered strategy.

### Generic backend

Where a model is compatible with a common embedding framework such as Sentence Transformers, reuse the generic backend instead of creating unnecessary model-specific code.

### Specialized backend

If a model requires a specialized implementation, create a dedicated backend/adapter only for that model or model family.

For example:

```text
EmbeddingModel
│
├── GenericSentenceTransformerBackend
│
├── BGEBackend
│
├── JinaBackend
│
└── Other specialized backend(s)
```

The exact backend structure should be determined from the actual model requirements.

The objective is to minimize duplicated code.

Do NOT create six completely independent embedding implementations if several models can share the same backend.

---

# 5. Configuration-Driven Model Selection

The selected embedding model must be configurable without modifying Python source code.

Prefer the existing `.env` mechanism if it is already used by the project.

For example:

```env
EMBEDDING_MODEL=BAAI/bge-m3
```

Optionally introduce:

```env
EMBEDDING_BACKEND=auto
```

only if this provides a real architectural benefit.

Avoid requiring developers to edit Python source code simply to switch models.

The configuration system should validate that the selected model is supported.

For an unsupported model:

* fail early
* provide a clear error message
* list supported model identifiers
* do not silently fall back to another embedding model

---

# 6. Model-Specific Configuration

Avoid scattering model-specific constants throughout the source code.

Centralize information such as:

```text
model identifier
backend
dimension
query encoding strategy
document encoding strategy
normalization
max sequence length
task/instruction
special loading options
```

The exact representation can be a Python configuration structure, registry, or another maintainable mechanism consistent with the project.

The important requirement is that model-specific behavior is centralized.

---

# 7. Query and Document Encoding Must Be Separate

The abstraction must explicitly distinguish:

```text
Query embedding
Document/passage embedding
```

Do NOT reduce everything to a single generic:

```python
embed(text)
```

because different embedding models may use different query/document protocols.

For example:

```text
embed_query(query)
embed_documents(documents)
```

The selected backend is responsible for applying the correct model-specific protocol.

This is particularly important for models that use:

* query/passage prefixes
* retrieval tasks
* instructions
* task-specific adapters
* different pooling behavior

Do not apply a protocol from one model to another model.

---

# 8. Preserve the Existing Metadata-Aware Representation

Do not change the current document representation merely because the embedding model changes.

The current legal document embedding representation is:

```text
[Document]
<document_title>

[Chapter]
<chapter>

[Article]
<article>

[Clause]
<clause>

[Point]
<point>

[Content]
<original content>
```

Preserve the existing rules:

* only legal-text chunks use the structural metadata representation
* optional metadata is included only when non-null/non-empty
* `document_number` is excluded from the embedding representation
* original content remains under `[Content]`
* QA/non-legal chunks continue using their existing text representation

The purpose of this refactor is to change the embedding model, not the retrieval methodology.

---

# 9. 1024-Dimensional Compatibility

The project currently uses a 1024-dimensional pgvector representation.

Every supported model must produce exactly:

```text
1024 dimensions
```

before being inserted into the experiment's vector storage.

If a model natively supports 1024 dimensions, use its supported 1024-dimensional representation.

If a model has a larger native dimension but officially supports a 1024-dimensional Matryoshka representation, use the documented mechanism for obtaining 1024 dimensions.

Do NOT arbitrarily truncate vectors unless the model documentation explicitly supports that procedure.

After encoding, validate the dimension before database insertion.

Example conceptual validation:

```text
expected dimension = 1024
actual dimension   = 1024
→ continue

expected dimension = 1024
actual dimension   = 1536
→ fail clearly
```

This prevents accidental mixing of incompatible vector representations.

---

# 10. Prevent Cross-Model Vector Mixing

This is a critical requirement.

Vectors produced by different embedding models must NEVER be mixed in the same retrieval experiment.

Even if two models both produce 1024-dimensional vectors:

```text
BGE-M3 1024d
Jina v3 1024d
```

they represent different embedding spaces.

Therefore the project must provide a clear mechanism to identify which model produced a vector/index.

Possible approaches include:

* model-specific vector columns
* model-specific tables
* model-specific experiment namespaces
* model-specific indexes
* another clean storage mechanism

Choose the approach that best fits the current database architecture.

Do not unnecessarily redesign the existing schema.

The final design must make it difficult or impossible to accidentally query a BGE-M3 vector index using a Jina query vector.

---

# 11. Embedding Pipeline

Separate the process into clear stages.

## Stage 1 — Model Preparation / Download

The embedding model is obtained from Hugging Face.

The project should make it clear that the model is downloaded automatically when required by the selected backend, or provide a dedicated preparation command if the existing architecture benefits from explicit preparation.

Do not require users to manually download model files into arbitrary project directories unless the existing architecture already does so.

Document:

```text
Selected model
      ↓
Hugging Face
      ↓
Local model cache
```

Do not assume a specific OS or shell for the conceptual documentation.

---

# 12. Stage 2 — Document Embedding

Provide a consistent command/interface for generating document embeddings.

The operation should:

1. Read the current legal chunks
2. Build the existing metadata-aware representation
3. Load the selected embedding model
4. Generate document embeddings
5. Validate the embedding dimension
6. Store embeddings in the selected model-specific storage/index
7. Record enough information to identify the embedding model used
8. Handle failures without corrupting existing experiment data

The embedding process must not depend on BGE-M3-specific code.

---

# 13. Stage 3 — Vector Indexing

Ensure the selected model's vectors receive the appropriate pgvector index.

The existing HNSW configuration should remain unchanged for controlled comparison unless there is a technical reason that a model-specific change is required.

If the same:

```text
distance metric
HNSW parameters
ef_search
```

can be preserved, preserve them.

The benchmark should isolate the embedding model as the primary experimental variable.

---

# 14. Stage 4 — Retrieval

`retrieval.py` should automatically use the currently selected embedding model.

Conceptually:

```text
EMBEDDING_MODEL
      ↓
EmbeddingModel
      ↓
embed_query()
      ↓
1024d query vector
      ↓
matching model-specific vector index
      ↓
Top-K results
```

The retrieval implementation should not contain model-specific branches.

Do not change:

* cosine similarity calculation
* Top-K behavior
* `ef_search`
* document filtering
* metadata matching
* result structure

unless required to support the model abstraction.

---

# 15. Stage 5 — Retrieval Evaluation

`test_retrieval.py` must work against any of the six supported models without modifying the evaluation methodology.

The same:

* 15 evaluation queries
* hierarchical ground truth
* Document → Article → Clause → Point matching
* Hit@K
* Recall@K
* MRR
* similarity diagnostics
* evaluation output format

must be reusable across all models.

The only intended experimental variable should be the embedding model.

For example:

```text
BGE-M3
    ↓
test_retrieval.py
    ↓
metrics

VNLegal-LAL
    ↓
test_retrieval.py
    ↓
metrics

VietLegal-Harrier
    ↓
test_retrieval.py
    ↓
metrics
```

---

# 16. Benchmark Reproducibility

Provide a way to record the configuration used for each experiment.

At minimum, the evaluation output should make it possible to identify:

```text
Embedding model
Embedding dimension
Backend
Device
Precision
Top-K
ef_search
Number of evaluation queries
Evaluation methodology
```

Do not change the current evaluation metrics.

If the project already has a `.txt` evaluation output mechanism, reuse it.

Do not duplicate evaluation formatting logic.

---

# 17. Bash Scripts

You may create a dedicated Bash script for simplifying model-specific experiment execution.

The script should NOT duplicate the embedding/retrieval implementation.

Instead, it should act as a thin orchestration layer.

For example, conceptually:

```text
scripts/
└── run_embedding_experiment.sh
```

It may support:

```bash
./scripts/run_embedding_experiment.sh bge-m3
./scripts/run_embedding_experiment.sh vnlegal-lal
./scripts/run_embedding_experiment.sh vietlegal-harrier
./scripts/run_embedding_experiment.sh vietlegal-e5
./scripts/run_embedding_experiment.sh jina-v3
./scripts/run_embedding_experiment.sh deepx
```

The exact interface can be improved if necessary.

The script should:

1. Validate the model identifier
2. Set/select the appropriate configuration
3. Prepare/load the model
4. Run document embedding
5. Run required indexing steps
6. Leave the system ready for retrieval evaluation

Do not hard-code OS-specific paths.

Do not require Conda.

Do not assume a particular Python environment manager.

Do not make Bash responsible for embedding logic.

Bash should only orchestrate existing application commands.

If the project already has useful scripts, reuse them rather than creating duplicate implementations.

---

# 18. Model Selection Aliases

Human-friendly aliases are encouraged.

For example:

```text
bge-m3
vnlegal-lal
vietlegal-harrier
vietlegal-e5
jina-v3
deepx
```

These aliases may map to the actual Hugging Face identifiers.

The mapping should be centralized.

Do not spread model names across multiple shell scripts and Python files.

---

# 19. Documentation

Create a new Vietnamese documentation file, for example:

```text
docs/embedding-model-switching.md
```

The exact filename may follow the project's existing documentation conventions.

The document must explain the model-switching process at an **abstract level** and must NOT be tied to a specific operating system.

Do not assume:

* Windows
* Linux
* macOS
* PowerShell
* Bash
* Conda
* virtualenv
* a particular IDE

The document should explain the conceptual workflow first.

---

# 20. Documentation Structure

The new documentation should contain approximately:

```text
# Switching and Benchmarking Embedding Models

## 1. Purpose

## 2. Supported Models

## 3. Architecture

## 4. Model Compatibility

## 5. Stage 1 — Model Download / Preparation

## 6. Stage 2 — Document Embedding

## 7. Stage 3 — Vector Indexing

## 8. Stage 4 — Retrieval

## 9. Stage 5 — Retrieval Evaluation

## 10. Comparing Models

## 11. Avoiding Cross-Model Vector Mixing

## 12. Reproducibility

## 13. Common Errors and Their Meaning
```

Do not add a roadmap or future-work section.

The documentation should describe the system that actually exists after this implementation.

---

# 21. Abstract Model-Switching Workflow

The documentation should communicate a workflow similar to:

```text
Choose embedding model
        ↓
Resolve model configuration
        ↓
Download/load from Hugging Face
        ↓
Generate document embeddings
        ↓
Validate 1024 dimensions
        ↓
Store in model-specific vector space
        ↓
Create/use matching vector index
        ↓
Select same model for query embedding
        ↓
Run retrieval
        ↓
Run test_retrieval
        ↓
Record metrics
        ↓
Compare with other model experiments
```

Make clear that switching models is not simply changing the query model.

The document vectors must correspond to the same model as the query vectors.

---

# 22. Model Comparison

The documentation must explain how to conduct a fair comparison.

Keep constant:

```text
Dataset
Chunking
Metadata-aware representation
Queries
Ground truth
Top-K
ef_search
Distance metric
Evaluation metrics
```

Change only:

```text
Embedding model
```

Where possible, also keep:

```text
hardware
precision
batch configuration
```

consistent.

Do not present external benchmark results as results of this project.

Clearly distinguish:

```text
External benchmark
vs
Project's own retrieval benchmark
```

---

# 23. Comparison Output

The system should make it easy to produce a table such as:

```text
| Model | Dim | Hit@3 | Hit@5 | Hit@10 | Recall@10 | MRR |
|------|-----|-------|-------|--------|-----------|-----|
| BGE-M3 | 1024 | ... | ... | ... | ... | ... |
| VNLegal-LAL | 1024 | ... | ... | ... | ... | ... |
| VietLegal-Harrier | 1024 | ... | ... | ... | ... | ... |
| VietLegal-E5 | 1024 | ... | ... | ... | ... | ... |
| Jina v3 | 1024 | ... | ... | ... | ... | ... |
| DeepX | 1024 | ... | ... | ... | ... | ... |
```

Do not fabricate results.

The implementation should only provide the infrastructure for collecting them.

---

# 24. Existing Project Compatibility

Before modifying anything:

1. Inspect the existing `scripts/embedding.py`
2. Inspect `scripts/index_embeddings.py`
3. Inspect `scripts/retrieval.py`
4. Inspect `scripts/test_retrieval.py`
5. Inspect existing tests related to embedding and retrieval
6. Inspect `.env.example` if present
7. Inspect existing database/schema/index definitions
8. Inspect the current documentation conventions

Understand the current implementation before refactoring.

Do not rewrite working code unnecessarily.

Preserve existing public interfaces where practical.

---

# 25. Backward Compatibility

The existing configuration:

```env
EMBEDDING_MODEL=BAAI/bge-m3
```

must continue to work.

The current BGE-M3 retrieval pipeline must remain functional after the refactor.

The default behavior should remain BGE-M3 unless the project already defines another default.

Existing commands should require as little modification as possible.

---

# 26. Testing Requirements

Add or update tests for the abstraction.

At minimum, test:

### Configuration

* supported model resolves correctly
* unsupported model fails clearly
* default model remains BGE-M3

### Model metadata

* model identifier is correct
* expected dimension is 1024
* backend selection is correct

### Embedding interface

* query embedding returns the expected dimension
* document embedding returns the expected dimension
* query/document interfaces are distinct

### Model isolation

* vectors from different models cannot accidentally use the wrong index/storage

### Regression

* BGE-M3 still works
* existing retrieval evaluation logic remains unchanged

Do not require all six large models to be downloaded and executed during every unit-test run.

Use mocks/stubs or lightweight validation where appropriate.

If integration tests for actual model loading are added, clearly distinguish them from normal unit tests.

---

# 27. Resource Considerations

The target development hardware is a local NVIDIA RTX 3050 6GB Laptop GPU.

Design the implementation so that:

* model loading does not unnecessarily duplicate models in memory
* models can be loaded one at a time
* batch size remains configurable
* precision remains configurable where supported
* GPU/CPU selection remains configurable
* a model that is too large or unsupported produces a clear error rather than silently degrading behavior

Do not introduce unnecessary dependencies.

Do not assume that all six models can remain loaded simultaneously.

The intended workflow is:

```text
Load one model
    ↓
Embed/index/evaluate
    ↓
Release model resources
    ↓
Switch model
    ↓
Repeat
```

---

# 28. Do Not Change Retrieval Methodology

This refactor is about model flexibility.

Do NOT change:

* chunking strategy
* metadata-aware representation
* ground-truth methodology
* evaluation queries
* metric definitions
* similarity calculation
* retrieval filtering
* HNSW methodology

unless a change is strictly required by the model abstraction.

If a compatibility issue requires a methodological change, document it explicitly instead of silently changing it.

---

# 29. No Python Execution Requirement

This task is primarily implementation and documentation work.

Do not run the complete embedding pipeline automatically.

Do not download all six models automatically.

Do not generate 617 embeddings for all models as part of the refactoring task.

Do not run a full retrieval benchmark unless explicitly requested.

The implementation should make those operations possible, but the actual expensive experiments can be executed separately.

Do not modify project files merely to produce experimental results during this task.

---

# 30. No OS/Environment Assumptions in Documentation

The new documentation must describe commands conceptually or provide clearly marked examples without making a specific OS or Conda environment mandatory.

Do not write instructions such as:

```text
conda activate chatbot
```

as a required prerequisite.

The documentation should instead explain:

```text
Use the project's configured Python environment.
Install the required dependencies.
Run the model preparation/embedding/evaluation command.
```

If shell examples are necessary, make them generic and explain what the command accomplishes rather than tying the workflow to a specific OS.

---

# 31. Files to Modify/Create

Determine the exact files after inspecting the repository.

Likely areas include:

```text
scripts/embedding.py
scripts/index_embeddings.py
scripts/retrieval.py
scripts/test_retrieval.py
tests/test_embedding.py
tests/test_retrieval.py
.env.example
docs/embedding-model-switching.md
scripts/run_embedding_experiment.sh
```

Do NOT modify unrelated files.

Do NOT create duplicate implementations.

Do NOT rewrite the entire project architecture.

---

# 32. Final Acceptance Criteria

The implementation is considered successful when all of the following are true:

1. `BAAI/bge-m3` continues to work.
2. All six specified Hugging Face models are represented in the supported-model configuration.
3. Model-specific loading behavior is isolated from retrieval/evaluation logic.
4. Query and document embedding are abstracted separately.
5. All supported models produce exactly 1024-dimensional vectors using a documented/valid method.
6. Different model vector spaces cannot accidentally be mixed.
7. The model can be changed through configuration rather than editing retrieval code.
8. Existing retrieval methodology remains unchanged.
9. `test_retrieval.py` can evaluate whichever supported model is selected.
10. A lightweight Bash orchestration script can run the model-specific pipeline without duplicating application logic.
11. A Vietnamese documentation file explains the complete model-switching and benchmarking workflow.
12. Documentation is OS-agnostic and does not require Conda.
13. Existing BGE-M3 behavior remains backward compatible.
14. Tests cover model resolution, embedding abstraction, dimension validation, and model isolation.
15. No fabricated benchmark results are added.
16. No roadmap or future-work section is introduced.

The final implementation should make the following conceptual operation possible:

```text
Select model
    ↓
Prepare/download from Hugging Face
    ↓
Embed documents
    ↓
Create/use model-specific vector index
    ↓
Run retrieval
    ↓
Run test_retrieval
    ↓
Record metrics
    ↓
Switch model
    ↓
Repeat using the exact same evaluation methodology
```

The most important design principle is:

> **Changing the embedding model should be a configuration/experiment change, not a retrieval-code change.**

Keep the implementation simple, maintainable, and compatible with the project's current architecture.
