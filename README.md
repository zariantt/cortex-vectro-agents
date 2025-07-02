# cortex-vectro-agents

## CortexNote Schema
The collection storing notes for retrieval is defined as follows:
- **Class name**: `CortexNote`
- **Properties**:
  - `content` (text)
- **Vectorizer**: none (vectors are embedded manually)

This schema enables Cortex agents to perform RAG retrieval over stored notes.

## Usage
1. Install the required packages:
   ```bash
   pip install weaviate-client sentence-transformers pyyaml
   ```
2. Start a Weaviate instance reachable via the `VECTRO_URL` environment variable (defaults to `http://localhost:8181`).
3. Run the sample pipeline:
   ```bash
   python run_vectro.py
   ```
   This executes the tasks listed in `codex/vectro-index.yaml` to create the schema, insert example notes and run a similarity query.
   Provide a custom query with:
   ```bash
   INPUT_QUERY="What does Cortex do?" python run_vectro.py
   ```
   Results are saved to `results.json`.

See [USER_INSTRUCTIONS.md](USER_INSTRUCTIONS.md) for more details on running individual tasks.
