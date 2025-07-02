# User Instructions

These notes describe how to run the tasks in `vectro_test.py`.

## Requirements
- Python 3.11+
- A running Weaviate instance. Set `VECTRO_URL` to its endpoint. The default is `http://localhost:8181`.
- Packages `weaviate-client`, `sentence-transformers` and `pyyaml`.

Install dependencies with:
```bash
pip install weaviate-client sentence-transformers pyyaml
```

## Running individual tasks
The script `vectro_test.py` exposes several tasks. Examples:
```bash
python vectro_test.py --task define_schema
python vectro_test.py --task insert_vectors
INPUT_QUERY="hello world" python vectro_test.py --task embed_query
python vectro_test.py --task query_similarity
python vectro_test.py --task emit_results
```

The `embed_query` task creates `query.json`. The `query_similarity` task writes search results to `results.json` and prints details to the console.

## Running the pipeline
To execute the default sequence of tasks (`define_schema`, `insert_vectors` and `query_similarity`) run:
```bash
python run_vectro.py
```
Outputs and short logs are stored under `logs/`.
