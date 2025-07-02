import os
from pathlib import Path
from urllib.parse import urlparse
import argparse
import json
from datetime import datetime
import contextlib
from io import StringIO

import weaviate
from sentence_transformers import SentenceTransformer
from weaviate.connect import ConnectionParams, ProtocolParams
from weaviate.classes.config import Configure, Property, DataType


VECTRO_URL = None
if Path('.codexrc').exists():
    for line in Path('.codexrc').read_text().splitlines():
        if line.startswith('VECTRO_URL='):
            VECTRO_URL = line.split('=', 1)[1].strip()
            break
VECTRO_URL = os.environ.get('VECTRO_URL', VECTRO_URL or 'http://localhost:8080')

EMBED_MODEL = os.environ.get('EMBED_MODEL', 'all-MiniLM-L6-v2')
CLASS_NAME = os.environ.get('CLASS_NAME', 'CortexNote')


def _log_telemetry(task: str, summary: str) -> None:
    path = Path('logs/telemetry.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = json.loads(path.read_text()) if path.exists() else []
        if not isinstance(existing, list):
            existing = []
    except Exception:
        existing = []

    entry = {
        'task': task,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'summary': summary.strip(),
    }
    existing.append(entry)
    path.write_text(json.dumps(existing, indent=2))


def _client_from_url(url: str) -> weaviate.WeaviateClient:
    parsed = urlparse(url)
    secure = parsed.scheme == 'https'
    host = parsed.hostname or 'localhost'
    port = parsed.port or (443 if secure else 80)
    params = ConnectionParams(
        http=ProtocolParams(host=host, port=port, secure=secure),
        grpc=ProtocolParams(host=host, port=50051, secure=secure),
    )
    client = weaviate.WeaviateClient(connection_params=params)
    client.connect()
    return client


def define_schema():
    client = _client_from_url(VECTRO_URL)
    if client.collections.exists(CLASS_NAME):
        print(f"Collection {CLASS_NAME} already exists")
    else:
        client.collections.create(
            name=CLASS_NAME,
            properties=[Property(name="text", data_type=DataType.TEXT)],
            vectorizer_config=Configure.Vectorizer.none(),
        )
        print(f"Created collection {CLASS_NAME}")
    client.close()


def insert_vectors():
    client = _client_from_url(VECTRO_URL)
    coll = client.collections.get(CLASS_NAME)
    model = SentenceTransformer(EMBED_MODEL)
    texts = [
        "Cortex helps you find examples in your code.",
        "Codex stores embeddings inside Vectro for fast search.",
        "Vectro works with Weaviate as the vector database.",
        "This is a simple sample note for similarity search.",
    ]
    vectors = model.encode(texts)
    for text, vector in zip(texts, vectors):
        coll.data.insert({"text": text}, vector=vector.tolist())
    print("Inserted", len(texts), "notes")
    client.close()


def query_similarity():
    client = _client_from_url(VECTRO_URL)
    coll = client.collections.get(CLASS_NAME)
    model = SentenceTransformer(EMBED_MODEL)
    query = "How can Cortex assist with code?"
    q_vec = model.encode([query])[0]
    res = coll.query.near_vector(q_vec.tolist(), limit=2, return_properties=["text"])
    for obj in res.objects:
        text = obj.properties.get("text")
        distance = obj.distance
        print(f"{distance:.4f}: {text}")
    client.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["define_schema", "insert_vectors", "query_similarity"], required=True)
    args = parser.parse_args()

    func = globals()[args.task]
    buffer = StringIO()
    with contextlib.redirect_stdout(buffer):
        func()
    output = buffer.getvalue()
    print(output, end="")
    _log_telemetry(args.task, output)


if __name__ == "__main__":
    main()
