import os
from pathlib import Path
from urllib.parse import urlparse
import argparse
import json
from datetime import datetime
import contextlib
from io import StringIO
import socket

import weaviate
from sentence_transformers import SentenceTransformer
from weaviate.connect import ConnectionParams, ProtocolParams
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery

def get_vectro_url():
    if Path('.codexrc').exists():
        for line in Path('.codexrc').read_text().splitlines():
            if line.startswith('VECTRO_URL='):
                return line.split('=', 1)[1].strip()
    return os.environ.get('VECTRO_URL', 'http://localhost:8181')

VECTRO_URL = get_vectro_url()
EMBED_MODEL = os.environ.get('EMBED_MODEL', 'all-MiniLM-L6-v2')
CLASS_NAME = os.environ.get('CLASS_NAME', 'CortexNote')
QUERY_FILE = Path('query.json')
RESULTS_FILE = Path('results.json')

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

def _check_port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except Exception:
        return False

def _client_from_url(url: str) -> weaviate.WeaviateClient:
    parsed = urlparse(url)
    secure = parsed.scheme == 'https'
    host = parsed.hostname or 'localhost'
    rest_port = parsed.port or (443 if secure else 80)
    grpc_port = 50051

    # Preflight check REST
    if not _check_port_open(host, rest_port):
        raise SystemExit(f"Cannot reach Weaviate REST at {host}:{rest_port} (URL: {url}) - is Docker up? Correct port?")
    # Preflight check gRPC
    if not _check_port_open(host, grpc_port):
        raise SystemExit(f"Cannot reach Weaviate gRPC at {host}:{grpc_port} - is Docker exposing gRPC? (Check docker-compose.yml)")

    params = ConnectionParams(
        http=ProtocolParams(host=host, port=rest_port, secure=secure),
        grpc=ProtocolParams(host=host, port=grpc_port, secure=secure)
    )
    client = weaviate.WeaviateClient(connection_params=params)
    client.connect()  # No skip_init_checks, expect health check to pass!
    return client

def define_schema():
    client = _client_from_url(VECTRO_URL)
    try:
        if client.collections.exists(CLASS_NAME):
            print(f"Collection {CLASS_NAME} already exists")
        else:
            client.collections.create(
                name=CLASS_NAME,
                properties=[Property(name="text", data_type=DataType.TEXT)],
                vectorizer_config=Configure.Vectorizer.none(),
            )
            print(f"Created collection {CLASS_NAME}")
    finally:
        client.close()

def insert_vectors():
    client = _client_from_url(VECTRO_URL)
    try:
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
    finally:
        client.close()

def embed_query():
    query = os.environ.get('INPUT_QUERY', os.environ.get('QUERY', ''))
    if not query:
        raise SystemExit('No query provided via INPUT_QUERY or QUERY')
    model = SentenceTransformer(EMBED_MODEL)
    vector = model.encode([query])[0].tolist()
    QUERY_FILE.write_text(json.dumps({'query': query, 'vector': vector}))
    print(f"Encoded query: {query}")

def query_similarity():
    if not QUERY_FILE.exists():
        raise SystemExit(f"{QUERY_FILE} not found. Run embed_query first.")
    data = json.loads(QUERY_FILE.read_text())
    q_vec = data.get('vector')
    client = _client_from_url(VECTRO_URL)
    try:
        coll = client.collections.get(CLASS_NAME)
        res = coll.query.near_vector(
            q_vec,
            limit=3,
            return_properties=["text"],
            return_metadata=MetadataQuery(distance=True, certainty=True, score=True),
        )
        results = []
        for obj in res.objects:
            # Print available fields for debugging
            print("Object fields:", list(vars(obj).keys()))
            if hasattr(obj, "metadata"):
                print("Metadata fields:", list(vars(obj.metadata).keys()))
            results.append({
                'distance': getattr(obj.metadata, 'distance', None),
                'certainty': getattr(obj.metadata, 'certainty', None),
                'score': getattr(obj.metadata, 'score', None),
                'text': obj.properties.get("text"),
            })
        RESULTS_FILE.write_text(json.dumps(results, indent=2))
    finally:
        client.close()

def emit_results():
    if not RESULTS_FILE.exists():
        raise SystemExit(f"{RESULTS_FILE} not found. Run query_similarity first.")
    results = json.loads(RESULTS_FILE.read_text())
    for item in results[:3]:
        distance = item.get('distance')
        text = item.get('text')
        print(f"{distance:.4f}: {text}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=[
            "define_schema",
            "insert_vectors",
            "embed_query",
            "query_similarity",
            "emit_results",
        ],
        required=True,
    )
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





