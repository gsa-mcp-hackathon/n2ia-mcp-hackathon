
import os
from functools import lru_cache
from pathlib import Path

from databricks.sdk import WorkspaceClient

VOLUME_MODEL_PATH = (
    "/Volumes/users/satish_venkataraman/"
    "huggingface_models/all-MiniLM-L6-v2"
)

LOCAL_MODEL_PATH = Path("/tmp/n2ia-models/all-MiniLM-L6-v2")


def download_model():
    """Copy the model from a UC Volume to local App storage."""
    client = WorkspaceClient()

    def copy_directory(source, destination):
        destination.mkdir(parents=True, exist_ok=True)

        for item in client.files.list_directory_contents(
            directory_path=source
        ):
            target = destination / item.name

            if item.is_directory:
                copy_directory(item.path, target)
            else:
                response = client.files.download(
                    file_path=item.path
                )
                with response.contents as stream:
                    with open(target, "wb") as output:
                        while True:
                            chunk = stream.read(1024 * 1024)
                            if not chunk:
                                break
                            output.write(chunk)

    if not (LOCAL_MODEL_PATH / "modules.json").exists():
        copy_directory(VOLUME_MODEL_PATH, LOCAL_MODEL_PATH)

    return str(LOCAL_MODEL_PATH)


@lru_cache(maxsize=1)
def get_model():
    """Load the embedding model once per App process."""
    from sentence_transformers import SentenceTransformer

    model_path = download_model()
    return SentenceTransformer(model_path, device="cpu")


def calculate_embeddings(descriptions):
    """Return normalized embeddings, reusing Delta-cached vectors when possible."""
    import numpy as np
    from mcp_server_n2ia.services import contract_cache

    descriptions = [d or '' for d in descriptions]
    if not descriptions:
        return np.empty((0, 384), dtype=np.float32)
    cached = {}
    if contract_cache.enabled():
        try:
            cached = contract_cache.load_embeddings(descriptions)
        except Exception:
            # Embedding cache is an optimization, not a correctness dependency.
            cached = {}
    hashes = [contract_cache.description_hash(d) for d in descriptions]
    missing = {h: d for h, d in zip(hashes, descriptions) if h not in cached}
    if missing:
        model = get_model()
        new_vectors = model.encode(
            list(missing.values()), normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        generated = {h: vec.tolist() for h, vec in zip(missing, new_vectors)}
        cached.update(generated)
        if contract_cache.enabled():
            try:
                contract_cache.save_embeddings(generated)
            except Exception:
                pass  # Computed vectors still used for this analysis.
    return np.asarray([cached[h] for h in hashes], dtype=np.float32)
