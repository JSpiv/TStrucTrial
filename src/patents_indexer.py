from __future__ import annotations

import json
import re
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import click
import numpy as np

try:
    import orjson  # type: ignore
except Exception:  # pragma: no cover
    orjson = None  # type: ignore

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "faiss-cpu is required. Install with: uv add faiss-cpu"
    ) from e

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "sentence-transformers is required. Install with: uv add sentence-transformers"
    ) from e

from tqdm import tqdm  # type: ignore

from src.utils.chunk_text import clean_text, chunk_text_by_chars


@dataclass
class Config:
    data_dir: Path
    index_dir: Path
    model_name: str
    batch_size: int
    chunk_size: int
    chunk_overlap: int
    fields: List[str]
    id_fields: List[str]
    store_text_in_metadata: bool


def load_config(config_path: Path) -> Config:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    data: Dict[str, Any]
    parsed: Optional[Dict[str, Any]] = None
    if yaml is not None:
        try:
            parsed = yaml.safe_load(text)
        except Exception:
            parsed = None
    if parsed is None:
        try:
            parsed = json.loads(text)
        except Exception:
            raise RuntimeError(
                "Failed to parse config. Install PyYAML (uv add pyyaml) or use JSON."
            )
    data = parsed or {}
    chunk = (data.get("chunk") or {}) if isinstance(data.get("chunk"), dict) else {}
    return Config(
        data_dir=Path(data.get("data_dir", "data")),
        index_dir=Path(data.get("index_dir", "vector_store")),
        model_name=str(data.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")),
        batch_size=int(data.get("batch_size", 256)),
        chunk_size=int(chunk.get("size", 1200)),
        chunk_overlap=int(chunk.get("overlap", 200)),
        fields=list(data.get("fields", ["title", "abstract", "claims", "description"])),
        id_fields=list(
            data.get("id_fields", ["publication_number", "patent_number", "application_number"])
        ),
        store_text_in_metadata=bool(data.get("store_text_in_metadata", True)),
    )


def find_json_files(root: Path) -> List[Path]:
    return [Path(p) for p in root.rglob("*.json")]


def load_json(path: Path) -> Dict[str, Any]:
    raw = path.read_bytes()
    if orjson is not None:
        return orjson.loads(raw)  # type: ignore
    return json.loads(raw.decode("utf-8"))


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        # Prefer common text keys if present
        for key in ("text", "value", "content"):
            if key in value and isinstance(value[key], (str, int, float, bool)):
                return str(value[key])
        # Fallback to JSON-serialized
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
    if isinstance(value, (list, tuple)):
        return " ".join([stringify(v) for v in value])
    return str(value)


def select_patent_id(doc: Dict[str, Any], id_fields: List[str], fallback: str) -> str:
    for key in id_fields:
        val = doc.get(key)
        if isinstance(val, (str, int)):
            s = str(val).strip()
            if s:
                return s
    return fallback


def extract_sections(
    doc: Dict[str, Any], fields: List[str]
) -> List[Tuple[str, str]]:
    sections: List[Tuple[str, str]] = []
    for field in fields:
        if field in doc:
            text = stringify(doc[field])
            text = clean_text(text)
            if text:
                sections.append((field, text))
    # Fallback if nothing found: flatten all string-like fields
    if not sections:
        text_bits: List[str] = []
        for k, v in doc.items():
            if isinstance(v, (str, int, float, bool, list, dict)):
                s = stringify(v)
                if s:
                    text_bits.append(s)
        combined = clean_text(" ".join(text_bits))
        if combined:
            sections.append(("content", combined))
    return sections


def split_claims(claims_value: Any) -> List[Tuple[Optional[int], str]]:
    """
    Split claims into individual (claim_number, claim_text) pairs.
    - If value is a list, treat each entry as a claim; number sequentially if not detectable.
    - If value is a string blob, split on leading numbered headings like '1.' or '2)' at line starts.
    """
    results: List[Tuple[Optional[int], str]] = []
    # List form
    if isinstance(claims_value, list):
        for idx, entry in enumerate(claims_value, start=1):
            text = clean_text(stringify(entry))
            if not text:
                continue
            # Try to detect explicit number at the start
            num: Optional[int] = None
            m = re.match(r"^\s*(\d{1,4})[\.\)]\s+", text)
            if m:
                try:
                    num = int(m.group(1))
                except Exception:
                    num = None
            if num is None:
                num = idx
            results.append((num, text))
        return results
    # String form
    blob = clean_text(stringify(claims_value))
    if not blob:
        return results
    # Find positions of numbered claims at line starts
    pattern = re.compile(r"(?m)^\s*(\d{1,4})[\.\)]\s+")
    starts: List[Tuple[int, Optional[int]]] = []
    for m in pattern.finditer(blob):
        try:
            num = int(m.group(1))
        except Exception:
            num = None
        starts.append((m.start(), num))
    if not starts:
        # No obvious numbering; treat whole blob as one claim without number
        results.append((None, blob))
        return results
    # Slice into segments
    for i, (pos, num) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(blob)
        seg = clean_text(blob[pos:end])
        if seg:
            results.append((num, seg))
    return results


def has_all_required_fields(doc: Dict[str, Any], required_fields: List[str]) -> bool:
    for field in required_fields:
        raw = doc.get(field)
        text = clean_text(stringify(raw))
        if not text:
            return False
    return True


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int) -> np.ndarray:
    embeddings: List[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding", unit="batch"):
        batch = texts[i : i + batch_size]
        vecs = model.encode(batch, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=False)
        embeddings.append(vecs.astype(np.float32, copy=False))
    if not embeddings:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
    matrix = np.vstack(embeddings)
    # L2-normalize for cosine similarity via inner product
    faiss.normalize_L2(matrix)
    return matrix


def build_index(vectors: np.ndarray) -> faiss.Index: ## FAISS indes uses inner product here when vectors are normalized
    if vectors.ndim != 2:
        raise ValueError("vectors must be a 2D array")
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


@click.command()
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
def main(config_path: Path) -> None:
    cfg = load_config(config_path)
    ensure_dir(cfg.index_dir)

    files = find_json_files(cfg.data_dir)
    if not files:
        click.echo(f"No JSON files found in {cfg.data_dir}")
        return

    # Load and chunk
    chunks: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    skipped_missing_required = 0
    for path in tqdm(files, desc="Parsing JSON", unit="file"):
        try:
            loaded = load_json(path)
        except Exception:
            # Skip unreadable file
            continue
        # Normalize to a list of patent dicts
        if isinstance(loaded, list):
            docs_iter = [d for d in loaded if isinstance(d, dict)]
        elif isinstance(loaded, dict):
            docs_iter = [loaded]
        else:
            continue

        for doc in docs_iter:
            # Skip entire patent if any required field is missing or empty
            if not has_all_required_fields(doc, cfg.fields):
                skipped_missing_required += 1
                continue
            # Capture optional metadata fields for each chunk
            meta_title = clean_text(stringify(doc.get("title"))) if "title" in doc else ""
            meta_filename = clean_text(stringify(doc.get("filename"))) if "filename" in doc else ""
            meta_doc_number = clean_text(stringify(doc.get("doc_number"))) if "doc_number" in doc else ""

            sections = extract_sections(doc, cfg.fields)
            for section_name, text in sections:
                if section_name == "claims":
                    # Split per numbered claim; further chunk if an individual claim is long
                    raw_claims = doc.get("claims", text)
                    per_claim = split_claims(raw_claims)
                    for claim_number, claim_text in per_claim:
                        claim_chunks = chunk_text_by_chars(claim_text, cfg.chunk_size, cfg.chunk_overlap)
                        for idx, chunk in enumerate(claim_chunks):
                            vector_index = len(chunks)
                            chunks.append(chunk)
                            meta: Dict[str, Any] = {
                                "vector_index": vector_index,
                                "section": section_name,
                                "chunk_idx": idx,
                                "source_path": str(path),
                            }
                            if claim_number is not None:
                                meta["claim_number"] = int(claim_number)
                            if meta_title:
                                meta["title"] = meta_title
                            if meta_filename:
                                meta["filename"] = meta_filename
                            if meta_doc_number:
                                meta["doc_number"] = meta_doc_number
                            if cfg.store_text_in_metadata:
                                meta["text"] = chunk
                            metadatas.append(meta)
                    continue
                # Default: regular section chunking
                section_chunks = chunk_text_by_chars(text, cfg.chunk_size, cfg.chunk_overlap)
                for idx, chunk in enumerate(section_chunks):
                    vector_index = len(chunks)
                    chunks.append(chunk)
                    meta = {
                        "vector_index": vector_index,
                        "section": section_name,
                        "chunk_idx": idx,
                        "source_path": str(path),
                    }
                    if meta_title:
                        meta["title"] = meta_title
                    if meta_filename:
                        meta["filename"] = meta_filename
                    if meta_doc_number:
                        meta["doc_number"] = meta_doc_number
                    if cfg.store_text_in_metadata:
                        meta["text"] = chunk
                    metadatas.append(meta)

    if not chunks:
        click.echo("No chunks produced; nothing to index.")
        return

    # Embed
    click.echo(f"Loading embedding model: {cfg.model_name}")
    model = SentenceTransformer(cfg.model_name)
    vectors = embed_texts(model, chunks, cfg.batch_size)

    # Build FAISS
    index = build_index(vectors)

    # Persist artifacts
    index_path = cfg.index_dir / "index.faiss"
    meta_path = cfg.index_dir / "metadata.jsonl"
    stats_path = cfg.index_dir / "stats.json"
    faiss.write_index(index, str(index_path))
    write_jsonl(meta_path, metadatas)
    stats = {
        "model_name": cfg.model_name,
        "dim": int(vectors.shape[1]),
        "num_vectors": int(vectors.shape[0]),
        "created_at": int(time.time()),
        "data_dir": str(cfg.data_dir),
        "index_path": str(index_path),
        "metadata_path": str(meta_path),
        "chunk_size": cfg.chunk_size,
        "chunk_overlap": cfg.chunk_overlap,
        "skipped_missing_required": int(skipped_missing_required),
    }
    (cfg.index_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    click.echo(f"Wrote index: {index_path}")
    click.echo(f"Wrote metadata: {meta_path}")
    click.echo(f"Wrote stats: {stats_path}")


if __name__ == "__main__":
    main()
