from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import numpy as np

try:
    import yaml  # type: ignore
except Exception:
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


def load_config_path(config_path: Path) -> Dict[str, Any]:
    text = config_path.read_text(encoding="utf-8")
    if yaml is not None:
        try:
            return yaml.safe_load(text)
        except Exception:
            pass
    return json.loads(text)


def load_metadata(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def l2_normalize(vecs: np.ndarray) -> np.ndarray:
    # In-place safe copy
    vecs = vecs.astype(np.float32, copy=True)
    faiss.normalize_L2(vecs)
    return vecs


@click.command()
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--query", "query_text", type=str, required=True, help="Natural language query")
@click.option("--k", "k", type=int, default=5, show_default=True, help="Top-k results to return")
@click.option("--section", "section_filter", type=str, default=None, help="Optional section filter (e.g., claims)")
@click.option("--save-json", "save_json_path", type=click.Path(dir_okay=False, path_type=Path), default=None)
def main(
    config_path: Path,
    query_text: str,
    k: int,
    section_filter: Optional[str],
    save_json_path: Optional[Path],
) -> None:
    cfg = load_config_path(config_path)
    index_dir = Path(cfg.get("index_dir", "vector_store"))
    model_name = cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    index_path = index_dir / "index.faiss"
    meta_path = index_dir / "metadata.jsonl"
    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Index or metadata not found. Build the index first.")

    index = faiss.read_index(str(index_path))
    metadata = load_metadata(meta_path)
    model = SentenceTransformer(model_name)
    qvec = model.encode([query_text], convert_to_numpy=True, normalize_embeddings=False).astype(np.float32)
    qvec = l2_normalize(qvec) ## Vector in 2d

    # Search broadly to select top patents (by best-matching chunk across any section)
    pool = min(max(k * 50, k * 10), len(metadata))
    scores, idxs = index.search(qvec, pool)

    def patent_key(m: Dict[str, Any]) -> str:
        return str(m.get("doc_number") or m.get("filename") or m.get("title") or m.get("source_path"))

    # Precompute claim indices per patent for quick lookup
    claims_by_patent: Dict[str, List[int]] = {}
    title_by_patent: Dict[str, Optional[str]] = {}
    docnum_by_patent: Dict[str, Optional[str]] = {}
    for i, m in enumerate(metadata):
        key = patent_key(m)
        if key not in title_by_patent:
            title_by_patent[key] = m.get("title")
            docnum_by_patent[key] = m.get("doc_number")
        if str(m.get("section")) == "claims":
            claims_by_patent.setdefault(key, []).append(i)

    # Collect best patent hits
    best_patents: Dict[str, Dict[str, Any]] = {}
    ordered_patents: List[str] = []
    for idx, score in zip(idxs[0].tolist(), scores[0].tolist()):
        if idx < 0 or idx >= len(metadata):
            continue
        m = metadata[idx]
        if section_filter and str(m.get("section")) != section_filter:
            continue
        key = patent_key(m)
        if key not in best_patents:
            best_patents[key] = {"score": float(score), "idx": idx}
            ordered_patents.append(key)
        else:
            if float(score) > best_patents[key]["score"]:
                best_patents[key] = {"score": float(score), "idx": idx}
        if len(ordered_patents) >= k:
            break

    # For each selected patent, find the most similar claim by reconstructing vectors
    qv = qvec[0]
    final_results: List[Dict[str, Any]] = []
    for rank, key in enumerate(ordered_patents, start=1):
        title = title_by_patent.get(key)
        docnum = docnum_by_patent.get(key)
        best_patent_score = best_patents[key]["score"]
        best_claim = None
        best_claim_score = None
        for ci in claims_by_patent.get(key, []):
            try:
                vec = index.reconstruct(ci)
            except Exception:
                continue
            score = float(np.dot(qv, vec))
            if (best_claim_score is None) or (score > best_claim_score):
                best_claim_score = score
                best_claim = metadata[ci]
        final_results.append(
            {
                "rank": rank,
                "patent_title": title,
                "doc_number": docnum,
                "best_patent_score": best_patent_score,
                "best_claim_score": best_claim_score,
                "best_claim_number": best_claim.get("claim_number") if best_claim else None,
                "best_claim_chunk_idx": best_claim.get("chunk_idx") if best_claim else None,
                "best_claim_text": best_claim.get("text") if best_claim else None,
                "best_claim_source_path": best_claim.get("source_path") if best_claim else None,
            }
        )

    # Print concise table
    if not final_results:
        click.echo("No results.")
        return
    for r in final_results:
        ident = r.get("patent_title") or r.get("doc_number") or "(no id)"
        click.echo(f"[{r['rank']}] patent_score={r['best_patent_score']:.4f} title={ident} doc={r.get('doc_number')}")
        snippet = (r.get("best_claim_text") or "")[:240].replace("\n", " ")
        if snippet:
            if r.get("best_claim_score") is not None:
                cn = r.get("best_claim_number")
                cn_str = f" claim={cn}" if cn is not None else ""
                click.echo(f"    best_claim_score={r['best_claim_score']:.4f}{cn_str}  {snippet}...")
            else:
                click.echo(f"    (no claims found)")

    if save_json_path:
        with save_json_path.open("w", encoding="utf-8") as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        click.echo(f"Saved results to {save_json_path}")


if __name__ == "__main__":
    main()
