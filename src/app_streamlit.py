from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore

try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover
    st.error("faiss-cpu is required. Install with: uv add faiss-cpu")
    raise

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception as e:  # pragma: no cover
    st.error("sentence-transformers is required. Install with: uv add sentence-transformers")
    raise


st.set_page_config(page_title="Semantic Patent Search", layout="wide")


@st.cache_resource(show_spinner=False)
def load_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def load_config(config_path: Path) -> Dict[str, Any]:
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


@st.cache_resource(show_spinner=False)
def load_index(index_path: Path):
    return faiss.read_index(str(index_path))


def l2_normalize(vecs: np.ndarray) -> np.ndarray:
    vecs = vecs.astype(np.float32, copy=True)
    faiss.normalize_L2(vecs)
    return vecs


def patent_key(m: Dict[str, Any]) -> str:
    return str(m.get("doc_number") or m.get("filename") or m.get("title") or m.get("source_path"))


def run_indexing(config_path: Path) -> Tuple[bool, float, Optional[str]]:
    start = time.time()
    try:
        # Call the existing CLI to avoid duplicating logic
        # Use the current interpreter so installed deps are available
        cmd = [sys.executable, "-m", "src.patents_indexer", "--config", str(config_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.time() - start
        if result.returncode != 0:
            return False, duration, result.stderr.strip() or result.stdout.strip()
        return True, duration, result.stdout[-2000:]  # tail for display
    except Exception as e:
        duration = time.time() - start
        return False, duration, str(e)


def search_top_patents_and_best_claims(
    cfg: Dict[str, Any], query_text: str, k: int = 5
) -> List[Dict[str, Any]]:
    index_dir = Path(cfg.get("index_dir", "vector_store"))
    index_path = index_dir / "index.faiss"
    meta_path = index_dir / "metadata.jsonl"
    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Index or metadata not found. Please run indexing first.")

    index = load_index(index_path)
    metadata = load_metadata(meta_path)
    model_name = cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    model = load_model(model_name)

    qvec = model.encode([query_text], convert_to_numpy=True, normalize_embeddings=False).astype(np.float32)
    qvec = l2_normalize(qvec)

    pool = min(max(k * 50, k * 10), len(metadata))
    scores, idxs = index.search(qvec, pool)

    # Precompute claim indices per patent
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

    # Select top-k unique patents
    best_patents: Dict[str, Dict[str, Any]] = {}
    ordered_patents: List[str] = []
    for idx, score in zip(idxs[0].tolist(), scores[0].tolist()):
        if idx < 0 or idx >= len(metadata):
            continue
        m = metadata[idx]
        key = patent_key(m)
        if key not in best_patents:
            best_patents[key] = {"score": float(score), "idx": idx}
            ordered_patents.append(key)
        else:
            if float(score) > best_patents[key]["score"]:
                best_patents[key] = {"score": float(score), "idx": idx}
        if len(ordered_patents) >= k:
            break

    # Best claim per patent (cosine via dot product with normalized vectors)
    qv = qvec[0]
    results: List[Dict[str, Any]] = []
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
        results.append(
            {
                "rank": rank,
                "patent_title": title,
                "doc_number": docnum,
                "best_patent_score": best_patent_score,
                "best_claim_score": best_claim_score,
                "best_claim_number": best_claim.get("claim_number") if best_claim else None,
                "best_claim_text": best_claim.get("text") if best_claim else None,
            }
        )
    return results


def search_top_patents_and_best_claims_graph_adjusted(
    cfg: Dict[str, Any],
    query_text: str,
    k: int = 5,
    pool: int = 100,
    alpha: float = 0.85,
    iters: int = 50,
    tol: float = 1e-8,
) -> List[Dict[str, Any]]:
    index_dir = Path(cfg.get("index_dir", "vector_store"))
    index_path = index_dir / "index.faiss"
    meta_path = index_dir / "metadata.jsonl"
    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Index or metadata not found. Please run indexing first.")

    index = load_index(index_path)
    metadata = load_metadata(meta_path)
    model_name = cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    model = load_model(model_name)

    # Encode and normalize query
    qvec = model.encode([query_text], convert_to_numpy=True, normalize_embeddings=False).astype(np.float32)
    qvec = l2_normalize(qvec)
    qv = qvec[0]

    # Retrieve initial pool (top N chunks)
    pool = int(max(1, min(pool, len(metadata))))
    scores, idxs = index.search(qvec, pool)
    top_scores = scores[0].astype(np.float64, copy=False)
    top_indices = idxs[0].astype(int, copy=False)

    # Reconstruct vectors for top pool
    vectors = []
    valid_meta_indices = []
    for i in top_indices.tolist():
        if i < 0 or i >= len(metadata):
            continue
        try:
            v = index.reconstruct(i)
        except Exception:
            continue
        vectors.append(v.astype(np.float32, copy=False))
        valid_meta_indices.append(i)

    if not vectors:
        return []

    V = np.vstack(vectors).astype(np.float32, copy=False)  # [m, d], rows normalized already
    m = V.shape[0]

    # Personalization vector from initial retrieval scores (nonnegative, normalized)
    idx_to_pos = {idx: pos for pos, idx in enumerate(valid_meta_indices)}
    s = np.zeros((m,), dtype=np.float64)
    for src_score, src_idx in zip(top_scores.tolist(), top_indices.tolist()):
        pos = idx_to_pos.get(int(src_idx))
        if pos is not None:
            s[pos] += max(0.0, float(src_score))
    s_sum = float(s.sum())
    if s_sum <= 0.0:
        s[:] = 1.0 / m
    else:
        s /= s_sum

    # Similarity matrix (cosine via dot product on normalized vectors), nonnegative and no self-loops
    ## THE MAGIC
    S = np.matmul(V, V.T).astype(np.float64, copy=False)
    np.fill_diagonal(S, 0.0)
    np.maximum(S, 0.0, out=S)  # clip negatives to zero

    # Row-normalize to get transition matrix P
    row_sums = S.sum(axis=1, keepdims=True)
    zero_rows = (row_sums == 0.0).reshape(-1)
    P = np.zeros_like(S, dtype=np.float64)
    if np.any(~zero_rows):
        P[~zero_rows] = S[~zero_rows] / row_sums[~zero_rows]
    if np.any(zero_rows):
        P[zero_rows] = 1.0 / m

    # Personalized Random Walk with Restart (RANDOM WALK)
    p = s.copy()
    for _ in range(max(1, iters)):
        new_p = alpha * s + (1.0 - alpha) * (p @ P)
        if np.linalg.norm(new_p - p, ord=1) < tol:
            p = new_p
            break
        p = new_p

    # Rank nodes by p, then dedupe by patent key to top-k
    order = np.argsort(-p).tolist()
    selected_patent_keys: List[str] = []
    first_node_for_patent: Dict[str, int] = {}
    for pos in order:
        meta_idx = valid_meta_indices[pos]
        mrow = metadata[meta_idx]
        key = patent_key(mrow)
        if key not in first_node_for_patent:
            first_node_for_patent[key] = pos
            selected_patent_keys.append(key)
        if len(selected_patent_keys) >= k:
            break

    # Precompute claims and titles per patent
    claims_by_patent: Dict[str, List[int]] = {}
    title_by_patent: Dict[str, Optional[str]] = {}
    docnum_by_patent: Dict[str, Optional[str]] = {}
    for i, mrow in enumerate(metadata):
        key = patent_key(mrow)
        if key not in title_by_patent:
            title_by_patent[key] = mrow.get("title")
            docnum_by_patent[key] = mrow.get("doc_number")
        if str(mrow.get("section")) == "claims":
            claims_by_patent.setdefault(key, []).append(i)

    # Build results
    results: List[Dict[str, Any]] = []
    for rank, key in enumerate(selected_patent_keys, start=1):
        title = title_by_patent.get(key)
        docnum = docnum_by_patent.get(key)
        node_pos = first_node_for_patent[key]
        patent_graph_score = float(p[node_pos])

        # Best claim by dot(q, claim_vec)
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

        results.append(
            {
                "rank": rank,
                "patent_title": title,
                "doc_number": docnum,
                "best_patent_score": patent_graph_score,
                "best_claim_score": best_claim_score,
                "best_claim_number": best_claim.get("claim_number") if best_claim else None,
                "best_claim_text": best_claim.get("text") if best_claim else None,
            }
        )

    return results


def main():
    st.title("Semantic Patent Search")

    # Always-visible run stats (persist in session_state)
    st.sidebar.title("Run Stats")
    last_query_time = st.session_state.get("last_query_time")
    last_query_mode = st.session_state.get("last_query_mode")
    last_query_ts = st.session_state.get("last_query_ts")
    if last_query_time is not None:
        ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_query_ts)) if last_query_ts else "n/a"
        st.sidebar.write(f"Last search: {last_query_time:.2f}s ({last_query_mode or 'unknown'}) at {ts_str}")
    else:
        st.sidebar.write("Last search: none")

    last_index_time = st.session_state.get("last_index_time")
    last_index_ts = st.session_state.get("last_index_ts")
    if last_index_time is not None:
        its_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_index_ts)) if last_index_ts else "n/a"
        st.sidebar.write(f"Last indexing: {last_index_time:.2f}s at {its_str}")
    else:
        st.sidebar.write("Last indexing: none")

    default_config = "config/patents.yaml"
    cfg_path_str = st.text_input("Config path", value=default_config)
    config_path = Path(cfg_path_str).resolve()

    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("Index"):
            with st.spinner("Indexing..."):
                ok, took, tail = run_indexing(config_path)
            if ok:
                st.success(f"Chunking completed in {took:.2f}s")
                st.session_state["last_index_time"] = took
                st.session_state["last_index_ts"] = time.time()
                if tail:
                    st.caption("Indexer output (tail):")
                    st.code(tail, language="text")
            else:
                st.error(f"Indexing failed after {took:.2f}s")
                if tail:
                    st.code(tail, language="text")

    query = st.text_area("Query", height=160, placeholder="e.g., Wheels")

    search_mode = st.radio(
        "Mode",
        options=["Search - Semantic", "Search -- Graph Adjusted"],
        index=0,
        horizontal=True,
    )

    if st.button("Search"):
        if not query.strip():
            st.warning("Please enter a query.")
        else:
            try:
                cfg = load_config(config_path)
                start = time.time()
                if search_mode == "Search -- Graph Adjusted":
                    results = search_top_patents_and_best_claims_graph_adjusted(cfg, query, k=5, pool=100)
                else:
                    results = search_top_patents_and_best_claims(cfg, query, k=5)
                qtime = time.time() - start
                st.session_state["last_query_time"] = qtime
                st.session_state["last_query_mode"] = search_mode
                st.session_state["last_query_ts"] = time.time()
                st.caption(f"Query time: {qtime:.2f}s ({search_mode})")
                if not results:
                    st.info("No results.")
                else:
                    for r in results:
                        st.subheader(r.get("patent_title") or r.get("doc_number") or "(no id)")
                        st.write(f"doc_number: {r.get('doc_number')}, patent_score={r.get('best_patent_score'):.4f}")
                        if r.get("best_claim_text"):
                            cn = r.get("best_claim_number")
                            cn_str = f"Claim {cn}: " if cn is not None else ""
                            st.write(f"{cn_str}{r['best_claim_text']}")
                            if r.get("best_claim_score") is not None:
                                st.caption(f"best_claim_score={r['best_claim_score']:.4f}")
            except FileNotFoundError as e:
                st.error(str(e))
            except Exception as e:
                st.exception(e)


if __name__ == "__main__":
    main()
