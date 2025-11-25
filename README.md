### OKAY. Here we go. Attempt 2

My first attempt I tried to implement a FastAPI backend and a Next.JS frontend. I went way to fast and database migrations caused me a lot of issues.

This time, I am going to stick to the task at hand.

First, I am going to create a search egine using semantic chunking and cosine similarity. It will be very very basic. All of the indeces will be saved locally and easily implemented. If a JSON does not have all the required data, I drop it none of it is chunked. This will give me the optiopn later do add things on top of my semantic search -- hybrid search, knowledge graph, etc. I am going to allow queries to natural language queries. The results will return the top 5 most similar patents (Title and doc_number) and the most similar claim to the query in each of those patents. See Set up below for part 1:


STEP BY STEP

Setup:
- Initialize (only once if you don’t already have a project):
  - `uv init`
  - `uv venv`
- Install deps:
  - `uv add sentence-transformers faiss-cpu numpy tqdm orjson pydantic click pyyaml`

Configure:
- Edit `config/patents.yaml`
  - `data_dir`: set to your folder, e.g. `data/patent_data_small`
  - `index_dir`: defaults to `vector_store`
  - `model_name`: `sentence-transformers/all-MiniLM-L6-v2`
  - Chunking: `size: 1200`, `overlap: 200`
  - Fields indexed: `title`, `abstract`, `detailed_description`, `claims`
    - Note: `filename` and `bibtex` are NOT chunked/indexed
  - Metadata stored per chunk: includes `title`, `filename`, `section`, `chunk_idx`, `source_path`, and `text` (for convenience)
  - Required fields rule: a patent is skipped entirely if any of `title`, `abstract`, `detailed_description`, or `claims` is missing or empty

Build the index:
- Downloads the embedding model on first run; streams and batches embeddings.
```
uv run python -m src.patents_indexer --config config/patents.yaml
```
Artifacts:
- `vector_store/index.faiss` — FAISS inner-product index (cosine via L2-normalization)
- `vector_store/metadata.jsonl` — one JSON line per chunk (aligned by index)
- `vector_store/stats.json` — build stats (`num_vectors`, `dim`, `skipped_missing_required`, etc.)

Query the index:
```
uv run python -m src.patents_query --config config/patents.yaml --query "lithium battery electrolyte additive" --k 5
```
Output shows:
- rank, score, id (prefers `title`, falls back to `filename`), section, chunk number
- a short snippet of the chunk text

How similarity works:
- We L2-normalize all embeddings and use FAISS `IndexFlatIP` so inner product ≈ cosine similarity.

### Streamlit UI (optional)

Install once:
- `uv add streamlit`

Run:
```
uv run streamlit run src/app_streamlit.py -- --config config/patents.yaml
```
UI provides:
- Index button (shows elapsed time and indexing output tail)
- Query box returning top-5 patents with best claim per patent (title, doc_number, scores)


## Part 2

So, I am going to tackle one and a half problems. First, my streamlit UI gives a gooid terminal interfacer that users can interact with. However, due to time constraints I am not going to deal with log in or simultaneous use. However, on the "failed" attempt, the easiest adjustment would be to add a prisma database to give users accounts and allow them to log in (I have used google auth for this before).

For the full problem I am going to complete, I will do my best to improve my basic semantic algorithm with a graph bases reranking. I plan to take the top 100 candidates, then make a similarity matrix between them (100 x 100). I will then perform Random Walk Reranking and return the top 5 canditates used from my random walk reranking.

## Search modes

### Graph Adjusted (Random-Walk reranking)
- Pool selection:
  - Retrieve the top \(N=100\) chunks from FAISS for the query.
- Graph construction:
  - Reconstruct chunk vectors \(V \in \mathbb{R}^{N \times d}\) (L2-normalized rows).
  - Build similarity matrix \(S = V V^\top\); set diagonal to 0 and clamp negatives to 0.
  - Row-normalize \(S\) to obtain transition matrix \(P\). If a row sum is 0, use uniform transitions for that row.
- Personalization:
  - Let \(s \in \mathbb{R}^N\) be the seed distribution from FAISS scores over the \(N\) nodes, normalized to sum to 1 (fallback to uniform if all zeros).
- Random walk with restart:
  - Iterate \(p_{t+1} = \alpha s + (1-\alpha)\, p_t P\) until convergence or a max number of iterations.
  - Typical parameters: \(N=100\), \(\alpha=0.85\), max iters \(=50\), tolerance \(=10^{-8}\) (L1 norm).
- Patent aggregation:
  - Sort nodes by final score \(p\), deduplicate by patent key, and keep the top-5 patents.
- Best claim per patent:
  - Same as the baseline: choose the patent’s best claim by cosine with the query.

### Streamlit UI usage
- Index:
  - Click “Index” to build or rebuild the FAISS index. The sidebar shows last indexing duration and timestamp.
- Query:
  - Enter query text.
  - Choose mode:
    - “Search - Semantic” (baseline)
    - “Search -- Graph Adjusted” (graph reranking)
  - Click “Search”. The sidebar shows last search duration, timestamp, and mode.

## Math details for the rank algorithm/RANDOM WALK

- Similarity:
  \[
  S = \max(0, V V^\top), \quad \text{with } \operatorname{diag}(S) = 0
  \]
  where rows of \(V\) are L2-normalized chunk embeddings.

- Transition matrix:
  \[
  P_{ij} =
  \begin{cases}
  \dfrac{S_{ij}}{\sum_k S_{ik}}, & \sum_k S_{ik} > 0 \\
  \dfrac{1}{N}, & \text{otherwise (uniform row)}
  \end{cases}
  \]

- Personalized PageRank / Random Walk with Restart:
  \[
  p_{t+1} = \alpha s + (1-\alpha) \, p_t P
  \]
  with \(p_0 = s\), \(\alpha \in (0,1)\), iterate until \(\lVert p_{t+1} - p_t \rVert_1 < \varepsilon\) or until a max iteration cap.

- Final ranking and aggregation:
  - Rank nodes by \(p\) descending.
  - Deduplicate by patent key; keep top \(k=5\) patents.
  - For each, choose the best claim by cosine with the query vector.
