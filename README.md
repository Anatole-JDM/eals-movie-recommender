# eALS Recommender — Spark-Parallelised Implicit-Feedback Matrix Factorization

A from-scratch PySpark implementation of **eALS** (element-wise Alternating Least
Squares), the implicit-feedback recommendation algorithm from:

> X. He, H. Zhang, M.-Y. Kan, T.-S. Chua, *"Fast Matrix Factorization for Online
> Recommendation with Implicit Feedback"*, SIGIR 2016.

The project trains and evaluates two variants of eALS on the Amazon *Movies &
TV* ratings dataset:

- **eALS-Uniform** (α = 0) — missing interactions are weighted uniformly, as in
  Hu et al. (ICDM 2008).
- **eALS-Popularity** (α = 0.5) — missing interactions are weighted by item
  popularity, the paper's proposed method.

Both are compared on **HR@K** (Hit Ratio) and **NDCG@K** under the standard
leave-one-out offline evaluation protocol.

An interactive **Streamlit app** wraps the pipeline (data loading/filtering,
training, evaluation, recommendation) so the model can be explored without
touching the code, and the whole app is **containerized with Docker** for
reproducible, one-command deployment.

## Contents

- [Architecture](#architecture)
- [Project layout](#project-layout)
- [Running locally](#running-locally)
- [Running the Streamlit app](#running-the-streamlit-app)
- [Running the original batch experiment](#running-the-original-batch-experiment)
- [Tests](#tests)
- [Docker](#docker)
- [Continuous Integration](#continuous-integration)
- [Reproducibility notes](#reproducibility-notes)

## Architecture

```
raw CSV (SNAP Amazon dataset)
   │  download_data()
   ▼
load_and_filter()        binarize ratings → implicit interactions
                          iterative k-core filtering (drop users/items
                          with < min_interactions)
   ▼
encode_ids()              string IDs → dense integer indices
   ▼
leave_one_out_split()     hold out each user's chronologically last
                          interaction as the test instance
   ▼
compute_item_confidence() per-item missing-data weight c_i (Eq. 8)
   ▼
build_training_rdd()      Spark RDD of (user_idx, item_idx, r_ui, w_ui)
   ▼
eALS.fit()                 per-epoch, Spark-parallelised closed-form
                            updates of user/item latent factors (Eq. 12/13)
   ▼
Evaluator.evaluate()       HR@K / NDCG@K on the held-out test interactions
```

Spark's role is specifically in the training loop: the paper proves that,
given fixed item factors and a small `K×K` cache matrix, every user's factor
vector can be updated independently (and symmetrically for items). This
project maps those updates over grouped RDDs (`user_items_rdd`,
`item_users_rdd`) each epoch, while the cheap `K×K` cache matrices are
computed on the driver via NumPy/BLAS.

## Project layout

```
config.py          Central hyperparameter configuration (dataclass)
data_loader.py      Data download, filtering, encoding, splitting, confidence weights
eals_spark.py        Spark-parallelised eALS model (fit / predict)
evaluation.py         HR@K / NDCG@K evaluator (leave-one-out protocol)
main.py                Original CLI experiment runner (Uniform vs. Popularity)
app.py                  Streamlit app wrapping the same pipeline interactively
tests/                   Unit tests for the data import/filtering functions
Dockerfile                Container image (Python + JDK + Spark + Streamlit)
docker-compose.yml         One-command local run
.github/workflows/ci.yml    CI: tests + Docker build on every push/PR
```

## Running locally

Requires **Python 3.11+** and a **JDK 17** (PySpark needs a JVM).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
```

## Running the Streamlit app

```bash
streamlit run app.py
```

Then open http://localhost:8501. The sidebar exposes every hyperparameter
from `config.py` (filtering threshold, sample fraction, K, λ, c0, α, epochs,
top-K). The app has four tabs:

1. **Data** — download/filter the dataset and inspect summary statistics.
2. **Train** — train a single eALS model with a live HR@K/NDCG@K chart.
3. **Compare weighting** — run both α=0 and α=0.5 variants and compare them.
4. **Recommend** — generate top-N item recommendations for a given user.

By default the sidebar uses a small sample fraction and epoch count so the
demo stays responsive; increase them for closer-to-paper results (slower).

## Running the original batch experiment

The original non-interactive experiment script (both variants, full metrics,
CSV export) still works standalone:

```bash
python main.py
```

## Tests

Unit tests cover the pandas-only data import/filtering functions in
`data_loader.py` (`load_and_filter`, `encode_ids`, `leave_one_out_split`,
`compute_item_confidence`) using small synthetic CSVs — no network access or
Spark session required, so they run in a few seconds:

```bash
pytest tests/ -v
```

## Docker

Build and run the containerized Streamlit app:

```bash
docker build -t eals-streamlit .
docker run -p 8501:8501 -v "$(pwd)/data:/app/data" eals-streamlit
```

or with Docker Compose:

```bash
docker compose up --build
```

Then open http://localhost:8501. The `data/` volume mount lets the dataset
download persist across container restarts instead of being re-downloaded
each time.

> **Note on image size:** the image bundles a JRE 17 to run PySpark inside
> the container faithfully to the original implementation, so it is larger
> (~1.5–2 GB) than a typical pure-Python Streamlit image.

## Continuous Integration

`.github/workflows/ci.yml` runs on every push/PR to `main`:

1. Sets up Python 3.11 and Java 17.
2. Installs dependencies.
3. Runs the `pytest` suite.
4. Builds the Docker image to catch container regressions early.

## Reproducibility notes

- The raw dataset (~190 MB) is **not** committed to the repo; `download_data()`
  fetches it on first run and caches it under `data/`, which is gitignored.
- All randomness (sampling, factor initialization, evaluation user
  sub-sampling) is seeded via `config.py`'s `seed` field for repeatable runs.
- `requirements.txt` pins minimum versions for the runtime app;
  `requirements-dev.txt` adds `pytest` for local/CI testing.
- Hyperparameters match the paper's SIGIR 2016 defaults (`λ=0.01`,
  `c0` search range, `α≈0.5`) as documented in `config.py`.
