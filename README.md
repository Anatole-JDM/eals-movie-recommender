# eALS Recommender — Spark-Parallelised Implicit-Feedback Matrix Factorization

[![CI](https://github.com/Anatole-JDM/eals-movie-recommender/actions/workflows/ci.yml/badge.svg)](https://github.com/Anatole-JDM/eals-movie-recommender/actions/workflows/ci.yml)

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
- [Requirements](#requirements)
- [Streamlit app](#streamlit-app)
- [Training](#training)
- [Evaluation](#evaluation)
- [Pretrained models](#pretrained-models)
- [Results](#results)
- [Docker](#docker)
- [Tests](#tests)
- [Continuous integration](#continuous-integration)
- [Development tooling](#development-tooling)
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
config.py                   Central hyperparameter configuration (dataclass)
data_loader.py               Data download, filtering, encoding, splitting, confidence weights
eals_spark.py                 Spark-parallelised eALS model (fit / predict)
evaluation.py                  HR@K / NDCG@K evaluator (leave-one-out protocol)
main.py                         Batch experiment runner (Uniform vs. Popularity)
app.py                           Streamlit app wrapping the same pipeline interactively
tests/                            Unit tests for the data import/filtering functions
Dockerfile                         Container image (Python + JDK + Spark + Streamlit)
docker-compose.yml                  One-command local run
.github/workflows/ci.yml             CI: tests + Docker build on every push/PR
.pre-commit-config.yaml               Pre-commit hooks (formatting, linting, hygiene)
ruff.toml                              Ruff linter/formatter configuration
```

## Requirements

- **Python 3.11+**
- A **JDK** (Java 17 or 21) — PySpark needs a JVM at runtime
- Dependencies are pinned in `requirements.txt` (runtime) and
  `requirements-dev.txt` (adds `pytest`, `ruff`, `pre-commit` for development)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
```

To reproduce the exact environment used to produce the results below, install
from `requirements.txt` unmodified — every dependency is pinned to a specific
version, not a floor (`==`, not `>=`).

## Streamlit app

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
demo stays responsive in a browser session; increase them to reproduce the
scale of the [Results](#results) below (slower).

## Training

The batch script trains both variants back-to-back with the hyperparameters
declared in `config.py` (K=64, λ=0.01, c0=64, 20 epochs, 20% of the raw
dataset sampled before filtering):

```bash
python main.py
```

This downloads the dataset on first run (cached under `data/`), builds the
Spark training RDDs, and prints per-epoch HR@100/NDCG@100 for both
`eALS-Uniform` (α=0) and `eALS-Popularity` (α=0.5), followed by a summary
comparison. Per-epoch histories are also written to `results_uniform.csv`
and `results_popularity.csv`.

To train a single variant interactively instead, use the Streamlit app's
**Train** tab, which exposes every hyperparameter as a sidebar control.

## Evaluation

Evaluation is built into the training loop rather than a separate script:
`Evaluator.evaluate()` runs after every epoch, computing **HR@K** and
**NDCG@K** under the leave-one-out protocol (Section 5.1 of the paper):

- The test set holds exactly one interaction per user — their chronologically
  last one.
- Items already seen in training are excluded from the candidate ranking.
- HR@K is the fraction of test users whose held-out item lands in the
  top-K ranked list; NDCG@K additionally rewards ranking it higher within
  that list.

`main.py` and the Streamlit app's **Compare weighting** tab both call this
same evaluator, so the numbers in the [Results](#results) table below are
reproduced by the exact `python main.py` command above — no separate
evaluation script or extra flags are needed.

## Pretrained models

Not applicable: eALS trains from randomly initialised factors in a few
minutes on a laptop at the scale used here (see [Results](#results)), so no
pretrained checkpoints are distributed. `eALS.__init__` seeds factor
initialisation via `config.py`'s `seed` field for repeatable training runs
from scratch.

## Results

Reproduced with `python main.py` (unmodified `config.py` defaults: K=64,
λ=0.01, c0=64, 20 epochs, 20% sample, min_interactions=10) on the Amazon
*Movies & TV* dataset (812 users, 1,057 items, 15,595 train / 812 test
interactions after filtering):

| Model                    | Best Epoch | HR@100 | NDCG@100 | Avg time/epoch (s) |
|--------------------------|:----------:|:------:|:--------:|:------------------:|
| eALS-Uniform (α=0)       |     18     | 0.1687 |  0.0372  |        8.06         |
| eALS-Popularity (α=0.5)  |      9     | 0.1761 |  0.0386  |        8.85         |

Relative improvement of popularity-aware weighting over uniform weighting:

- HR@100: **+4.39%**
- NDCG@100: **+3.76%**

This matches the paper's qualitative finding — weighting missing data by
item popularity (α=0.5) outperforms the uniform baseline (α=0) — at a
modest ~10% extra cost per epoch. Full per-epoch curves for this run are in
`results_uniform.csv` / `results_popularity.csv` after running the command
above.

## Docker

A pre-built image is published on Docker Hub:
**[anatolejdm/eals-movie-recommender](https://hub.docker.com/r/anatolejdm/eals-movie-recommender)**

Pull and run it directly — no clone or build required:

```bash
docker pull anatolejdm/eals-movie-recommender:latest
docker run -p 8501:8501 -v "$(pwd)/data:/app/data" anatolejdm/eals-movie-recommender
```

Or build it yourself from source:

```bash
docker build -t eals-streamlit .
docker run -p 8501:8501 -v "$(pwd)/data:/app/data" eals-streamlit
```

or with Docker Compose (builds from source):

```bash
docker compose up --build
```

Then open http://localhost:8501. The `data/` volume mount lets the dataset
download persist across container restarts instead of being re-downloaded
each time. The image also bundles `main.py`, so the exact command from
[Training](#training) reproduces the same results inside the container:

```bash
docker run --rm --entrypoint python -v "$(pwd)/data:/app/data" anatolejdm/eals-movie-recommender main.py
```

> **Note on image size:** the image bundles a JRE to run PySpark inside
> the container faithfully to the original implementation, so it is larger
> (~1.5–2 GB) than a typical pure-Python Streamlit image.

## Tests

Unit tests cover the pandas-only data import/filtering functions in
`data_loader.py` (`load_and_filter`, `encode_ids`, `leave_one_out_split`,
`compute_item_confidence`) using small synthetic CSVs — no network access or
Spark session required, so they run in a few seconds:

```bash
pytest tests/ -v
```

With coverage:

```bash
pytest tests/ --cov=data_loader --cov-report=term-missing
```

## Continuous integration

`.github/workflows/ci.yml` runs on every push/PR to `main`:

1. Sets up Python 3.11 and Java 17.
2. Installs dependencies.
3. Runs the `pytest` suite.
4. Builds the Docker image to catch container regressions early.

## Development tooling

Code quality is enforced with [pre-commit](https://pre-commit.com/) hooks,
configured in `.pre-commit-config.yaml`:

- **Ruff** — linting (`ruff check`) and formatting (`ruff format`), configured
  in `ruff.toml`.
- **pre-commit-hooks** — trailing whitespace, end-of-file fixer, YAML
  validation, and a large-file guard.

To enable them locally:

```bash
pip install -r requirements-dev.txt
pre-commit install
```

Hooks then run automatically on every `git commit`; to run them on demand
against the whole repo:

```bash
pre-commit run --all-files
```

Docstrings follow a consistent convention throughout the codebase: a short
summary line in descriptive third-person form (e.g. "Fits the model."),
`:param:`/`:return:` fields with no duplicated type information (types live
in the function signature via type hints), and a blank line separating the
summary from a longer body where one is needed.

## Reproducibility notes

- The raw dataset (~190 MB) is **not** committed to the repo; `download_data()`
  fetches it on first run and caches it under `data/`, which is gitignored.
- All randomness (sampling, factor initialization, evaluation user
  sub-sampling) is seeded via `config.py`'s `seed` field for repeatable runs.
- `requirements.txt` pins exact versions for the runtime app;
  `requirements-dev.txt` layers `pytest`, `ruff`, and `pre-commit` on top for
  local/CI development.
- Hyperparameters match the paper's SIGIR 2016 defaults (`λ=0.01`,
  `c0` search range, `α≈0.5`) as documented in `config.py`.
