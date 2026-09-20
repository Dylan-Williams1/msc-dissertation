# msc-dissertation

# Evaluating Memorisation in LLM-Generated Quantitative Trading Alphas

MSc dissertation. Dylan Williams (2521133), supervised by Dr Jia Shao.

Three LLMs generate formulaic S&P 500 alphas. Those alphas are screened on
Deflated Sharpe Ratio, then tested for two kinds of memory contamination:
parametric look-ahead bias (Test 1) and event-level memorisation (Test 2).
Two control corpora calibrate the tests — 50 Kakushadze (2016) alphas as a
clean human baseline, and 56 synthetic alphas with a known injected leak.

72 generated alphas → 24 survive screening → 4 survive both tests.
All figures are gross; no transaction costs are modelled.

The generated alphas are committed as artefacts and do not need regenerating.
The screening and testing pipeline runs against them and reproduces every
result in the dissertation.

## What's committed

| Path | Contents |
|---|---|
| `alphas/raw/<tag>/` | Every generated alpha, one JSON per API call — prompt, raw response, SHA-256 hash, parsed code and validation flags |
| `alphas/survived/<tag>/` | The subset that passed Phase 1 screening |
| `alphas/runs/` | Per-run generation manifests |
| `evaluation/initial_screening/results/` | Phase 1 verdicts and the daily Rank IC panel per corpus |
| `evaluation/test1/instrument1_output/` | Test 1 per-alpha votes and verdicts |
| `evaluation/test2/test2_output/<tag>/` | Test 2 event pool, per-event statistics, per-alpha verdicts and diagnostic figures |

Corpus tags are `claude-opus-5`, `gemini-3.6-flash-v5`, `gpt-5.6-sol`,
`kakushadze-101-v1` (human control) and `biased_control` (synthetic leaked
control).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10+. Four data files are needed in `data/` and are not committed
(licensing and size): `daily_ohlcv.parquet`, `meta.json`, `constituents.csv`,
`WSJ_Stock_Jumps.xlsx`.

## Running the pipeline

**1. Build the control corpora**

```bash
python control/build.py alphas/raw      # 50 Kakushadze alphas + manifest
python control/build_biased_control.py  # 56 synthetic leaked alphas
```

**2. Phase 1 screening** — set `TARGET_MODEL` in the script, run once per corpus

```bash
cd evaluation/initial_screening/results
python ../../phase1_evaluation.py
```

Executes each alpha, computes daily Rank IC, keeps those with DSR > 0.95.
Writes `phase1_screening_results_<tag>.csv` and `phase1_daily_rank_ic_<tag>.csv`
to the current folder, and copies survivors to `alphas/survived/<tag>/`.

**3. Test 1** — set `TREATMENT_KEY` and `CONTROL_KEY`

```bash
python evaluation/test1/test1.py
```

Flags alphas whose performance drops after the model's knowledge cutoff by more
than the human controls do. Outputs to `instrument1_output/`.

**4. Test 2** — set `LLM_MODEL_TAG`

```bash
python evaluation/test2/test2.py
```

Flags alphas that perform suspiciously well during famous market shocks.
Outputs to `test2_output/<tag>/`. Pass `--no-plots` to skip the figures.

Test 2 needs `daily_ohlcv.parquet`. Without it the run still completes, but
every event falls back to Scenario A and the results are not interpretable.

## Alpha generation

`generation/alph_gen_code_claude.py`, `_gemini.py` and `_gpt.py` call each
provider's API with the shared v4.0 prompt, 24 alphas per model, writing one
JSON to `alphas/raw/` per call. They are here so the generation process can be
inspected. Re-running them needs `CLAUDE_API_KEY`, `GEMINI_API_KEY` and
`OPENAI_API_KEY` in a `.env`, costs money, and may not reproduce the committed
corpus exactly — temperature is 0.9 and no provider exposes a usable seed. Each alpha
JSON carries the full provenance of its own generation, including the applied
temperature, knowledge cutoff and prompt hashes; failed calls are recorded too,
so attempts always equal artefacts.

## Other files

| Path | Purpose |
|---|---|
| `control/specs.py`, `control/prelude.py` | Per-alpha specs and the operator library `control/build.py` emits |
| `requirements.txt` | Dependencies; the provider SDKs at the bottom are needed only for generation |

Kakushadze, Z. (2016). *101 Formulaic Alphas*. arXiv:1601.00991. This repo
contains an independent implementation, not the published expression text.