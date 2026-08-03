"""
Stage 1 alpha generation — Gemini arm.

Compliance targets:
  - Master doc Section 5   (mandatory machine-readable metadata block)
  - Master doc Section 3b  (generation protocol & provenance disclosures)
  - Master doc Section 6   (technical data spec; close-t / execute-t+1-open)

Design invariant: THIS SCRIPT NEVER DISCARDS A GENERATION.
Every API call produces exactly one artifact on disk — a success record or a
failure record. Validation flags problems in metadata; it never filters. Any
filtering performed later must be an explicit, documented, post-hoc step so
that `selection_applied` remains truthfully False at the generation boundary.

Theme conditioning (user prompt v4.0) narrows the economic mechanism the model
is asked to target. It does NOT discard outputs, so `selection_applied` remains
truthfully False: the prompt is conditioned, nothing is filtered.
"""

import os
import ast
import time
import json
import uuid
import random
import hashlib
import platform
from datetime import datetime, timezone

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise EnvironmentError(
        "GEMINI_API_KEY not set. Put GEMINI_API_KEY=<your_key> in a .env file "
        "next to this script (never hardcode it in source, never paste it 1into "
        "chat, and rotate immediately if it is ever exposed either way)."
    )

# ---------------------------------------------------------------------------
# 1. RUN CONFIGURATION
# ---------------------------------------------------------------------------

MODEL_NAME = "gemini-3.6-flash"
PROVIDER = "Google"
INTERFACE = "API"                 # Section 3b: browser interfaces are prohibited
TEMPERATURE = 0.9
N_SAMPLES_PER_PROMPT = 1          # one draw kept per call; no best-of selection
SELECTION_APPLIED = False         # invariant enforced by the no-discard rule
RETRIEVAL_AUGMENTATION_ENABLED = False
THINKING_LEVEL = "medium"         # recorded: affects reproducibility, not disclosed by default
MASTER_SEED = 20260716            # per-call seeds derived from this, all recorded

# N_ALPHAS is NOT set here. It is defined by CALL_PLAN in section 4b so that
# the planned call count and the theme cell structure cannot drift apart.

# ---------------------------------------------------------------------------
# 2. KNOWLEDGE CUTOFF — THE TREATMENT VARIABLE
# ---------------------------------------------------------------------------
# This is not bookkeeping. Instrument 1 tests for a structural break AT this
# date, so an incorrect value invalidates the entire Memory Contamination test.
#
# Take it from the official model card ONLY. Third-party aggregators disagree
# with Google's own documentation for this model. Model cards are revised, so
# the retrieval date is recorded alongside the value.
#
# If the model card states a RANGE or a dual cutoff (e.g. "March 2026, though
# in some domains limited to January 2025"), the break date is an interval, not
# a point. Record both bounds and treat the ambiguity as a stated limitation of
# the break-location test — do not silently pick one.

MODEL_KNOWLEDGE_CUTOFF = "2026-03"           # VERIFY against the model card before running
MODEL_KNOWLEDGE_CUTOFF_LOWER = "2025-01"     # earliest plausible cutoff
MODEL_KNOWLEDGE_CUTOFF_UPPER = "2026-03"     # latest plausible cutoff; widen if dual-stated
MODEL_KNOWLEDGE_CUTOFF_SOURCE = (
    "https://deepmind.google/models/model-cards/gemini-3-6-flash/"
)
MODEL_KNOWLEDGE_CUTOFF_SOURCE_TYPE = "official_model_card"
MODEL_KNOWLEDGE_CUTOFF_RETRIEVED_UTC = "2026-07-30"

if MODEL_KNOWLEDGE_CUTOFF is None:
    raise ValueError(
        "model_knowledge_cutoff must be set from the official model card. "
        "It is the treatment variable for Instrument 1."
    )

# ---------------------------------------------------------------------------
# 3. PATHS
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(REPO_ROOT, "alphas", "raw")
RUN_DIR = os.path.join(REPO_ROOT, "alphas", "runs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RUN_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 4. PROMPTS
# ---------------------------------------------------------------------------
# Changes from v1.0, each tied to a spec requirement:
#   - Section 1 must declare parameters with default values (Section 5 schema).
#   - Section 4 execution rule fixed to t+1 OPEN. The previous "open or close"
#     wording let the model choose, which breaks the Zhang et al. one-switch
#     clean reference protocol.
#   - Section 5 signature must expose parameters as keyword arguments.
#   - Explicit prohibition on undeclared imputation, so that Code-Documentation
#     Fidelity failures are genuine drift rather than an artefact of the prompt
#     never asking for the behaviour to be stated.
#
# SYSTEM PROMPT IS HELD AT v3.0 AND IS BYTE-IDENTICAL TO THE v3.0 ARM.
# Theme conditioning is the single toggle relative to that arm, so the system
# prompt version must NOT be bumped: its sha256 is unchanged and the metadata
# must say so. Only the user prompt advances to v4.0.

SYSTEM_PROMPT_VERSION = "v4.0"
USER_PROMPT_VERSION = "v4.0"

SYSTEM_INSTRUCTION = """
You are a quantitative researcher and algorithmic trading system developer specialising in US Equities. Your objective is to design one candidate quantitative trading alpha utilizing exclusively daily OHLCV market data.

The alpha must use only the following daily market variables:

open
high
low
close
volume
returns

You must output your response adhering strictly to the 5-section schema below. Do not include conversational filler, introductions, or conclusions.

Prioritise internal consistency, economic plausibility, reproducibility, and precise implementation over apparent predictive performance. The alpha should be fully specified and reproducible without requiring additional assumptions.

# Section 1: Alpha Specification
Provide a structured summary containing:
- Formula Name: (Descriptive name)
- Holding Period: (Expected holding time in days)
- Orientation: (Long/Short or Long-only)
- Parameters: (List every lookback window, span, or threshold as a NAMED parameter with an explicit default value, e.g. `N_vol = 20`. Every numeric constant in Section 2 other than trivial arithmetic literals must appear here as a named parameter.)
- Required Inputs: (Must be restricted to OHLCV and Returns)

# Section 2: Mathematical Formulation
Provide the exact, step-by-step mathematical formulation of the alpha signal. You must use LaTeX for all mathematical notation. Clearly define all variables. Refer to parameters by the names declared in Section 1.
State explicitly any handling of undefined values that is part of the formulation: minimum observation counts for rolling windows, treatment of division by zero, and treatment of missing observations. If the formulation performs no such handling, say so.

# Section 3: Causal Rationale
Provide a concise 3-5 sentence explanation linking the mathematical signal to a plausible economic or behavioural mechanism.
Ground the mechanism in constructs that are observable in daily OHLCV data. Do not appeal to unobservable constructs (institutional versus retail order flow, investor conviction, credit quality, positioning, sentiment) unless you state the specific OHLCV proxy that stands in for them.
Do not justify the alpha using historical performance, backtest results, or any claim about how the signal has performed in any period.

# Section 4: Structural Bounds
State explicit constraints regarding the execution of this strategy. You must confirm that the data relies solely on Point-in-Time (PIT) OHLCV data.
The execution timing rule is FIXED and not yours to choose: the signal is generated using data available up to the market close on day t, and is executed at the OPEN of day t+1. State this rule explicitly and confirm the formulation complies with it.
State the minimum number of trailing observations required before the signal is defined for a given symbol.

# Section 5: Python Implementation
Provide a fully vectorized Python code block using `pandas` that can be executed directly without defining custom helper functions or classes.
Technical constraints:
- Wrap your logic inside a single function named `generate_alpha`.
- Every parameter declared in Section 1 must be exposed as a keyword argument with the default value stated in Section 1. For example: `def generate_alpha(df, N_vol=20, N_smooth=5):`. Do not hardcode these constants inside the function body.
- The input is a single `pandas` DataFrame named `df` indexed by a MultiIndex of `(date, symbol)`.
- Available columns are strictly: `open`, `high`, `low`, `close`, `volume`, `returns`.
- Use `df.groupby(level='symbol')` for time-series rolling operations.
- Use `df.groupby(level='date')` for cross-sectional ranking or z-scoring.
- The function must return exactly one `pandas.Series` indexed identically to `df`.
- The returned Series must be aligned to date t (the signal as known at the close of day t). Do NOT apply the execution lag inside the function; the backtesting harness applies it. Do not use `.shift(-1)` or any negative shift.
- No future look-ahead bias is permitted; operations on row t must only use data up to and including t.
- Any imputation, clipping, infinity handling, or minimum-period setting present in the code must correspond to something stated in Section 2. Do not add silent `fillna`, `replace`, or `clip` steps that are absent from the mathematical formulation.

Output only a single executable Python code block.
No explanatory text should appear within Section 5. The final expression must evaluate to a single pandas.Series aligned to df. Do not include backtesting code, portfolio construction, transaction cost modelling, plotting, or performance evaluation.
"""

# ---------------------------------------------------------------------------
# 4b. THEME CONDITIONING (user prompt v4.0)
# ---------------------------------------------------------------------------
# PRE-REGISTERED. Frozen before generation; no theme added, removed, or
# reworded after any output was seen. THEME_SPEC_SHA256 pins this, so the
# claim is checkable rather than merely asserted.
#
# The eight themes are the price- and volume-derived anomalies of the standard
# cross-sectional asset pricing literature that are computable from OHLCV
# alone. Fundamentals-based factors (value, size, quality, investment,
# profitability) are absent because the data tier excludes them, not by choice.
#
# The prompt's ENTIRE theme contribution is the theme title. No description, no
# construction guidance, no operations named. The construction remains the
# model's to choose — otherwise the corpus measures the researcher's design
# choices and the Stage 3 battery tests the wrong author.
#
# Citations are recorded here for the write-up and are NOT sent to the model:
#   momentum                      Jegadeesh & Titman (1993)
#   short-term reversal           Jegadeesh (1990); Lehmann (1990)
#   long-term reversal            De Bondt & Thaler (1985)
#   volatility                    Ang, Hodrick, Xing & Zhang (2006)
#   liquidity                     Amihud (2002); Datar, Naik & Radcliffe (1998)
#   trading volume                Gervais, Kaniel & Mingelgrin (2001)
#   nearness to the 52-week high  George & Hwang (2004)
#   beta                          Frazzini & Pedersen (2014)
#
# DISCLOSURE (governs Stage 4): naming canonical anomalies points generation at
# the Benchmark Leakage comparison corpus, and bare canonical labels are
# stronger recall triggers than descriptive phrasing would be. The themed arm's
# leakage rate is therefore CONDITIONAL and is not an unbiased estimate of
# unsteered model behaviour. The v2.0/v3.0 unconditioned arms supply that
# baseline. Report both, labelled.

THEME_SPEC = [
    "momentum",
    "short-term reversal",
    "long-term reversal",
    "volatility",
    "liquidity",
    "trading volume",
    "nearness to the 52-week high",
    "beta",
]
N_PER_THEME = 3


def theme_slug(theme: str) -> str:
    """Machine-readable id for metadata grouping and cell counting."""
    return theme.lower().replace("-", "_").replace(" ", "_")


def build_user_prompt(theme: str) -> str:
    return (
        "Generate one candidate quantitative trading alpha. "
        "Follow all requirements specified in the system prompt.\n\n"
        f"The alpha's economic mechanism must centre on {theme}. "
        "Within that constraint, the specific construction is entirely yours "
        "to choose."
    )


# Explicit call plan: inspectable before any API spend, and the auditable
# denominator for the attempts == artifacts invariant.
CALL_PLAN = [
    (theme_slug(theme), theme, build_user_prompt(theme))
    for theme in THEME_SPEC
    for _ in range(N_PER_THEME)
]
N_ALPHAS = len(CALL_PLAN)   # 8 themes x 3 samples = 24


def sha256(text: str) -> str:
    """Content hash so a version string is verifiable, not merely asserted."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


SYSTEM_PROMPT_SHA256 = sha256(SYSTEM_INSTRUCTION)
USER_PROMPT_SHA256_BY_THEME = {
    theme_slug(theme): sha256(build_user_prompt(theme))
    for theme in THEME_SPEC
}
THEME_SPEC_SHA256 = sha256(json.dumps(THEME_SPEC, sort_keys=True))

# ---------------------------------------------------------------------------
# 5. STATIC VALIDATION (FLAGS ONLY — NEVER FILTERS)
# ---------------------------------------------------------------------------

REQUIRED_SECTIONS = [
    "Section 1", "Section 2", "Section 3", "Section 4", "Section 5",
]

# Tokens implying data outside the OHLCV tier. `cap` is checked as a whole word
# to avoid matching `capital`, `capture`, etc.
FORBIDDEN_DATA_TOKENS = [
    "vwap", "market_cap", "marketcap", "indclass", "industry",
    "sector", "adv20_cap", "shares_outstanding", "float",
]

# Not forbidden — flagged for the Code-Documentation Fidelity gate to inspect.
FIDELITY_WATCH_TOKENS = [
    "fillna", "ffill", "bfill", "pad(", "interpolate",
    "clip", "replace", "dropna", "np.inf", "inf)",
]


def extract_code_block(raw: str):
    """Pull the last fenced python block out of the response."""
    fences = [i for i in range(len(raw)) if raw.startswith("```", i)]
    if len(fences) < 2:
        return None
    start, end = fences[-2], fences[-1]
    block = raw[start:end]
    newline = block.find("\n")
    return block[newline + 1:] if newline != -1 else None


def validate(raw: str) -> dict:
    """Return a flag dictionary. Nothing here rejects an alpha."""
    flags = {
        "missing_sections": [s for s in REQUIRED_SECTIONS if s not in raw],
        "code_block_found": False,
        "code_parses": False,
        "generate_alpha_defined": False,
        "keyword_args": [],
        "has_keyword_args": False,
        "negative_shift_present": False,
        "forbidden_data_tokens": [],
        "fidelity_watch_tokens": [],
        "syntax_error": None,
    }

    code = extract_code_block(raw)
    if code is None:
        return flags
    flags["code_block_found"] = True

    lowered = code.lower()
    flags["forbidden_data_tokens"] = [
        t for t in FORBIDDEN_DATA_TOKENS if t in lowered
    ]
    # whole-word check for bare `cap`
    if any(w.strip("(),.[]= ") == "cap" for w in lowered.split()):
        flags["forbidden_data_tokens"].append("cap")
    flags["fidelity_watch_tokens"] = [
        t for t in FIDELITY_WATCH_TOKENS if t in lowered
    ]

    try:
        tree = ast.parse(code)
        flags["code_parses"] = True
    except SyntaxError as exc:
        flags["syntax_error"] = str(exc)
        return flags

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "generate_alpha":
            flags["generate_alpha_defined"] = True
            n_defaults = len(node.args.defaults)
            named = [a.arg for a in node.args.args][-n_defaults:] if n_defaults else []
            named += [a.arg for a in node.args.kwonlyargs]
            flags["keyword_args"] = named
            flags["has_keyword_args"] = len(named) > 0
        # detect any negative shift
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "attr", None) or getattr(func, "id", None)
            if name == "shift":
                for arg in list(node.args) + [k.value for k in node.keywords]:
                    if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                        flags["negative_shift_present"] = True
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                        if arg.value < 0:
                            flags["negative_shift_present"] = True

    return flags


# ---------------------------------------------------------------------------
# 6. CLIENT
# ---------------------------------------------------------------------------

client = genai.Client(api_key=api_key)


def build_config(seed: int):
    """
    Build the generation config, recording what was actually applied.

    tools=[] is passed EXPLICITLY. Section 3b requires positive confirmation
    that no retrieval augmentation ran, and omitting the argument is not
    confirmation. The response is separately checked for grounding metadata.
    """
    applied = {
        "temperature": TEMPERATURE,
        "seed": seed,
        "tools": [],
        "thinking_level": None,
    }
    kwargs = {
        "system_instruction": SYSTEM_INSTRUCTION,
        "temperature": TEMPERATURE,
        "seed": seed,
        "tools": [],
    }
    try:
        kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=THINKING_LEVEL
        )
        applied["thinking_level"] = THINKING_LEVEL
    except (AttributeError, TypeError):
        # Older SDK: thinking level not settable. Recorded as None so the
        # write-up does not claim a setting that was never applied.
        pass

    return types.GenerateContentConfig(**kwargs), applied


def grounding_evidence(response) -> dict:
    """
    Positive check that no retrieval/tool use occurred.

    If this returns anything non-empty, the alpha's effective cutoff is the
    GENERATION DATE, not the model cutoff, and every look-ahead test applied
    to it is void. Such an alpha must be quarantined, not silently kept.
    """
    evidence = {"grounding_metadata_present": False, "detail": None}
    try:
        for cand in (response.candidates or []):
            gm = getattr(cand, "grounding_metadata", None)
            if gm:
                evidence["grounding_metadata_present"] = True
                evidence["detail"] = str(gm)[:2000]
    except Exception as exc:
        evidence["detail"] = f"grounding check failed: {exc}"
    return evidence


# ---------------------------------------------------------------------------
# 7. GENERATION LOOP
# ---------------------------------------------------------------------------

run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
run_started = datetime.now(timezone.utc).isoformat()
rng = random.Random(MASTER_SEED)
run_records = []

print(f"Run {run_id}: generating {N_ALPHAS} alphas from {MODEL_NAME}")
print(f"Plan: {len(THEME_SPEC)} themes x {N_PER_THEME} samples")
print(f"Recorded knowledge cutoff: {MODEL_KNOWLEDGE_CUTOFF} "
      f"({MODEL_KNOWLEDGE_CUTOFF_SOURCE_TYPE})")

for i, (theme_id, theme_title, user_prompt) in enumerate(CALL_PLAN):
    alpha_id = f"alpha_{uuid.uuid4().hex[:8]}"
    call_seed = rng.randint(0, 2**31 - 1)
    print(f"[{i+1}/{N_ALPHAS}] {alpha_id} theme={theme_id} (seed={call_seed})")

    metadata = {
        # --- Section 5 mandatory block ---
        "alpha_id": alpha_id,
        "model": MODEL_NAME,
        "provider": PROVIDER,
        "model_knowledge_cutoff": MODEL_KNOWLEDGE_CUTOFF,
        "retrieval_augmentation_enabled": RETRIEVAL_AUGMENTATION_ENABLED,
        "interface": INTERFACE,
        "temperature": TEMPERATURE,
        "seed": call_seed,
        "selection_applied": SELECTION_APPLIED,
        "system_prompt_version": SYSTEM_PROMPT_VERSION,
        "user_prompt_version": USER_PROMPT_VERSION,
        "parameters": None,  # populated from validation below
        "theme": theme_id,
        "theme_title": theme_title,
        "timestamp_utc": None,

        # --- Section 3b supporting provenance ---
        "run_id": run_id,
        "call_index": i,
        "n_samples_per_prompt": N_SAMPLES_PER_PROMPT,
        "thinking_level_requested": THINKING_LEVEL,
        "thinking_level_applied": None,
        "master_seed": MASTER_SEED,
        "system_prompt_sha256": SYSTEM_PROMPT_SHA256,
        "user_prompt_sha256": USER_PROMPT_SHA256_BY_THEME[theme_id],
        "theme_spec_sha256": THEME_SPEC_SHA256,
        "model_knowledge_cutoff_lower": MODEL_KNOWLEDGE_CUTOFF_LOWER,
        "model_knowledge_cutoff_upper": MODEL_KNOWLEDGE_CUTOFF_UPPER,
        "model_knowledge_cutoff_source": MODEL_KNOWLEDGE_CUTOFF_SOURCE,
        "model_knowledge_cutoff_source_type": MODEL_KNOWLEDGE_CUTOFF_SOURCE_TYPE,
        "model_knowledge_cutoff_retrieved_utc": MODEL_KNOWLEDGE_CUTOFF_RETRIEVED_UTC,
        "sdk_version": getattr(genai, "__version__", "unknown"),
        "python_version": platform.python_version(),
        "generation_status": None,
    }

    config, applied = build_config(call_seed)
    metadata["thinking_level_applied"] = applied["thinking_level"]

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_prompt,
            config=config,
        )
        raw_text = response.text
        metadata["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        metadata["generation_status"] = "success"

        grounding = grounding_evidence(response)
        if grounding["grounding_metadata_present"]:
            # Effective cutoff is now the generation date. Quarantine, do not keep.
            metadata["generation_status"] = "quarantined_retrieval_detected"
            metadata["retrieval_augmentation_enabled"] = True
            print("   !! GROUNDING METADATA PRESENT — QUARANTINED")

        flags = validate(raw_text)
        metadata["parameters"] = flags["keyword_args"] or None

        try:
            usage = response.usage_metadata
            usage_record = {
                "prompt_tokens": getattr(usage, "prompt_token_count", None),
                "output_tokens": getattr(usage, "candidates_token_count", None),
                "total_tokens": getattr(usage, "total_token_count", None),
            }
        except Exception:
            usage_record = None

        try:
            finish_reason = str(response.candidates[0].finish_reason)
        except Exception:
            finish_reason = None

        payload = {
            "metadata": metadata,
            "raw_response": raw_text,
            "response_sha256": sha256(raw_text),
            "validation_flags": flags,
            "api_config_applied": applied,
            "api_response_audit": {
                "finish_reason": finish_reason,
                "usage": usage_record,
                "grounding": grounding,
            },
        }

        problems = []
        if flags["missing_sections"]:
            problems.append(f"missing {flags['missing_sections']}")
        if not flags["code_parses"]:
            problems.append("code does not parse")
        if not flags["has_keyword_args"]:
            problems.append("no keyword args")
        if flags["negative_shift_present"]:
            problems.append("NEGATIVE SHIFT")
        if flags["forbidden_data_tokens"]:
            problems.append(f"off-tier data {flags['forbidden_data_tokens']}")
        print("   flags: " + ("; ".join(problems) if problems else "clean"))

    except Exception as exc:
        # A failed call is still a data point about the corpus. Recording it is
        # what keeps selection_applied=False honest.
        metadata["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        metadata["generation_status"] = "api_error"
        payload = {
            "metadata": metadata,
            "raw_response": None,
            "response_sha256": None,
            "validation_flags": None,
            "api_config_applied": applied,
            "api_response_audit": {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        }
        print(f"   ERROR: {type(exc).__name__}: {exc}")

    file_path = os.path.join(OUTPUT_DIR, f"{alpha_id}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)

    run_records.append({
        "alpha_id": alpha_id,
        "theme": theme_id,
        "status": metadata["generation_status"],
        "seed": call_seed,
        "response_sha256": payload["response_sha256"],
        "file": os.path.relpath(file_path, REPO_ROOT),
    })

    if i < N_ALPHAS - 1:
        time.sleep(35)  # free-tier rate limit

# ---------------------------------------------------------------------------
# 8. RUN MANIFEST
# ---------------------------------------------------------------------------
# The manifest is the auditable record that attempts == artifacts. It is what
# you cite in the write-up when asserting that no selection occurred between
# generation and storage.

hashes = [r["response_sha256"] for r in run_records if r["response_sha256"]]
duplicate_hashes = len(hashes) - len(set(hashes))

theme_counts = {
    theme_slug(t): sum(r["theme"] == theme_slug(t) for r in run_records)
    for t in THEME_SPEC
}

manifest = {
    "run_id": run_id,
    "started_utc": run_started,
    "finished_utc": datetime.now(timezone.utc).isoformat(),
    "model": MODEL_NAME,
    "model_knowledge_cutoff": MODEL_KNOWLEDGE_CUTOFF,
    "model_knowledge_cutoff_source": MODEL_KNOWLEDGE_CUTOFF_SOURCE,
    "interface": INTERFACE,
    "retrieval_augmentation_enabled": RETRIEVAL_AUGMENTATION_ENABLED,
    "temperature": TEMPERATURE,
    "master_seed": MASTER_SEED,
    "system_prompt_version": SYSTEM_PROMPT_VERSION,
    "system_prompt_sha256": SYSTEM_PROMPT_SHA256,
    "user_prompt_version": USER_PROMPT_VERSION,
    "user_prompt_sha256_by_theme": USER_PROMPT_SHA256_BY_THEME,
    "theme_spec": THEME_SPEC,
    "theme_spec_sha256": THEME_SPEC_SHA256,
    "n_themes": len(THEME_SPEC),
    "n_per_theme": N_PER_THEME,
    "theme_counts": theme_counts,
    "n_attempted": N_ALPHAS,
    "n_artifacts_written": len(run_records),
    "n_success": sum(r["status"] == "success" for r in run_records),
    "n_api_error": sum(r["status"] == "api_error" for r in run_records),
    "n_quarantined": sum(
        r["status"] == "quarantined_retrieval_detected" for r in run_records
    ),
    "n_exact_duplicate_responses": duplicate_hashes,
    "selection_applied": SELECTION_APPLIED,
    "records": run_records,
}

manifest_path = os.path.join(RUN_DIR, f"{run_id}_manifest.json")
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=4)

prompt_path = os.path.join(RUN_DIR, f"{run_id}_prompts.txt")
with open(prompt_path, "w", encoding="utf-8") as f:
    f.write(f"SYSTEM ({SYSTEM_PROMPT_VERSION}, sha256={SYSTEM_PROMPT_SHA256})\n")
    f.write(SYSTEM_INSTRUCTION)
    f.write(f"\n\nTHEME SPEC (sha256={THEME_SPEC_SHA256}, "
            f"{len(THEME_SPEC)} themes x {N_PER_THEME} samples)\n")
    for theme in THEME_SPEC:
        slug = theme_slug(theme)
        f.write(f"\n--- USER ({USER_PROMPT_VERSION}, theme={slug}, "
                f"sha256={USER_PROMPT_SHA256_BY_THEME[slug]})\n")
        f.write(build_user_prompt(theme) + "\n")

print(f"\nAttempted {N_ALPHAS}, wrote {len(run_records)} artifacts.")
print(f"success={manifest['n_success']} "
      f"api_error={manifest['n_api_error']} "
      f"quarantined={manifest['n_quarantined']} "
      f"exact_duplicates={duplicate_hashes}")
print(f"theme_counts={theme_counts}")
print(f"Manifest: {manifest_path}")

assert len(run_records) == N_ALPHAS, (
    "Artifact count != attempt count. Undisclosed attrition — do not use this run."
)

# Total count alone is not sufficient: a dropped call topped up from a second
# run keeps the total right while silently unbalancing the design. Check cells.
assert all(theme_counts[theme_slug(t)] == N_PER_THEME for t in THEME_SPEC), (
    f"Theme cells unbalanced: {theme_counts}. "
    "Do not top up — rerun the whole plan."
)
