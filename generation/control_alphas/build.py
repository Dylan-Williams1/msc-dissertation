"""Build the non-LLM control corpus (Kakushadze 2016, 101 Formulaic Alphas).

Emits one JSON per control alpha, in the same envelope as the LLM-generated
alphas, so that phase1_evaluation.py runs over it unmodified.

Run:  python build.py <output_dir>
"""
import ast
import csv
import datetime as dt
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prelude import PRELUDE, TEMPLATE          # noqa: E402
from specs import SPECS, EXCLUSIONS            # noqa: E402

CORPUS_VERSION = "kakushadze-101-v1"
MODEL_KEY = "kakushadze_2016_control"
BUILT_UTC = dt.datetime(2026, 8, 4, tzinfo=dt.timezone.utc).isoformat()

SOURCE = {
    "citation": ("Kakushadze, Z. (2016). 101 Formulaic Alphas. "
                 "Wilmott Magazine 2016(84), 72-80. arXiv:1601.00991"),
    "appendix": "Appendix A.1 (expressions), A.2 (operators), A.3 (input data)",
    "rights_note": ("Formulae in Kakushadze Appendix A are reproduced there by "
                    "permission of WorldQuant LLC, which retains copyright. This "
                    "corpus contains an independent Python implementation and a "
                    "restated step decomposition, not the published expression text."),
}

RATIONALE_NA = (
    "Not applicable. The source publication provides no per-alpha causal rationale: "
    "the 101 are presented as production formulae with only a corpus-level "
    "characterisation (mean-reversion vs. momentum building blocks, Section 2 of the "
    "source). No rationale is authored here, because writing one would fabricate the "
    "object that Narrative Groundedness is meant to measure. The control arm is "
    "therefore excluded from Narrative Groundedness by construction, exactly as it is "
    "excluded from Benchmark Leakage. Consequence to disclose: those two criteria have "
    "no control differential and their LLM failure rates are reported as levels, not "
    "differentials."
)


def uses_adv(spec):
    return "adv(" in spec["body"]


def build_signature(spec):
    parts = [f"{k}={v!r}" for k, v in spec["params"]]
    if uses_adv(spec):
        parts.append("adv_dollar=False")
    parts += ["neutralize=True", "min_names=20"]
    return ", ".join(parts)


def build_code(spec):
    if not uses_adv(spec):
        # keep the emitted code free of an unused free variable
        return TEMPLATE.replace("close * volume if adv_dollar else volume",
                                "volume").format(
            signature=build_signature(spec),
            prelude=PRELUDE.rstrip("\n"),
            n=spec["n"],
            body=spec["body"],
        )
    return TEMPLATE.format(
        signature=build_signature(spec),
        prelude=PRELUDE.rstrip("\n"),
        n=spec["n"],
        body=spec["body"],
    )


def build_document(spec, code):
    n = spec["n"]
    _p = [f"  - `{k} = {v}`" for k, v in spec["params"]]
    if uses_adv(spec):
        _p.append("  - `adv_dollar = False` (unit convention for `adv{d}`; see Section 4)")
    param_lines = "\n".join(_p) or "  - (none)"
    step_lines = "\n".join(
        f"{i}. {s}" for i, s in enumerate(spec["steps"], 1))
    return f"""# Section 1: Alpha Specification

- Formula Name: Kakushadze Alpha#{n} (control)
- Source: {SOURCE['citation']}, {SOURCE['appendix']}
- Holding Period: not specified per alpha in the source; the corpus average holding \
period is reported as roughly 0.6-6.4 days
- Orientation: Long/Short (dollar-neutral; see Section 4)
- Parameters:
{param_lines}
- Required Inputs: {', '.join('`%s`' % c for c in spec['inputs'])}

---

# Section 2: Mathematical Formulation

Reference specification, restated as an ordered step decomposition. This is the \
object against which the Section 5 code is audited under the Code-Documentation \
Fidelity gate. The canonical single-line expression is Alpha#{n} of {SOURCE['appendix']} \
in the source publication; it is cited rather than reproduced here ({SOURCE['rights_note']}).

{step_lines}

Operator semantics follow Appendix A.2 of the source. Implementation conventions that \
the source leaves unspecified are fixed corpus-wide and listed in \
CONTROL_GROUP_METHODOLOGY.md Section 3; they are applied identically to every control \
alpha.

---

# Section 3: Causal Rationale

{RATIONALE_NA}

---

# Section 4: Structural Bounds

- **Point-in-Time Confirmation:** Uses daily OHLCV known at or before the close of day \
$t$. No forward-looking data and no negative index shifts.
- **Execution Timing Rule:** Signal formed at the close of day $t$; rebalancing executes \
at the open of day $t+1$. Alpha#{n} is a delay-1 alpha in the source's terminology, so it \
is compatible with this convention without modification. The four delay-0 alphas in the \
source (#42, #48, #53, #54) are excluded from this corpus for that reason.
- **Data Tier:** Strictly OHLCV. Where `adv{{d}}` appears, it defaults to the d-day \
rolling mean of share `volume`. Appendix A.3 of the source defines it on dollar volume, \
but the alphas that compare or divide `volume` by `adv{{d}}` (#7, #17, #21, #39, #43) are \
dimensionally inconsistent and degenerate under that reading - Alpha#7 collapses to the \
constant -1 across the whole panel. The literal dollar reading is retained behind \
`adv_dollar=True` so the choice is auditable and a sensitivity run is one flag away. \
True dollar volume would require vwap, which is not in this panel.
- **Dollar Neutrality:** A cross-sectional de-meaning and unit-gross rescaling is applied \
as the final step (`neutralize=True`). The source's alphas are traded in a dollar-neutral \
portfolio and its reported Sharpe figures assume that construction; several raw \
expressions (any terminating in `rank()`) are strictly positive and would otherwise be a \
net-long book. The LLM arm z-scores cross-sectionally inside its own code, so both arms \
deliver a de-meaned cross-sectional signal. The step is declared here so the Fidelity \
gate sees it, and is toggleable via the keyword argument.
- **Minimum Cross-Section:** dates with fewer than `min_names` valid observations are set \
to NaN, matching `IC_MIN_NAMES` in the screening harness.

---

# Section 5: Python Implementation

```python
{code}```
"""


def main(out_root):
    corpus_dir = os.path.join(out_root, CORPUS_VERSION)
    os.makedirs(corpus_dir, exist_ok=True)

    manifest = []
    for spec in SPECS:
        n = spec["n"]
        code = build_code(spec)
        ast.parse(code)  # fail loudly at build time, not in the harness
        doc = build_document(spec, code)
        alpha_id = f"alpha_k{n:03d}"

        record = {
            "metadata": {
                "alpha_id": alpha_id,
                "model": MODEL_KEY,
                "provider": "Kakushadze (2016) / WorldQuant LLC",
                "model_knowledge_cutoff": None,
                "retrieval_augmentation_enabled": False,
                "interface": "not_applicable",
                "temperature": None,
                "seed": None,
                "selection_applied": True,
                "selection_note": (
                    "Two selection layers, both inherited from the source and neither "
                    "under this project's control. (a) The source states the 101 were "
                    "picked from a much larger production pool on simplicity grounds, "
                    "and that 80 were in production at publication; the effective "
                    "multiple-testing burden behind them is therefore far larger than "
                    "101. (b) This corpus retains 50 of the 101 on data availability and "
                    "execution convention only - never on performance. See "
                    "control_group_manifest.csv for the per-alpha decision."),
                "system_prompt_version": None,
                "user_prompt_version": None,
                "parameters": ([k for k, _ in spec["params"]]
                               + (["adv_dollar"] if uses_adv(spec) else [])),
                "theme": "control_kakushadze",
                "theme_title": "non-LLM control (formulaic, OHLCV, cross-sectional)",
                "timestamp_utc": BUILT_UTC,
                "run_id": CORPUS_VERSION,
                "arm": "control",
                "kakushadze_number": n,
                "required_inputs": spec["inputs"],
                "source_citation": SOURCE["citation"],
                "publication_year": 2016,
                "generation_status": "success",
            },
            "raw_response": doc,
            "validation_flags": {
                "missing_sections": [],
                "code_block_found": True,
                "code_parses": True,
                "generate_alpha_defined": True,
                "keyword_args": ([k for k, _ in spec["params"]]
                                 + (["adv_dollar"] if uses_adv(spec) else [])),
                "has_keyword_args": bool(spec["params"]),
                "negative_shift_present": False,
                "forbidden_data_tokens": [],
                "syntax_error": None,
            },
        }
        record["response_sha256"] = hashlib.sha256(
            doc.encode("utf-8")).hexdigest()

        path = os.path.join(corpus_dir, f"{alpha_id}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=4)

        manifest.append({
            "kakushadze_number": n,
            "alpha_id": alpha_id,
            "included": "yes",
            "exclusion_basis": "",
            "reason": "",
            "required_inputs": " ".join(spec["inputs"]),
            "n_parameters": len(spec["params"]),
        })

    for n, (basis, reason) in sorted(EXCLUSIONS.items()):
        manifest.append({
            "kakushadze_number": n,
            "alpha_id": "",
            "included": "no",
            "exclusion_basis": basis,
            "reason": reason,
            "required_inputs": "",
            "n_parameters": "",
        })

    manifest.sort(key=lambda r: r["kakushadze_number"])
    mpath = os.path.join(out_root, "control_group_manifest.csv")
    with open(mpath, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)

    assert len(manifest) == 101, f"manifest covers {len(manifest)} of 101"
    print(f"wrote {len(SPECS)} control alphas to {corpus_dir}")
    print(f"wrote manifest ({len(manifest)} rows) to {mpath}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
