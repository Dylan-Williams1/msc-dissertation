# Short-Window Variant — Changes

Five changes, each reducing required `T_eff`. **`r` is untouched** and remains measured from control data. Nothing here widens the margin or weakens the standard; every gain comes from a better estimator or a better test, not a looser threshold.

Baseline requirement: `T_eff = (k · σ_d / (r · IC_IS))² / N_eff`.

---

## 1. Non-inferiority replaces two-sided equivalence — `k: 2.927 → 2.487`

TOST requires **both** one-sided tests to reject. At a true shift of zero, power `= 2Φ(δ/SE − z₀.₉₅) − 1`, so 80% power needs `z₀.₉₅ + z₀.₉₀ = 2.927`.

Certification asks whether the alpha **degraded**, not whether it changed. An alpha performing *better* out-of-sample is not a contamination finding. Dropping the upper tail leaves one test: `H₀: μ ≤ −δ`, requiring `z₀.₉₅ + z₀.₈₀ = 2.487`.

**Effect:** `(2.487/2.927)² = 0.72` → **28% fewer effective observations.**

⚠️ §7 locks `k = 2.927` and frames Instrument 1 as *equivalence*. This is a methodological change requiring one sentence of justification in the write-up. `TEST_MODE = "tost"` reverts it.

## 2. Multi-factor PC pairing replaces the single control mean

The equal-weighted control mean approximates PC1 but not optimally. The variant regresses each alpha's IC on the first `N_PCS` principal components of the control panel, with **loadings fitted on IS dates only** and held fixed out-of-sample.

Residual σ falls as `√(1 − R²)`, so reduction `= 1 − √(1 − R²)`:

| basis | R² | σ reduction | T_eff multiplier |
|---|---|---|---|
| single factor, ρ = 0.5 | 0.25 | 13% | ×0.75 |
| 3 PCs, R² = 0.50 | 0.50 | 29% | ×0.50 |
| 3 PCs, R² = 0.70 | 0.70 | 45% | ×0.30 |

With thousands of IS days, three regressors carry no overfitting risk. **Largest single gain, and it is pure estimator quality.**

## 3. GLS pooling replaces equal weights

Equal weighting is optimal only for an equicorrelated, equal-variance panel. The minimum-variance combination is `w = Σ⁻¹1 / (1'Σ⁻¹1)`, giving

```
N_eff_GLS = (1' Σ⁻¹ 1) · mean(diag Σ)
```

versus `n / (1 + (n−1)ρ̄)` for equal weights. Σ is estimated IS-only and shrunk toward its diagonal (`GLS_SHRINKAGE`).

**Guard:** `N_eff > n` is capped and flagged. It is possible under genuine negative correlation, but at these panel sizes it is far more often ill-conditioning amplified by the inverse. On the test run this fired — uncapped 96.1 from 20 alphas, capped to 20.

## 4. Pre-registered primary subset for Romano–Wolf

⚠️ **Correction: this does not help reach 80 days. I filed it wrongly.**

RW's critical value is the max over the family, so a *larger* family gives a *larger* adjusted p, which flags *fewer* breaks, which makes certification *easier*. Shrinking the family therefore increases Method A's power to detect breaks — power **against** certification, not toward it. It also leaves the MDE gate and TOST untouched, so it has **zero effect on required T_eff or the frontier**.

Keep it because detecting contamination more reliably is the scientifically correct direction under §1, not because it shortens the window.

The variant restricts FWER control to `PRIMARY_K = 5` alphas ranked by **IS Rank IC** (no out-of-sample information). Secondaries fall back to their unadjusted permutation p — smaller, so more likely to flag a break, which is the conservative side.

**Only legitimate if pre-registered.** Selecting the family after seeing OOS results invalidates the placebo calibration.

## 5. T_eff of the differential reported against the raw IC

Diagnostic, not a lever. Removing the persistent common component *may* raise `T_eff/T` above the assumed ~0.75. The variant measures both and prints whether de-factoring gained anything, so the assumption stops being an assumption.

---

## Combined

Only items 1–3 affect required `T_eff`; items 4 and 5 do not. Multiplicatively: `0.72 × 0.50 ÷ (N_eff gain ~1.2)` → roughly **2.5–3× shorter window**, dominated by items 1 and 2.

**This is not enough on its own if IS Rank IC sits near 0.03.** The frontier scales as `1/IC²`, so raising median IS IC from 0.03 to 0.045 via the DSR threshold (§8) still outweighs everything above. These changes make an 80-day window *reachable* — they do not make it *comfortable*.

## Unchanged

`r` (measured, noise-corrected), δ's definition, the MDE gate's logic, the reconciliation table, placebo construction, leave-one-out control calibration, and all four definite fixes from `instrument1.py`.
