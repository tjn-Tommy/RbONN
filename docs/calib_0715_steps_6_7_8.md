# Calibration campaign 2026-07-15 — Steps 6, 7, 8

One-day chain on branch `0713updates`, building on the Step-3b calibration from 07-14
(`src/calib_data/calib_step3b_0714_1534.json`). Each step's result JSON embeds its inputs, so every
stage is driven by **one** file:

```
step 3b (07-14)                 channel layout (wavelength <-> SLM column, intensity LUT)
   └─> step 6  (0715_1714)      per-pair TPA efficiency eta + single-beam/dark background
          └─> step 7  (0715_1756)   comb-phase spectrum {Phi_k} vs reference pair
                 └─> step 8  (0715_1811)   random-input check of the composed forward model
```

| Step | Script | Input | Outputs (in `src/calib_data/`) |
|---|---|---|---|
| 6 | `src/drafts/calib_step6_test.py` | `calib_step3b_0714_1534.json` | `calib_step6_meas_0715_1714.csv`, `calib_step6_result_0715_1714.json` (embeds step 3), `calib_step6_pair{1,3,4,5}_0715_1714.png` |
| 7 | `src/drafts/calib_step7_test.py` | `calib_step6_result_0715_1714.json` | `calib_step7_meas_0715_1722.csv`, `calib_step7_result_0715_1756.json` (embeds steps 3+6, both fit methods) |
| 8 | `src/drafts/calib_step8_test.py` | `calib_step7_result_0715_1756.json` | `calib_step8_meas_0715_1811.csv`, `calib_step8_meas_0715_1811_compare_bounded.png` |

Common acquisition settings (all three steps, `draft_hw.py` + `daq_module`): NI-DAQ `Dev1/ai0`,
1 kS/s, ±0.1 V differential, 20 Hz low-pass; per point **5 s** when at most one beam is on
(single-beam lines, all-off dark) and **3 s** when both beams are on; 0.25 s settle after each SLM
pattern change. Every CSV row records the mean, its SEM, and `sem_ratio = sem/|mean|`.

---

## Step 6 — TPA pair efficiency η (per pair)

**What it measures.** For each channel pair *k*, the two-photon signal produced when its two
channels (x-side and w-side) are driven at commanded intensities `x, w ∈ [0,1]`, plus the
single-beam leakage and dark offset that everything downstream must subtract.

**Model** (weighted least squares in `slm_module.tpa_pair`, errors Birge-scaled):

$$Y(x, w) = \eta^2\,(x\,w) \;+\; a_x x + q_x x^2 \;+\; a_w w + q_w w^2 \;+\; d$$

The fit parameter is `b = η²`; η = √b is reported. The `q` terms are nuisance saturation terms
(all came out consistent with zero).

**Sweep design.** Pairs `[1, 3, 4, 5]`, and instead of a full 2-D grid, three reduced 1-D curves of
10 points each (ramp 0.1 → 1.0) plus one shared dark:

- **x-only** `(x=r, w=0)` — pins `a_x, q_x`
- **w-only** `(x=0, w=r)` — pins `a_w, q_w`
- **cross** `(x=1, w=r)` — pins η
- **dark** `(0, 0)` — anchors `d`

31 points, 6 parameters → dof = 25 per pair. All other channels held off.

**Results** (`calib_step6_result_0715_1714.json`, center λ = 778.038 nm):

| pair | λ_x (nm) | λ_w (nm) | η | a_x (mV/unit) | a_w (mV/unit) | d (mV) | χ²/dof | R² |
|---|---|---|---|---|---|---|---|---|
| 1 | 778.208 | 777.870 | 0.05001 ± 0.00123 | 0.88 ± 0.28 | 1.15 ± 0.24 | 1.938 ± 0.053 | 2.42 | 0.9951 |
| 3 | 778.435 | 777.641 | 0.04877 ± 0.00161 | 0.96 ± 0.28 | 1.41 ± 0.25 | 1.970 ± 0.053 | 2.78 | 0.9938 |
| 4 | 778.542 | 777.529 | 0.04766 ± 0.00112 | 0.75 ± 0.23 | 1.33 ± 0.19 | 2.075 ± 0.041 | 1.78 | 0.9920 |
| 5 | 778.654 | 777.419 | 0.04547 ± 0.00166 | 0.39 ± 0.28 | 1.51 ± 0.25 | 2.074 ± 0.049 | 3.01 | 0.9891 |

η is remarkably flat across the comb (0.045–0.050, a gentle ~9% roll-off from pair 1 to pair 5).
The TPA term at full drive is `η² ≈ 2.1–2.5 mV`, comparable to the total single-beam leakage
(`a_x + a_w ≈ 1.9–2.0 mV`) — which is why steps 7/8 must carry the single-beam model along as a
fixed background rather than ignore it.

Per-pair plots: `calib_step6_pair1_0715_1714.png` … `pair5…png`.

---

## Step 7 — comb-phase spectrum {Φ_k}

**What it measures.** Each pair's TPA field carries a fixed comb-phase offset Φ_k. Driving a target
pair together with a reference pair makes the two fields interfere; the fringe encodes
`ΔΦ_comb = Φ_tgt − Φ_ref`. Reference = **pair 1** (defines Φ = 0); targets = pairs 3, 4, 5.

**Drive geometry.** Reference fully on (`x_r = w_r = 1`); the target's two channels swept
**together** (`x_t = w_t = v`, ramp 0.1 → 1.0, 10 points) plus one all-off dark. A channel at
intensity `v` sits at panel phase `θ = 2·asin(√v)` with field `√v·e^{iθ/2}`, so the target's field
amplitude is `g = sin²(θ/2) = v` and its SLM-commanded phase is `ΔΦ_SLM = θ − π`; the sweep traces
the half-fringe θ ≈ 37° → 180°.

**Fit model** (`slm_module.tpa_phase.fit_phase_ratio`; step-6 single-beam response folded in as a
*fixed* background):

$$Y = a^2 + b^2 g^2 + 2\,a\,b\,g\,\cos(\Delta\Phi_{SLM} + \Delta\Phi_{comb}) \;+\; \text{bg}_{step6}(v) \;+\; d$$

with `a` = reference amplitude, `b` = target amplitude. Two ways the step-6 amplitudes enter
(both stored in the result JSON):

- **`--bounded`** — ratio `a:b` locked to `η_ref:η_tgt`, one shared scale `s` floats (boxed
  `±BOUND_FRAC = 1.0` about 1). Free: ΔΦ_comb, s, d → dof 7.
- **`--fix`** — `a, b` pinned exactly at the step-6 etas (`s ≡ 1`). Free: ΔΦ_comb, d → dof 8.
  Doubles as a drift diagnostic: if the gain moved since step 6, this fit *must* degrade.

**Results** (refit of `calib_step7_meas_0715_1722.csv` → `calib_step7_result_0715_1756.json`):

| pair | Φ bounded (deg) | s | χ²/dof | R² | Φ fix (deg) | χ²/dof fix |
|---|---|---|---|---|---|---|
| 3 | **+17.35 ± 2.87** | 1.017 | 3.77 | 0.9968 | +16.72 ± 3.09 | 4.18 |
| 4 | **+7.87 ± 2.19** | 0.912 | 0.53 | 0.9994 | +1.47 ± 7.76 | 19.38 |
| 5 | **−13.00 ± 4.23** | 0.864 | 2.38 | 0.9965 | −29.10 ± 5.88 | 15.37 |

**Interpretation — bounded vs fix.** For pair 3 (s ≈ 1) the two methods agree within errors. For
pairs 4 and 5 the bounded scale s = 0.912 / 0.864 absorbed a real **9–14% amplitude drop** between
the step-6 and step-7 runs; pinning the amplitudes (`--fix`) forces that missing gain into the
phase and the dark offset, biasing Φ by 6–16° and blowing χ²/dof up to 15–19. The **bounded**
values are therefore the adopted spectrum. Note the drift is monotonic across the comb
(1.02 → 0.91 → 0.86), i.e. per-pair, not a single global gain.

The combined result JSON carries `step3` and `step6` forward verbatim plus the `step7` fits keyed
by `(tgt_index, method)` (`tpa_phase.save_comb_phase_json` / `load_comb_phase_json`), so it is the
single input for step 8.

---

## Step 8 — random-input check of the composed forward model

**What it tests.** Steps 3–7 were all *pairwise* calibrations. Step 8 closes the loop: drive
several pairs **at once** with random commanded intensities and compare the measurement against
the **zero-free-parameter** prediction

$$E = \sum_k \eta_k \sqrt{x_k w_k}\; e^{\,i[\varphi(x_k) + \varphi(w_k) + \Phi_k]},
\qquad Y_{pred} = |E|^2 + \sum_k \text{single\_beam}_k(x_k, w_k) \;(+\,\text{dark})$$

with `φ(x) = asin(√x)` (each channel's field phase at intensity x) and `Φ_1 = 0`. Every quantity
comes from the calibrations; nothing is fitted to the new data. Crucially, points where two
**non-reference** pairs are both bright probe `Φ_3 − Φ_5`, a combination step 7 never measured
directly — this is a genuine test that the phase spectrum *composes*.

**Drive.** Pairs `[1, 3, 5]`, 20 random vectors `[x_1,x_3,x_5], [w_1,w_3,w_5]` uniform in [0,1]
(seed `20260715`, reproducible), one all-off dark first. Spectrum: the **bounded** step-7 fits.

**Results** (`calib_step8_meas_0715_1811.csv`, dark = 2.233 mV, dark-subtracted signals
2.0–14.8 mV):

- Measured vs predicted tracks the y = x line across the full range
  (`calib_step8_meas_0715_1811_compare_bounded.png`): **rms residual ≈ 1.0 mV**, mean residual
  ≈ +0.02 mV (no systematic bias), i.e. the composed model predicts arbitrary 3-pair drives to
  ~10–15% point-wise with no tuning.
- Diagnostic global gain **α = 0.970** (all η² scaled together; `s = √α ≈ 0.985`) — no large
  common drift on top of what the bounded fits already absorbed.
- The **incoherent baseline** (cross terms dropped: self-TPA + background only) sits far off the
  measured points — the comb phases carry most of the predictive power, including the
  never-directly-measured Φ₃−Φ₅ cross term.
- **χ²/dof = 53.4** (55.0 after α-scaling), pulls up to ±15: the per-point SEM (0.07–0.25 mV,
  pure detector noise over 3 s) understates the true point-to-point error — the ~1 mV scatter is
  systematic.
- **Per-pair scale refit** (`fit_pair_scales`: s₁, s₃, s₅ float, one field-amplitude scale per
  pair, phases and background fixed — the per-pair analogue of step 7's bounded s):
  s = 0.877 ± 0.148 / 1.164 ± 0.150 / 0.867 ± 0.105, **χ²/dof = 55.3** (dof 17), rms only
  0.99 → 0.91 mV. **No collapse** — so the residual scatter is *not* per-pair amplitude drift.
  A synthetic control (injected s = [1.0, 0.9, 0.8] at SEM-level noise) confirms the refit would
  have caught a real drift: it recovers s to ±0.02 and collapses χ²/dof from 55 to **0.74**
  (a global α alone only reaches 3.9). With amplitude drift ruled out as dominant, the remaining
  suspects are drive-configuration-dependent systematics — the largest outliers sit at specific
  patterns (all-pairs-bright +2.8 mV, pairs 1+5 bright with 3 off −2.1 mV, x₅=1/w₅≈0.04 +1.6 mV) —
  e.g. SLM encoding crosstalk between simultaneously driven channels, or few-degree phase errors.
- Analyzing the same CSV against the **fix** spectrum is uniformly worse (rms 1.13 mV,
  χ²/dof = 62.9; per-pair scales don't rescue it either, 57.8) — the bounded spectrum is the
  better description of the composed system, as expected.

**Analysis-path validation (synthetic).** Before the hardware run, the analysis was checked on
synthetic data built from the same calibrations with an injected α = 0.85 + SEM-level noise:
recovered α = 0.8497 ± 0.0011, χ²/dof → 0.35 after scaling; re-analyzing the same data against the
*wrong* (`fix`) spectrum left χ²/dof ≈ 84 even after gain scaling. So the test genuinely
discriminates "gain drift" (α soaks it up) from "wrong phases" (nothing does).

---

## Reproducing / re-analyzing offline

```bash
# step 6: collect + fit (hardware), or refit an existing CSV
python src/drafts/calib_step6_test.py
python src/drafts/calib_step6_test.py src/calib_data/calib_step6_meas_0715_1714.csv

# step 7: collect (hardware); refit offline with both amplitude methods
# (the refit also writes the combined calib_step7_result_*.json)
python src/drafts/calib_step7_test.py
python src/drafts/calib_step7_test.py src/calib_data/calib_step7_meas_0715_1722.csv --bounded --fix

# step 8: collect + analyze (hardware), or re-analyze offline
python src/drafts/calib_step8_test.py
python src/drafts/calib_step8_test.py src/calib_data/calib_step8_meas_0715_1811.csv --bounded
```

Offline refit/analyze paths need no hardware. Loaders/fitters live in `slm_module.tpa_pair`
(step 6) and `slm_module.tpa_phase` (steps 7/8 models + combined-JSON IO); unit tests in
`tests/test_pipeline.py` (`python -m unittest tests.test_pipeline`).
