# `calib_data`

Measured calibration and accuracy data, one folder per run. The CSV and JSON
are tracked; **PNGs are gitignored** (`.gitignore`: `src/calib_data/**/*.png`)
because every one of them is regenerable from the CSV/JSON beside it via the
step's own `save_plot`. That is why the folder is ~75 MB on disk but only
~13 MB in git.

## Naming

    <kind>_<MMDD>_<HHMM>/

`<kind>` says what the run *is*, `<MMDD>_<HHMM>` is when it **started**:

| kind | what it holds |
|---|---|
| `calib_` | a calibration: step 3c + step 6 + step 7 results for one panel setup |
| `accuracy_` | a `system_accuracy` sweep, built on some `calib_` run's step-7 JSON |
| `bench_` | bench/characterisation work that is neither |

Two folders must never differ only by punctuation. `run_0917`, `run_0917_` and
`run_0917_overnight` were three different things — a calibration, the 18-run
accuracy sweep built on it, and an earlier accuracy sweep built on the *0916*
calibration — and that cost us a batch pointed at the wrong folder. They are
now `calib_0917_1617/`, `accuracy_0917_1834/` and `accuracy_0917_0010/`.

`overnight_system_accuracy.bat` derives its `OUT` from the batch's start time,
so a night of accuracy runs can no longer land in the calibration folder it was
built from. Don't hard-code `OUT` back to a `calib_` folder.

A folder holding a *series* of runs keeps one `system_accuracy_combined.csv`
instead of one CSV per run: the per-run files were 18 and 12 copies of the same
53-column header, only comparable once you knew which file a row came from. The
combined file promotes each run's trailing `# key,value` metadata to `meta_*`
columns and adds `run` and `run_start`, so every row says for itself which run,
which hour and which step-7 calibration it belongs to. Nothing is recomputed --
it is a concatenation, verified field by field against the originals before they
were removed. The per-run report PNGs are kept.

Do not merge two such folders into one table. `accuracy_0917_1834` was built on
`calib_0917_1617` and `accuracy_0917_0010` on `run_0916_10pairs_w15g35`; they
share no drive vectors and their residuals are not comparable.

Older runs still carry the legacy `run_<MMDD>_<desc>` form. Renaming them is
not free — see below.

## Do not move these without updating the tests

Six runs are fixtures for the test suite. Renaming or archiving one breaks
tests with a path error, not an assertion, so it is easy to misread:

| folder | used by |
|---|---|
| `calib_0917_1617/` | `tests/test_system_accuracy.py` |
| `run_0911_10pairs_w20g30/` | `tests/test_system_accuracy.py`, `tests/test_levels.py` |
| `run_0916_10pairs_w15g35/` | `tests/test_gui_step7.py`, `tests/test_cascade.py` |
| `archive/2026-09/run_0907_1724/` | `tests/test_pair_v2.py`, `tests/test_gui_step6.py`, `tests/test_gui_step7.py` |
| `archive/2026-09/run_0908_1300/` | `tests/test_npair_verify.py` |
| `archive/2026-09/run_0908_1444/` | `tests/test_step67_cli.py`, `tests/test_gui_step6.py` |

To re-check the list:

    grep -rn "calib_data" tests/

## Which runs are comparable

The channel layout changed between 0911 and 0916: spacing went
**0.2530 -> 0.2811 nm/slot** with the centre unchanged at 778.04 nm. Because a
pair's two-photon sum detuning goes as the *square* of its offset from centre,
that pushed the outer pairs a long way off resonance — pair 9 from 7.4 to
9.1 GHz, pair 10 from 9.0 to 11.1 GHz — and their `eta` collapsed by ~3x.

So `run_0911_*` and `run_0916_*`/`calib_0917_*` are **different optical
epochs**. Comparing a per-pair `eta` across that boundary compares two
different channel placements. Within an epoch they agree to a few percent
(the two 0911 runs differ only in SLM window width, 15 px vs 20 px, which
scales every channel by a common factor).
