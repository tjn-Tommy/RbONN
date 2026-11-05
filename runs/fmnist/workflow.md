# runs/fmnist

Fashion-MNIST on the bench: measure first, score second.  The CSV and the npz
are the only two things the scoring needs, so step 5 can be re-run any time --
on a different npz, or on a CSV with bad rows cut out -- without the bench.

Not the 10-class task.  This is a **same/different pair** task: each trial is
two Fashion-MNIST images concatenated into one 200-wide input, and the one
output answers "are these the same class?".  So `n_out = 1`, `n_tiles = 20`,
20 reads per pair, 4000 reads for the 200-pair dump.

The decision is `logit = v . y + c > 0` with **`c = 0`**, which makes the
verdict **gain-invariant**: a common scale error on every read -- gain drift,
a PMT setting, a power change -- multiplies the logit and cannot move it
across zero.  The npz declares this as `meta.gain_invariant`.  It is the
reason the bench reproduced all 200 verdicts over a 5-hour run; do not expect
the same of `runs/mnist`, whose per-output `out_scale` is not argmax-invariant.

1. Get `calib.json` with the UI.  It lands in `src/calib_data/<run>/`.

2. Send that json to the training machine, retrain against it, bring the
   `bench_reference_*.npz` back here.

3. Rehearse.  Solves everything, opens no instrument, prints the time estimate:

       python .\src\onn_module\onn_infer.py --npz .\runs\fmnist\bench_reference_200pairs.npz --calib .\src\calib_data\1027_calib\ --out .\runs\fmnist\ --dry-run

4. Measure -- the same command without `--dry-run`:

       python .\src\onn_module\onn_infer.py --npz .\runs\fmnist\bench_reference_200pairs.npz --calib .\src\calib_data\1027_calib\ --out .\runs\fmnist\

   Writes `onn_infer_<MMDD_HHMM>.csv` here: one row per read, the volts and the
   recovered `Y`, plus `kind=gain` anchor rows.  No accuracy -- it scores
   nothing.  `--out` is required and has no default, so a run can never
   quietly land in another dataset's folder.  Budget ~5 h for 200 pairs.

5. Score -- note this script takes **flags**, not positionals, unlike
   `runs\mnist\report.py`:

       python .\runs\fmnist\report.py --csv .\runs\fmnist\onn_infer_1029_2034.csv --npz .\runs\fmnist\bench_reference_200pairs.npz

   Three 2x2 confusion matrices in `<csv stem>_confusion.png` and on the
   terminal: A simulation vs truth, B bench vs truth, C bench vs simulation.
   C is the one that matters -- A and B can score the same on different pairs.
   Writes no CSV.

   It then lists every pair where the bench and the simulation disagree, the
   `Y` residual (mean, rms, best-fit slope), and the npz's own
   `acc_vs_read_noise_mV` curve with the run's measured rms marked, so the
   measured accuracy can be read against what that much noise predicts.

   Only pairs read all the way through are scored; an interrupted run's
   partial pair is dropped and counted in the header.

---

Steps 3-4 are also driven unattended by `runs\queue.ps1`, which runs this
alongside the MNIST and 4-bit jobs and logs each to its own file.
