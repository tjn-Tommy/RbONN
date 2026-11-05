# runs/mnist

MNIST inference on the bench: measure first, score second.  The CSV and the npz
are the only two things the scoring needs, so step 5 can be re-run any time --
on a different npz, or on a CSV with bad rows cut out -- without the bench.

1. Get `calib.json` with the UI.  It lands in `src/calib_data/<run>/`.

2. Send that json to the training machine, retrain against it, bring the
   `bench_reference_*.npz` back here.

3. Rehearse.  Solves everything, opens no instrument, prints the time estimate:

       python .\src\onn_module\onn_infer.py --npz .\runs\mnist\bench_reference_200img.npz --calib .\src\calib_data\1027_calib\ --out .\runs\mnist\ --dry-run

4. Measure -- the same command without `--dry-run`:

       python .\src\onn_module\onn_infer.py --npz .\runs\mnist\bench_reference_200img.npz --calib .\src\calib_data\1027_calib\ --out .\runs\mnist\

   Writes `onn_infer_<MMDD_HHMM>.csv` here: one row per read, the volts and the
   recovered `Y`, plus `kind=gain` anchor rows.  No accuracy -- it scores
   nothing.  `--out` is required and has no default, so a run can never
   quietly land in another dataset's folder.

5. Score:

       python .\runs\mnist\report.py .\runs\mnist\onn_infer_1027_2145.csv .\runs\mnist\bench_reference_200img.npz

   Three confusion matrices in `<csv stem>_confusion.png`.
   Only images read all the way through are scored; an interrupted run's
   partial image is dropped and counted in the header.

---

Steps 3-4 are also driven unattended by `runs\queue.ps1`, which runs this
alongside the fMNIST and 4-bit jobs and logs each to its own file.
