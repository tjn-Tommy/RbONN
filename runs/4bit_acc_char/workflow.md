# runs/4bit_acc_char

How many bits the bench actually resolves, measured inside the band the ONN
really uses.  A 16-level ladder evenly spaced over `alpha` 0..0.25 -- the 1027
run never drove past 0.235 -- each level re-measured 40 times, the whole
(level x repeat) list shuffled so 45 minutes of drift cannot land on the level
axis.  Measure first, analyse second.

1. Get `calib.json` with the UI.  Only its etas and phases are used: they
   decide which drive is which `alpha`.  Its gain never enters, because nothing
   here is compared against an absolute prediction.

2. Rehearse.  Solves the 16 drives, prints the estimate, opens nothing:

       python .\src\drafts\bit_acc_chara.py --calib .\src\calib_data\1027_calib\ --dry-run

3. Measure, ~46 min.  The GUI must not be holding the DAQ or the SLM -- the
   DAQ is opened first, so a busy machine stops the run before the panel is
   touched:

       python .\src\drafts\bit_acc_chara.py --calib .\src\calib_data\1027_calib\

   Writes `bit_acc_chara_<MMDD_HHMM>.csv` here (`--out` moves it).  Stops
   itself if any raw trace peaks at 0.90 V, and re-reads the dark throughout
   rather than holding one for the whole run.

   `--t-meas` defaults to 4 s, the window the 1027 ONN run read at, because the
   resolution being measured should be the resolution that run had.  A longer
   window buys a lower sigma and answers a different question.

4. Analyse:

       python .\runs\4bit_acc_char\report.py .\runs\4bit_acc_char\bit_acc_chara_<MMDD_HHMM>.csv

   Per-level table, then INL, DNL, repeat noise and the resolvable bit count in
   the terminal, plus `<csv stem>_ladder.png` -- measured against predicted,
   with the residual underneath.

---

Steps 2-3 are also driven unattended by `runs\queue.ps1`, which runs the ladder
at both ends of the queue -- it does not anchor its own gain, so the pair is
what says whether the bench held still across the whole thing.
