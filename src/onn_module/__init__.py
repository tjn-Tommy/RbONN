"""The ONN layer as a trainable torch module, and the bench handover.

This package is the digital half of the project: it trains a network whose
classifier head is the bench's coherent quadratic sum, and it writes down what
the bench would have to display and read back for the same numbers to be
measured rather than simulated.  It drives no instrument.

    params   ONNParams -- the bench calibration (eta, Phi, Y_sat) in torch form.
             Loads through calibration_module.fit.verify, never by parsing a
             calibration JSON itself, so the phase-sign convention has exactly
             one definition in the repo.
    layer    ONNLayer -- K coherent channels -> O outputs, autograd all the way
             through, plus the tiling that runs a 784-feature layer as 79 shots
             of 10.  Pure torch; imports no calibration reader.
    finetune the MNIST runs.  On the bench PC what it is for is one short
             fine-tune after a recalibration -- it can still train from scratch
             (--model all, the digital baseline), which is how the checkpoint it
             fine-tunes was made, elsewhere.
    bench_reference   the .npz the bench side consumes: the drives to display,
             the weights, and the Y each measurement should produce.
    onn_infer   the bench run itself -- display those drives, read the detector,
             undo the instrument response, and report whether the optics picked
             the right digit.

One package, two environments
-----------------------------
Everything above runs in `rbonn-ml` (torch) EXCEPT `onn_infer`, which imports the
SLM and DAQ drivers and no torch, and so runs in `rbonn`.  The npz is the seam
between them and is read with numpy alone.  Nothing in the torch half imports
`onn_infer`, and `onn_infer` imports nothing from the torch half -- so neither
env ever needs the other's dependencies.

Why the forward model exists twice
----------------------------------
``calibration_module.fit.verify.predict`` is the one forward model of this
project, in numpy, and every calibration fit is taken under it.  ``ONNLayer``
evaluates the same equations in torch because autograd needs them there, and
because a per-(batch, out, channel) numpy loop over 512 x 79 x 10 is not a
training step.  That is a genuine duplication and the only defence against it is
a test: ``tests/test_onn_layer.py`` holds the two to 1e-12 on random drives.  If
that test is failing, the torch layer is wrong -- ``verify.predict`` is the one
the bench is calibrated against.

Environment
-----------
Trains in the ``rbonn-ml`` conda env (torch, torchvision, scikit-learn), NOT in
``rbonn``, which is the hardware env and carries no torch.  The two meet through
files only: a step-7/step-8 calibration JSON going in, a ``bench_reference.npz``
coming out.  The bench side reads that npz with numpy alone.
"""
