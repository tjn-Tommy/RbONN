"""The ONN layer as a trainable torch module, and the bench handover.

The digital half of the project: it trains a network whose classifier head is the
bench's coherent quadratic sum, and writes down what the bench would have to
display and read back for the same numbers to be measured rather than simulated.
It drives no instrument.

    params   ONNParams -- the bench calibration (eta, Phi, Y_sat) in torch form.
             Loads through calibration_module.fit.verify, never by parsing a
             calibration JSON itself, so the phase-sign convention has exactly
             one definition in the repo.
    layer    ONNLayer -- K coherent channels -> O outputs, autograd all the way
             through, plus the tiling that runs a 784-feature layer as 79 shots
             of 10.  Pure torch; imports no calibration reader.
    finetune the MNIST runs.  On the bench PC it is for one short fine-tune after
             a recalibration; it can still train all three models from scratch
             (--model all), which is how the checkpoint it fine-tunes was made.
    bench_reference   the .npz the bench side consumes: the drives to display,
             the weights, and the Y each measurement should produce.
    onn_infer   the bench run itself -- display those drives, read the detector,
             undo the instrument response, and report whether the optics picked
             the right digit.

ONE PACKAGE, TWO ENVIRONMENTS.  Everything above runs in ``rbonn-ml`` (torch,
torchvision, scikit-learn) EXCEPT ``onn_infer``, which imports the SLM and DAQ
drivers and no torch, and so runs in ``rbonn``.  The npz is the seam, read with
numpy alone.  Neither half imports the other, so neither env needs the other's
dependencies.

The forward model exists twice on purpose: ``fit.verify.predict`` is the one
forward model of this project, in numpy, and every calibration fit is taken under
it, while ``ONNLayer`` evaluates the same equations in torch because autograd
needs them there.  ``tests/ml/test_onn_layer.py`` holds the two to 1e-12 on
random drives and is the only defence against the duplication -- if it fails, the
torch layer is wrong.
"""
