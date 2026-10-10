"""The ONN layer as a trainable torch module, and the bench handover.

The digital half of the project: it trains a network whose classifier head is
the bench's coherent quadratic sum, and writes down what the bench would have to
display and read back for the same numbers to be measured rather than simulated.
It drives no instrument.

    params   ONNParams -- the bench calibration (eta, Phi, Y_sat) in torch form.
             Loads through calibration_module.fit.verify, never by parsing a
             calibration JSON itself.
    layer    ONNLayer -- K coherent channels -> O outputs, autograd all the way
             through, plus the tiling that runs a 784-feature layer as 79 shots
             of 10.  Pure torch; imports no calibration reader.
    finetune the MNIST runs: one short fine-tune after a recalibration, or all
             three models from scratch (--model all).
    bench_reference   the .npz the bench side consumes: the drives to display,
             the weights, and the Y each measurement should produce.
    onn_infer   the bench run itself -- display those drives, read the detector,
             undo the instrument response, and report whether the optics picked
             the right digit.

ONE PACKAGE, TWO ENVIRONMENTS.  Everything above runs in ``rbonn-ml`` (torch,
torchvision, scikit-learn) except ``onn_infer``, which imports the SLM and DAQ
drivers and no torch, and so runs in ``rbonn``.  The npz is the seam, read with
numpy alone, and neither half imports the other.

The forward model exists twice: ``fit.verify.predict`` is this project's one
forward model, in numpy, and every calibration fit is taken under it, while
``ONNLayer`` evaluates the same equations in torch for autograd.
``tests/ml/test_onn_layer.py`` holds the two to 1e-12 on random drives.
"""
