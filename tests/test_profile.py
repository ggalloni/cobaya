# Profiling of a random Gaussian likelihood using the minimize sampler.

import numpy as np
import pytest
import os

from cobaya import mpi, run, InputDict, Likelihood
from cobaya.samplers.minimize import valid_methods

pytestmark = pytest.mark.mpi

mean = np.array([0.30245268, 0.61884443, 0.5])
cov = np.array(
    [[0.00796336, -0.0014805, -0.00479433], [-0.0014805, 0.00561415, 0.00434189],
     [-0.00479433, 0.00434189, 0.03208593]])
sigma_c = np.sqrt(cov[2, 2])
_inv_cov = np.linalg.inv(cov)


class NoisyCovLike(Likelihood):
    params = {'a': [0, 1, 0.5, 0.3, 0.08], 'b': [0, 1, 0.5, 0.3, 0.08]}
    params['c'] = {'value': [mean-sigma_c, mean, mean+sigma_c], 'profiled': True}
    noise = 0

    def logp(self, **params_values):
        x = np.array([params_values['a'], params_values['b'], params_values['c']]) - mean
        return -_inv_cov.dot(x).dot(x) / 2 + self.noise * np.random.random_sample()


@mpi.sync_errors
def test_profile_gaussian(tmpdir):
    maxlogliks = [1, 0, 1]
    for method in reversed(valid_methods):
        NoisyCovLike.noise = 0.005 if method == 'bobyqa' else 0
        info: InputDict = {'likelihood': {'like': NoisyCovLike},
                           "sampler": {"minimize": {"ignore_prior": True,
                                                    "method": method}}}
        samplers = run(info)[1]
        errors = []
        for i, sampler in enumerate(samplers):
            products = sampler.products()
            errors.append(abs(maxlogliks[i] - -products["minimum"]["minuslogpost"]))
        assert errors[0] < 0.01
        assert errors[1] < 0.01
        assert errors[2] < 0.01
        return