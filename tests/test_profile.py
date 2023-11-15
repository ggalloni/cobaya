# Profiling of a random Gaussian likelihood using the minimize sampler.

import dill
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
_inv_cov = np.linalg.inv(cov)

mean_c = mean[2]
sigma_c = round(np.sqrt(cov[2, 2]), 3)
profiled_values = [
    mean_c - 2 * sigma_c,
     mean_c - sigma_c,
     mean_c,
     mean_c + sigma_c,
     mean_c + 2 * sigma_c
]

class NoisyCovLike(Likelihood):
    params = {'a': [0, 1, 0.5, 0.3, 0.08], 'b': [0, 1, 0.5, 0.3, 0.08]}
    params['c'] = {'value': profiled_values, 'profiled': True}
    noise = 0

    def logp(self, **params_values):
        x = np.array([params_values['a'], params_values['b'], params_values['c']]) - mean
        return -_inv_cov.dot(x).dot(x) / 2 + self.noise * np.random.random_sample()


@mpi.sync_errors
def test_profile_gaussian(tmpdir):
    loglike_vals = [2, 0.5, 0, 0.5, 2]
    for method in reversed(valid_methods):
        NoisyCovLike.noise = 0.005 if method == 'bobyqa' else 0
        info: InputDict = {'likelihood': {'like': NoisyCovLike},
                           "sampler": {"minimize": {"ignore_prior": True,
                                                    "method": method}}}
        samplers = run(info)[1]
        for i, sampler in enumerate(samplers):
            products = sampler.products()
            assert abs(loglike_vals[i] - products["minimum"]["minuslogpost"]) < 0.01
        return


@mpi.sync_errors
def test_profile_output(tmpdir):
    for method in reversed(valid_methods):
        NoisyCovLike.noise = 0.005 if method == 'bobyqa' else 0
        info: InputDict = {'likelihood': {'like': NoisyCovLike},
                           'force': True,
                           "sampler": {"minimize": {"ignore_prior": True,
                                                    "method": method}}}
        info['output'] = os.path.join(tmpdir, 'testprof')
        samplers = run(info)[1]
        if mpi.is_main_process():
            filename = info['output'] + '.output_minima.dill_pickle'
            with open(filename, "rb") as f:
                res = dill.load(f)
            assert isinstance(res, dict)
            keys = list(res["c"].keys())
            expected_keys = ['values', 'minima', 'full_sets_of_mins', 'hessians', 'other_params']
            assert all(k in expected_keys for k in keys)