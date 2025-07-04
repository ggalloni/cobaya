"""
Body of the best-fit test for cosmological likelihoods
"""

from copy import deepcopy

import numpy as np

from cobaya.cosmo_input import create_input, planck_base_model
from cobaya.input import update_info
from cobaya.model import get_model
from cobaya.tools import recursive_update
from cobaya.typing import empty_dict

from .common import process_packages_path
from .conftest import install_test_wrapper

# Tolerance for the tests of the derived parameters, in units of the sigma of Planck 2015
tolerance_derived = 0.055


def body_of_test(
    packages_path,
    best_fit,
    info_likelihood,
    info_theory,
    ref_chi2,
    best_fit_derived=None,
    extra_model=empty_dict,
    skip_not_installed=False,
):
    # Create base info
    theo = list(info_theory)[0]
    # In Class, theta_s is exact, but different from the approximate one cosmomc_theta
    # used by Planck, so we take H0 instead
    planck_base_model_prime = deepcopy(planck_base_model)
    planck_base_model_prime.update(extra_model or {})
    if "H0" in best_fit:
        planck_base_model_prime["hubble"] = "H"
        best_fit_derived = deepcopy(best_fit_derived) or {}
        best_fit_derived.pop("H0", None)
    info = create_input(planck_names=True, theory=theo, **planck_base_model_prime)
    # Add specifics for the test: theory, likelihoods and derived parameters
    info = recursive_update(info, {"theory": info_theory})
    info["theory"][theo]["use_renames"] = True
    info = recursive_update(info, {"likelihood": info_likelihood})
    info["params"].update(dict.fromkeys(best_fit_derived or []))
    # We need UPDATED info, to get the likelihoods' nuisance parameters
    info = update_info(info)
    # Notice that update_info adds an aux internal-only "params" property to the likes
    for lik in info["likelihood"]:
        info["likelihood"][lik].pop("params", None)
    info["packages_path"] = process_packages_path(packages_path)
    # Ask for debug output and force stopping at any error
    info["debug"] = True
    info["stop_at_error"] = True
    # Create the model and compute likelihood and derived parameters at best fit
    model = install_test_wrapper(skip_not_installed, get_model, info)
    best_fit_values = {p: best_fit[p] for p in model.parameterization.sampled_params()}
    likes, derived = model.loglikes(best_fit_values)
    likes = dict(zip(list(model.likelihood), likes))
    derived = dict(zip(list(model.parameterization.derived_params()), derived))
    # Check value of likelihoods
    for like in info["likelihood"]:
        chi2 = -2 * likes[like]
        if np.isnan(chi2):
            try:
                cl_dict = model.provider.get_Cl()
            except Exception:
                assert not np.isnan(chi2), f"Likelihood '{like}' is NaN"
            else:
                nan_cls = [key for key, arr in cl_dict.items() if np.isnan(arr).any()]
                assert not len(nan_cls), (
                    f"Testing likelihood '{like}': Cl arrays with NaN: {nan_cls}"
                )
        msg = (
            f"Testing likelihood '{like}': "
            f"| {chi2:.2f} (now) - {ref_chi2[like]:.2f} (ref) | = "
            f"{abs(chi2 - ref_chi2[like]):.2f} >=? {ref_chi2['tolerance']:.2f}"
        )
        assert abs(chi2 - ref_chi2[like]) < ref_chi2["tolerance"], msg
        print(msg)
    # Check value of derived parameters
    not_tested = []
    not_passed = []
    for p in best_fit_derived or {}:
        if best_fit_derived[p][0] is None or p not in best_fit_derived:
            not_tested += [p]
            continue
        rel = abs(derived[p] - best_fit_derived[p][0]) / best_fit_derived[p][1]
        if rel > tolerance_derived * (
            2 if p in ("YHe", "Y_p", "DH", "sigma8", "s8omegamp5", "thetastar") else 1
        ):
            not_passed += [(p, rel, derived[p], best_fit_derived[p][0])]
    if not_tested:
        print("Derived parameters not tested because not implemented: %r" % not_tested)
    assert not not_passed, (
        "Some derived parameters were off. Fractions of test tolerance: %r" % not_passed
    )
