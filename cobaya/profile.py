"""
.. module:: profile

:Synopsis: Functions necessary to profile a parameter
:Author: Giacomo Galloni

"""
import os
from numbers import Real
from typing import Any, Callable, Dict, List, Mapping, Sequence, Set, Tuple, Union

# Global
import numpy as np

from cobaya.conventions import Extension
from cobaya.log import HasLogger, LoggedError
from cobaya.model import Model, get_model
from cobaya.output import Output
from cobaya.sampler import Sampler
from cobaya.parameterization import is_profiled_param
from cobaya.tools import deepcopy_where_possible, recursive_update
from cobaya.yaml import yaml_load_file, yaml_dump_file

# Local
from cobaya.typing import InputDict


hess_attr = {"scipy": "hess_inv", "bobyqa": "hessian", "iminuit": "hess_inv"}


def check_if_any_profiled(info: InputDict) -> bool:
    """
    This checks whether a profiled parameter is present in the input dictionary.
    """
    info_params = info["params"]
    is_profiled = []
    for param in info_params:
        is_profiled.append(is_profiled_param(info_params[param]))
    if any(is_profiled):
        return True
    return False


def get_profiled_values(info: InputDict) -> Tuple[str, list]:
    """
    This returns the profiled parameter and the requested values.
    """
    info_params = info["params"]
    for param in info_params:
        if is_profiled_param(info_params[param]):
            return param, info_params[param]["value"]
    return None, None


def get_profiled_Model(oldModel: Model, profiled_param: str, value: float) -> Model:
    """
    This returns a new model with the profiled parameter fixed to one of the requested values.
    """
    info_model = oldModel.info()
    info_model["params"][profiled_param]["value"] = value
    info_model["params"][profiled_param]["profiled"] = False
    return get_model(info_model)


def initialize_results(profiled_param: str) -> Tuple[dict, dict]:
    """
    This initializes the dictionary that will contain the results of the run.
    """
    minima_results = {}
    minima_results[f"{profiled_param}"] = {
        "value": [],
        "minimum": [],
        "hessian": [],
        "other_params": [],
        "full_set_of_mins": [],
    }
    return minima_results


def get_results(
    profiled_param: str, value: float, sampler: Sampler, minima_results: dict):
    """
    This fills the results dictionary with the outcome of the single run.
    """
    minima_results[f"{profiled_param}"]["value"].append(value)

    minima_results[f"{profiled_param}"]["minimum"].append(
        sampler.minimum.data.get("chi2") / 2 if sampler.ignore_prior
        else sampler.minimum.data.get("minuslogpost"))

    minima_results[f"{profiled_param}"]["full_set_of_mins"].append(
        sampler.full_set_of_mins
    )

    hess_attr_ = hess_attr[sampler.method.lower()]
    transformation_mat = sampler._inv_affine_transform_matrix
    hessian = getattr(sampler.result, hess_attr_)
    if sampler.method.lower() == "iminuit":
        hessian = np.linalg.inv(hessian)
    if sampler.method.lower() == "scipy":
        hessian = np.linalg.inv(hessian.todense())
    hessian = transformation_mat @ hessian @ transformation_mat.T
    minima_results[profiled_param]["hessian"].append(hessian)

    x_min = sampler.inv_affine_transform(sampler.result.x)
    minima_results[f"{profiled_param}"]["other_params"].append(
        dict(zip(sampler.minimum.sampled_params +
                               sampler.minimum.derived_params,
                               list(x_min) +
                               list(sampler.model.logposterior(x_min, cached=False).derived)))
    )
    return minima_results


def save_results(output: Output, minima_results: dict):
    """
    This saves the results of the run in a binary file.
    """
    file = output.prefix + ".output_minima" + Extension.dill
    if os.path.isfile(file):
        with open(file, "rb") as f:
            try:
                import dill
                old_minima_results = dill.load(f)
            except ImportError:
                raise LoggedError(
                    output.log,
                    'Install "dill" to save reproducible options file.',
                )
            except EOFError:
                with open(file, "wb") as f:
                    dill.dump(minima_results, f)
                return
        for key in minima_results.keys():
            if key in old_minima_results.keys():
                for idx, val in enumerate(old_minima_results[key]["value"]):
                    if val not in minima_results[key]["value"]:
                        minima_results[key]["value"].append(val)
                        minima_results[key]["minimum"].append(
                            old_minima_results[key]["minimum"][idx]
                        )
                        minima_results[key]["full_set_of_mins"].append(
                            old_minima_results[key]["full_set_of_mins"][idx]
                        )
                        minima_results[key]["hessian"].append(
                            old_minima_results[key]["hessian"][idx]
                        )
                        minima_results[key]["other_params"].append(
                            old_minima_results[key]["other_params"][idx]
                        )
        old_minima_results = {**old_minima_results, **minima_results}
        with open(file, "wb") as f:
            dill.dump(old_minima_results, f)
    else:
        with open(file, "wb") as f:
            dill.dump(minima_results, f)
    return


def profiled_run(
    info: InputDict,
    model: Model,
    sampler_name: str,
    sampler_class: str,
    out,
    packages_path_input,
    mpi,
    logger_run,
) -> None:
    """
    This is the main function that performs multiple runs of the sampler to profile a parameter and save the results.
    """
    # This stores a copy of the input dictionary
    complete_info = deepcopy_where_possible(info)

    # This is the list of samplers that will be returned
    samplers = []

    # This collects the profiled parameter and the requested values to profile
    profiled_param, profiled_values = get_profiled_values(info)
    out.log.info("Profiling requested.\nParameter: %s\nValues: %s",
                    profiled_param, profiled_values)

    # This initializes the dictionary that will contain the results of the runs
    minima = initialize_results(profiled_param)

    # This loops over the values to profile
    out.log.info("Start looping on values...")
    for value in profiled_values:
        out.log.info("Run %s/%s, with %s = %s", profiled_values.index(value) + 1,
                        len(profiled_values), profiled_param, value)
        # This updates the input dictionary with the value to profile and get the new model
        info["params"][profiled_param]["value"] = value
        profiled_model = get_profiled_Model(model, profiled_param, value)

        sampler = sampler_class(
            info["sampler"][sampler_name],
            profiled_model,
            out,
            name=sampler_name,
            packages_path=info.get(packages_path_input),
        )
        info["sampler"][sampler_name] = recursive_update(
            info["sampler"][sampler_name], sampler.info()
        )

        mpi.sync_processes()
        if complete_info.get("test", False):
            logger_run.info(
                "Test initialization successful! "
                "You can probably run now without `--%s`.",
                "test",
            )
            return complete_info, sampler

        sampler.run()

        out.log.info("Finished run %s/%s.", profiled_values.index(value) + 1,
                        len(profiled_values))

        # This saves the results of the run
        if mpi.is_main_process():
            samplers.append(sampler)

            # This updates the dictionaries with the results of the run
            minima = get_results(profiled_param, value, sampler, minima)

            # Loads, updates and saves the results of the run
            print(minima)
            # save_results(out, minima)

    out.check_and_dump_info(None, info, check_compatible=False)

    return complete_info, samplers
