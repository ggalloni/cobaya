"""
.. module:: containers

:Synopsis: Functions and scripts to manage container images
:Author: Jesus Torrado

"""

import argparse
import os
import uuid
from io import StringIO
from subprocess import PIPE, Popen
from tempfile import NamedTemporaryFile
from textwrap import dedent

from requests import head

from cobaya.conventions import (
    code_path,
    data_path,
    packages_path_arg,
    packages_path_env,
    packages_path_input,
    products_path,
)
from cobaya.input import get_used_components, load_input
from cobaya.install import install
from cobaya.log import LoggedError, get_logger, logger_setup
from cobaya.tools import warn_deprecation
from cobaya.yaml import yaml_dump

log = get_logger(__name__)

_requirements_file = "requirements.yaml"
_help_file = "readme.md"

requirements_file_path = os.path.join(packages_path_input, _requirements_file)
help_file_path = os.path.join(packages_path_input, _help_file)

base_recipe = r"""
# OS -------------------------------------------------------------------------
FROM ubuntu:xenial
# POST -----------------------------------------------------------------------
RUN sed -i 's/# \(.*multiverse$\)/\1/g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y \
      autoconf automake make gcc-6-base nano \
      libopenblas-base liblapack3 liblapack-dev libcfitsio-dev \
      python python-pip git wget
# Python requisites -- LC_ALL=C: Necessary just for pip <= 8.1.2 (Xenial version)
ENV LC_ALL C
RUN python -m pip install --upgrade pip
RUN python -m pip install pytest-xdist matplotlib cython astropy --upgrade
# Prepare environment and tree for external packages -------------------------
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/lib
ENV CONTAINED TRUE
ENV {} {}
ENV COBAYA_PRODUCTS {}
RUN mkdir ${} && \
    mkdir $COBAYA_PRODUCTS
# COBAYA  --------------------------------------------------------------------
# getdist fork (it will be an automatic requisite in the future)
RUN cd ${} && git clone https://github.com/JesusTorrado/cobaya.git && \
    cd ${}/cobaya && python -m pip install -e .
""".format(
    packages_path_env,
    packages_path_input,
    products_path,
    packages_path_env,
    packages_path_env,
    packages_path_env,
)

MPI_URL = {
    "mpich": "https://www.mpich.org/static/downloads/_VER_/mpich-_VER_.tar.gz",
    "openmpi": (
        "https://www.open-mpi.org/software/ompi/v_VER_/downloads/"
        "openmpi-_VER__DOT_SUB_.tar.gz"
    ),
}

MPI_versions = {"mpich": {"3.2": None}, "openmpi": {"2.1": ["1"]}}  # , "2"]}}

MPI_recipe = {
    "mpich": """
    # NERSC: must be MPICH >= 3.2
    # https://www.nersc.gov/users/software/using-shifter-and-docker/using-shifter-at-nersc/
    RUN cd /tmp && wget _URL_ && tar xvzf mpich-_VER_.tar.gz && cd /tmp/mpich-_VER_ && \
      ./configure --prefix=/usr/local && make -j4 && make install && make clean && cd .. \
      rm -rf /tmp/mpich-* """,
    "openmpi": """
    # HPC: must be OpenMPI >= 2.1
    RUN cd /tmp && wget _URL_ && gunzip -c openmpi-_VER__DOT_SUB_.tar.gz | tar xf - && \
      cd /tmp/openmpi-_VER__DOT_SUB_ && ./configure --prefix=/usr/local && make -j4 && \
      make install && make clean && cd .. && rm -rf /tmp/openmpi-_VER__DOT_SUB_ """,
}

MPI_epilogue = """&& export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/ && \
                  ldconfig && python -m pip install mpi4py --no-binary :all:"""


def image_help(engine):
    e = engine.lower()
    assert e in ("singularity", "docker"), e + " not valid."
    mount_data = {
        "docker": "-v [/cluster/path/to/data]:/packages/data:rw",
        "singularity": "--bind [/cluster/path/to/data]:/packages/data",
    }
    mount_products = {
        "docker": "-v [/cluster/path/to/products]:/products:rw",
        "singularity": "--bind [/cluster/path/to/producs]:/products",
    }
    mount_tmp = {"docker": "-v /tmp:/products:rw", "singularity": "--bind /tmp:/products"}
    pre_prepare = {
        "docker": " ".join(["?????", mount_data[e]]),
        "singularity": " ".join(["singularity exec", mount_data[e], "[image_file]"]),
    }
    pre_run = {
        "docker": " ".join(["?????", mount_data[e], mount_products[e]]),
        "singularity": " ".join([mount_data[e], mount_products[e], "[image_file]"]),
    }
    pre_shell = {
        "docker": " ".join(["?????", mount_data[e], mount_tmp[e]]),
        "singularity": " ".join(
            ["singularity shell", mount_data[e], mount_tmp[e], "[image_file]"]
        ),
    }
    return dedent(
        """
        This is a {} image for Cobaya.

        To check the packages installed in the container, take a look at '{}'.

        Make sure that you have created a 'data' and a 'products' folder in your cluster.

        To prepare the data needed for the container, while in the cluster, run:

            $ {} cobaya-prepare-data  # --force

        To run a sample with an input file 'somename.yaml', send to you cluster scheduler:

            $ mpi[run|exec] [mpi options] {} cobaya-run somename.yaml

        To open a terminal in the container, for testing purposes do:

            $ {}

        Have fun!
        """.format(
            engine.title(),
            requirements_file_path,
            pre_prepare[engine.lower()],
            pre_run[engine.lower()],
            pre_shell[engine.lower()],
        )
    )


def get_docker_client():
    try:
        import docker
    except ImportError:
        raise LoggedError(
            log,
            "The Python Docker interface not installed: "
            "do 'python -m pip install docker'.",
        )
    return docker.from_env(version="auto")


def create_base_image(mpi=None, version=None, sub=None):
    """
    `mpi` : MPI library to compile: OpenMPI or MPICH.
    `ver` : major.minor, as a string.
    `sub` : (optional) the subversion of the major.minor version
    """
    if version is None:
        raise LoggedError(log, "Needs a major.minor version!")
    sub = sub or ""
    if sub:
        sub = "." + sub
    try:
        tag = f"cobaya/base_{mpi.lower()}_{version + sub}:latest"
    except KeyError:
        raise LoggedError(log, "MPI library '%s' not recognized.", mpi)
    URL = MPI_URL[mpi.lower()].replace("_VER_", str(version)).replace("_DOT_SUB_", sub)
    if head(URL).status_code >= 400:
        raise LoggedError(
            log,
            "Failed to download %s %s: couldn't reach URL: '%s'",
            mpi.lower(),
            version + sub,
            URL,
        )
    log.info("Creating base image %s v%s...", mpi.lower(), version + sub)
    this_MPI_recipe = dedent(
        MPI_recipe[mpi.lower()]
        .replace("_VER_", version)
        .replace("_DOT_SUB_", sub)
        .replace("_URL_", URL)
    )
    dc = get_docker_client()
    with StringIO(base_recipe + this_MPI_recipe + MPI_epilogue) as stream:
        dc.images.build(fileobj=stream, tag=tag, nocache=True)
    log.info("Base image '%s' created!", tag)


def create_all_base_images():
    log.info("Creating all base images. This will take a while.")
    for mpi in MPI_recipe:
        for version, subs in MPI_versions[mpi.lower()].items():
            for sub in subs or [None]:
                create_base_image(mpi, version, sub)
    log.info("All base images created. Don't forget to *PUSH*!")


def create_docker_image(filenames, MPI_version=None):
    log.info("Creating Docker image...")
    if not MPI_version:
        MPI_version = "3.2"
        # log.warning("You have not specified an MPICH version. "
        #          "It is strongly encouraged to request the one installed in your cluster,"
        #          " using '--mpi-version X.Y'. Defaulting to MPICH v%s.", MPI_version)
    dc = get_docker_client()
    components = yaml_dump(
        get_used_components(*[load_input(f) for f in filenames])
    ).strip()
    echos_reqs = "RUN " + " && \\ \n    ".join(
        [
            rf'echo "{block}" >> {requirements_file_path}'
            for block in components.split("\n")
        ]
    )
    echos_help = "RUN " + " && \\ \n    ".join(
        [
            rf'echo "{line}" >> {help_file_path}'
            for line in image_help("docker").split("\n")
        ]
    )
    recipe = r"""
    FROM cobaya/base_mpich_{}:latest
    {}
    RUN cobaya-install {} --{} {} --just-code --force ### NEEDS PYTHON UPDATE! --no-progress-bars
    {}
    CMD ["cat", "{}"]
    """.format(
        MPI_version,
        echos_reqs,
        requirements_file_path,
        packages_path_arg,
        packages_path_input,
        echos_help,
        help_file_path,
    )
    image_name = "cobaya:" + uuid.uuid4().hex[:6]
    with StringIO(recipe) as stream:
        dc.images.build(fileobj=stream, tag=image_name)
    log.info(
        "Docker image '%s' created! "
        "Do 'docker save %s | gzip > some_name.tar.gz'"
        "to save it to the current folder.",
        image_name,
        image_name,
    )


def create_singularity_image(filenames, MPI_version=None):
    log.info("Creating Singularity image...")
    if not MPI_version:
        MPI_version = "2.1.1"
        # log.warning("You have not specified an OpenMPI version. "
        #          "It is strongly encouraged to request the one installed in your cluster,"
        #          " using '--mpi-version X.Y.Z'. Defaulting to OpenMPI v%s.", MPI_version)
    components = yaml_dump(
        get_used_components(*[load_input(f) for f in filenames])
    ).strip()
    echos_reqs = "\n    " + "\n    ".join(
        [""]
        + [
            f'echo "{block}" >> {requirements_file_path}'
            for block in components.split("\n")
        ]
    )
    recipe = (
        dedent(
            """
        Bootstrap: docker
        From: cobaya/base_openmpi_%s:latest\n
        %%post\n"""
            % MPI_version
        )
        + dedent(echos_reqs)
        + dedent(
            """
        export CONTAINED=TRUE
        cobaya-install {} --{} {} --just-code --force ### --no-progress-bars
        mkdir $COBAYA_PRODUCTS

        %help

        {}
        """.format(
                requirements_file_path,
                # TODO: this looks wrong?
                packages_path_input,
                os.path.join(packages_path_arg, packages_path_input, data_path),
                "\n        ".join(image_help("singularity").split("\n")[1:]),
            )
        )
    )
    with NamedTemporaryFile(delete=False) as recipe_file:
        recipe_file.write(recipe.encode("utf-8"))
        recipe_file_name = recipe_file.name
    image_name = "cobaya_" + uuid.uuid4().hex[:6] + ".simg"
    process_build = Popen(
        ["singularity", "build", image_name, recipe_file_name], stdout=PIPE, stderr=PIPE
    )
    out, err = process_build.communicate()
    if process_build.returncode:
        log.info(out)
        log.info(err)
        raise LoggedError(log, "Image creation failed! See error message above.")
    log.info("Singularity image '%s' created!", image_name)


# Command-line scripts ###################################################################


def create_image_script():
    warn_deprecation()
    logger_setup()
    parser = argparse.ArgumentParser(
        prog="cobaya create-image",
        description=(
            "Cobaya's tool for preparing Docker (for Shifter) and Singularity images."
        ),
    )
    parser.add_argument(
        "files",
        action="store",
        nargs="+",
        metavar="input_file.yaml",
        help="One or more input files.",
    )
    parser.add_argument(
        "-v",
        "--mpi-version",
        action="store",
        default=None,
        metavar="X.Y(.Z)",
        dest="version",
        help="Version of the MPI lib.",
    )
    group_type = parser.add_mutually_exclusive_group(required=True)
    group_type.add_argument(
        "-d",
        "--docker",
        action="store_const",
        const="docker",
        help="Create a Docker image (for Shifter).",
        dest="type",
    )
    group_type.add_argument(
        "-s",
        "--singularity",
        action="store_const",
        dest="type",
        const="singularity",
        help="Create a Singularity image.",
    )
    arguments = parser.parse_args()
    if arguments.type == "docker":
        create_docker_image(arguments.files, MPI_version=arguments.version)
    elif arguments.type == "singularity":
        create_singularity_image(arguments.files, MPI_version=arguments.version)


def prepare_data_script():
    warn_deprecation()
    logger_setup()
    if "CONTAINED" not in os.environ:
        raise LoggedError(
            log,
            "This command should only be run within a container. "
            "Run 'cobaya-install' instead.",
        )
    parser = argparse.ArgumentParser(
        prog="cobaya prepare-data",
        description="Cobaya's installation tool for the data needed by a container.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        help="Force re-installation of apparently installed packages.",
    )
    arguments = parser.parse_args()
    try:
        info = load_input(requirements_file_path)
    except OSError:
        raise LoggedError(
            log, "Cannot find the requirements file. This should not be happening."
        )
    install(
        info,
        path=packages_path_input,
        force=arguments.force,
        **{code_path: False, data_path: True},
    )
