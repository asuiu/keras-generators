# pylint: disable=deprecated-module
import logging
from distutils.core import setup
from pathlib import Path
from typing import List

try:
    from pip._internal.network.session import PipSession
    from pip._internal.req import parse_requirements
except ImportError as exc:
    logging.critical("keras-generators requires pip>=20.0 in order to properly install dependencies. Consider upgrading pip")
    raise ImportError("keras-generators requires pip>=20.0 in order to properly install dependencies. Consider upgrading pip") from exc
from setuptools import find_packages

VERSION = "1.4.5"


def get_requirements(requirements_file: Path) -> List[str]:
    reqs = parse_requirements(str(requirements_file), session=PipSession())
    requirements_lst = [str(req.requirement) for req in reqs if req.requirement is not None]
    return requirements_lst


ROOT = Path(__file__).resolve().parent
readme = ROOT / "README.md"

requirements = get_requirements(ROOT / "requirements.txt")

setup(
    name="keras-generators",
    version=VERSION,
    description="Multi-dimensional/Multi-input/Multi-output Data preprocessing and Batch Generators for Keras models",
    package_data={"": ["README.md"]},
    long_description=readme.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Andrei Suiu",
    author_email="andrei.suiu@gmail.com",
    url="https://github.com/asuiu/keras-generators",
    keywords=["ML", "DataGenerators", "Keras", "tensorflow"],
    install_requires=requirements,
    extras_require={
        "tests": [],
    },
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries",
    ],
    license="Apache License 2.0",
    packages=find_packages(exclude=("*test*",)),
    setup_requires=[],
    python_requires=">=3.10",
)
