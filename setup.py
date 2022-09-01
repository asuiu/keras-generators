from distutils.core import setup
from pathlib import Path

from setuptools import find_packages

VERSION = '1.0.1'

this_file = Path(__file__).resolve()
readme = this_file.parent / "README.md"

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
    install_requires=[
        "packaging",
        "tensorflow>=2.8.0",
        "scikit-learn>=0.22.2",
        "numpy>=1.20",
    ],
    extras_require={
        "tests": [],
    },
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries",
    ],
    license="Apache License 2.0",
    packages=find_packages(exclude=("*test*",)),
    setup_requires=[],
    python_requires='>=3.7'
)
