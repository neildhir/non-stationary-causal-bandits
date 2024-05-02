#!/usr/bin/env python3
# Copyright (c) 2021 Neil Dhir
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from setuptools import setup, find_packages
import sys

# enforce >Python3 for all versions of pip/setuptools
assert sys.version_info >= (3,), "This package requires Python 3."

requirements = [
    "graphviz",
    "joblib",
    "matplotlib",
    "networkx",
    "numpy",
    "pandas",
    "pygraphviz",
    "scipy",
    "seaborn",
    "setuptools",
    "tqdm",
]

setup(
    name="ccb",
    version="0.1",
    description="Non-stationary multi-armed bandit under a causal perspective.",
    url="https://github.com/neildhir/non-stationary-causal-bandits",
    packages=find_packages(exclude=["test*"]),
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3",
    license="General Public License",
    author="Neil Dhir",
    author_email="ndhir@turing.ac.uk",
)
