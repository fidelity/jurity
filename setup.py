# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

import os

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as fh:
    required = fh.read().splitlines()

with open(os.path.join('jurity', '_version.py')) as fp:
    exec(fp.read())

setuptools.setup(
    name="jurity",
    description="fairness and evaluation library",
    long_description=long_description,
    version=__version__,
    author="FMR LLC",
    url="https://github.com/fidelity/jurity",
    packages=setuptools.find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Operating System :: OS Independent",
    ],
    project_urls={"Source": "https://github.com/fidelity/jurity"},
    install_requires=required,
    python_requires=">=3.6"
)
