#!/usr/bin/env/ python3
#
# Copyright (c) 2018 Zhixian MA <zxma_sjtu@qq.com>
# MIT license
#
# References:
# [1] Python Packaging User Guide
#     https://packaging.python.org/

import os
import sys

from setuptools import setup, find_packages
# from setuptools.command.test import test as TestCommand

import mcrgnet as pkg


def read(filename):
    return open(os.path.join(os.path.dirname(__file__), filename)).read()

# Check python version
if sys.version_info < (3, 4):
    sys.exit("Python >= 3.4 is required...")

setup(
    name=pkg.__pkgname__,
    version=pkg.__version__,
    description=pkg.__description__,
    long_description=read("README.md"),
    author=pkg.__author__,
    author_email=pkg.__author_email__,
    url=pkg.__url__,
    license=pkg.__license__,
    keywords=pkg.__keywords__,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Image Classification ",
    ],
    packages=find_packages(exclude=["docs", "tests"]),
    scripts=[],
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "xlrd",
        "matplotlib",
        "astropy",
        "Pillow",
        "requests",
        "scikit-image",
        "scikit-learn",
        "tensorflow-gpu"
    ],
)
