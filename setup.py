#!/usr/bin/env python
import re
import sys
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

from setuptools import setup, find_packages
from subprocess import CalledProcessError, check_output

PROJECT = "EmoNet"
VERSION = "0.1.0"
LICENSE = "GPLv3+"
AUTHOR = "Maurice Gerczuk"
AUTHOR_EMAIL = "maurice.gerczuk@informatik.uni-augsburg.de"
URL = 'https://github.com/EIHW/EmoNet'

install_requires = [
    "Click",
    "dill",
    "imbalanced-learn",
    "librosa",
    "numpy",
    "pandas",
    "Pillow",
    "numba==0.48.*",
    "scikit-learn==0.22",
    "tensorflow==2.5.1",
    "tensorboard==2.5",
    "tqdm"
]

tests_require = ['pytest>=4.4.1', 'pytest-cov>=2.7.1']
needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
setup_requires = ['pytest-runner'] if needs_pytest else []
packages = find_packages('src')

setup(
    name=PROJECT,
    version=VERSION,
    license=LICENSE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description="**EmoNet** is a Python toolkit for multi-corpus speech emotion recognition and other audio classification tasks.",
    platforms=["Any"],
    scripts=[],
    provides=[],
    python_requires="~=3.8.0",
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    namespace_packages=[],
    packages=packages,
    package_dir={'': 'src'},
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "EmoNet = emonet.cli:cli",
        ]
    },
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        'Environment :: GPU :: NVIDIA CUDA :: 11.0',
        # Indicate who your project is intended for
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',

        'Programming Language :: Python :: 3.8',
    ],
    keywords='machine-learning audio-analysis science research',
    project_urls={
        'Source': 'https://github.com/EIHW/EmoNet',
        'Tracker': 'https://github.com/EIHW/EmoNet/issues',
    },
    url=URL,
    zip_safe=False,
)