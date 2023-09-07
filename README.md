# Models Matter

Package to extend AiZynthFinder(https://github.com/MolecularAI/aizynthfinder) to interchangebly use different single-step retrosynthesis models as shown in Models Matter: The Impact of Single-Step Models on Synthesis Prediction (https://arxiv.org/abs/2308.05522). 

## Overview

This package is based on AiZynthFinder, relevant changes are detailed below. To interchangeably use alternate single-step models it relies on [SingleStepModelZoo] which is included as default. With this package it is possible to set up and use any single-step model in combination with any of the multi-step search algorithmns. 

# Install:

conda env create -f environments/environment.yml -n ssbenchmark
conda activate ssbenchmark
cd external/SSBenchmark/
poetry install
cd ../..
poetry install

# note: for localretro you have to:
unset LD_LIBRARY_PATH


# AiZynthFinder 

[![License](https://img.shields.io/github/license/MolecularAI/aizynthfinder)](https://github.com/MolecularAI/aizynthfinder/blob/master/LICENSE)
[![Tests](https://github.com/MolecularAI/aizynthfinder/workflows/tests/badge.svg)](https://github.com/MolecularAI/aizynthfinder/actions?workflow=tests)
[![codecov](https://codecov.io/gh/MolecularAI/aizynthfinder/branch/master/graph/badge.svg)](https://codecov.io/gh/MolecularAI/aizynthfinder)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black) 
[![version](https://img.shields.io/github/v/release/MolecularAI/aizynthfinder)](https://github.com/MolecularAI/aizynthfinder/releases)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MolecularAI/aizynthfinder/blob/master/contrib/notebook.ipynb)


AiZynthFinder is a tool for retrosynthetic planning. The algorithm is based on a Monte Carlo tree search that recursively breaks down a molecule to purchasable precursors. The tree search is guided by a policy that suggests possible precursors by utilizing a neural network trained on a library of known reaction templates.

An introduction video can be found here: [https://youtu.be/r9Dsxm-mcgA](https://youtu.be/r9Dsxm-mcgA)

## Prerequisites

Before you begin, ensure you have met the following requirements:

* Linux, Windows or macOS platforms are supported - as long as the dependencies are supported on these platforms.

* You have installed [anaconda](https://www.anaconda.com/) or [miniconda](https://docs.conda.io/en/latest/miniconda.html) with python 3.6 - 3.9

The tool has been developed on a Linux platform, but the software has been tested on Windows 10 and macOS Catalina.


## Installation

### For end-users

First time, execute the following command in a console or an Anaconda prompt

    conda env create -f https://raw.githubusercontent.com/MolecularAI/aizynthfinder/master/env-users.yml
    
And if you want to update the environment

    conda env update -n aizynth-env -f https://raw.githubusercontent.com/MolecularAI/aizynthfinder/master/env-users.yml
    
The package is now installed in a new conda environment, that you need to activate each time you want to use it

    conda activate aizynth-env

### For developers

First clone the repository using Git.

Then execute the following commands in the root of the repository 

    conda env create -f env-dev.yml
    conda activate aizynth-dev
    poetry install
    
the `aizynthfinder` package is now installed in editable mode.

### Troubleshooting

If the above simple instructions does not work, here are the more detailed instructions. You might have to modify conda channels or similar if the dependencies fails to install on your OS.

First, install these conda packages

    conda install -c conda-forge "rdkit=>2019.09.1" -y
    conda install graphviz -y

Secondly, install the ``aizynthfinder`` package

    python -m pip install https://github.com/MolecularAI/aizynthfinder/archive/v3.4.0.tar.gz


if you want to install the latest version

or, if you have cloned this repository
 
    conda install poetry
    python poetry


> Note on the graphviz installation: this package does not depend on any third-party python interfaces to graphviz but instead calls the `dot` executable directly. If the executable is not in the `$PATH` environmental variable, the generation of route images will not work. If unable to install it properly with the default conda channel, try using `-c anaconda`.


## Usage

The tool will install the ``aizynthcli`` and ``aizynthapp`` tools
as interfaces to the algorithm:

```
aizynthcli --config config.yml --smiles smiles.txt
aizynthapp --config config.yml
```

Consult the documentation [here](https://molecularai.github.io/aizynthfinder/) for more information.

To use the tool you need

    1. A stock file
    2. A trained rollout policy network (including the Keras model and the list of unique templates)
    3. A trained filer policy network (optional)

Such files can be downloaded from [figshare](https://figshare.com/articles/AiZynthFinder_a_fast_robust_and_flexible_open-source_software_for_retrosynthetic_planning/12334577) and [here](https://figshare.com/articles/dataset/A_quick_policy_to_filter_reactions_based_on_feasibility_in_AI-guided_retrosynthetic_planning/13280507) or they can be downloaded automatically using

```
download_public_data my_folder
```

where ``my_folder`` is the folder that you want download to.
This will create a ``config.yml`` file that you can use with either ``aizynthcli`` or ``aizynthapp``.

## Development

### Testing

Tests uses the ``pytest`` package, and is installed by `poetry`

Run the tests using:

    pytest -v

The full command run on the CI server is available through an `invoke` command

    invoke full-tests
    
 ### Documentation generation

The documentation is generated by Sphinx from hand-written tutorials and docstrings

The HTML documentation can be generated by

    invoke build-docs

## Contributing

We welcome contributions, in the form of issues or pull requests.

If you have a question or want to report a bug, please submit an issue.


To contribute with code to the project, follow these steps:

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the remote branch: `git push`
5. Create the pull request.

Please use ``black`` package for formatting, and follow ``pep8`` style guide.


## Contributors

* [@SGenheden](https://www.github.com/SGenheden)
* [@EBjerrum](https://www.github.com/EBjerrum)
* [@A-Thakkar](https://www.github.com/A-Thakkar)

The contributors have limited time for support questions, but please do not hesitate to submit an issue (see above).

## License

The software is licensed under the MIT license (see LICENSE file), and is free and provided as-is.

## References

1. Thakkar A, Kogej T, Reymond J-L, et al (2019) Datasets and their influence on the development of computer assisted synthesis planning tools in the pharmaceutical domain. Chem Sci. https://doi.org/10.1039/C9SC04944D
2. Genheden S, Thakkar A, Chadimova V, et al (2020) AiZynthFinder: a fast, robust and flexible open-source software for retrosynthetic planning. J. Cheminf. https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00472-1
3. Genheden S, Engkvist O, Bjerrum E (2020) A Quick Policy to Filter Reactions Based on Feasibility in AI-Guided Retrosynthetic Planning. ChemRxiv. Preprint. https://doi.org/10.26434/chemrxiv.13280495.v1 
4. Genheden S, Engkvist O, Bjerrum E (2021) Clustering of synthetic routes using tree edit distance. J. Chem. Inf. Model. 61:3899–3907 [https://doi.org/10.1021/acs.jcim.1c00232](https://doi.org/10.1021/acs.jcim.1c00232)
5. Genheden S, Engkvist O, Bjerrum E (2022) Fast prediction of distances between synthetic routes with deep learning. Mach. Learn. Sci. Technol. 3:015018 [https://doi.org/10.1088/2632-2153/ac4a91](https://doi.org/10.1088/2632-2153/ac4a91) 
