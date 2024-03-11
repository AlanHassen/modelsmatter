# Models Matter

This package extends [AiZynthFinder](https://github.com/MolecularAI/aizynthfinder) to seamlessly integrate various single-step retrosynthesis models, as illustrated in our papers [Models Matter: The Impact of Single-Step Models on Synthesis Prediction](https://doi.org/10.1039/D3DD00252G) and [Mind the Retrosynthesis Gap: Bridging the divide between Single-step and Multi-step Retrosynthesis Prediction](https://openreview.net/forum?id=LjdtY0hM7tf)

## Overview

**Models Matter** uses the default AiZynthFinder implementation and introduces the following enhancements:

- **Smiles-based Expansion Strategy**: This feature allows the integration of any retrosynthesis model operating at smiles-level with AiZynthFinder, not depending on the underlying search algorithm.
- **ModelZoo Integration**: The package includes [ModelZoo](https://github.com/PTorrenPeraire/modelsmatter_modelzoo), a dedicated framework that defines the single-step retrosynthesis approach within the smiles-based expansion strategy. Currently supported implementations are [Chemformer](https://github.com/PTorrenPeraire/aidd_chemformer), [MHNreact](https://github.com/PTorrenPeraire/aidd_mhn_react), and [LocalRetro](https://github.com/AlanHassen/modelsmatter_localretro_hpc).

## Installation Procedure

To install the package, follow the sequential steps below:

```bash
# Clone this repository and all submodules
git clone --recursive https://github.com/AlanHassen/modelsmatter

# Create and initialize a conda environment
conda env create -f environments/environment.yml -n ssbenchmark
conda activate ssbenchmark

# Transition to the SSBenchmark directory
cd external/modelsmatter_modelzoo/

# First the installation of the ModelZoo
poetry install

# Navigate back to the models matter directory
cd ../..

# Finalize the installation process
poetry install

# Subsequently, install the appropriate single-step models or the necessary libraries.
```

## Usage Guidelines

Adaptations to AiZynthFinder config are necessary to accommodate the different single-step models (examples provided in `config/`). The configurable settings include:

- **gpu_mode**: Enable GPU mode for accelerated inference.
  
- **module_path**: The path of the single-step retrosynthesis model repository.
  
- **model_path**: The location of the trained model.

- Additional parameters can be set, such as specifying the vocabulary for Chemformer via `vocab_path`.

Utilize the extended functionalities of AiZynthFinder through Models Matter for a comprehensive synthesis prediction experience.

## Datasets & Models

Datasets and models that are not proprietary will be accessible upon publication.
