""" Module routines for pre-processing data for expansion policy training
"""
import argparse
import os
from typing import Sequence, Optional


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from scipy import sparse

from aizynthfinder.training.utils import (
    Config,
    save_data_with_defined_split,
    smiles_to_fingerprint,
    is_sanitizable,
    split_and_save_data,
    split_reaction_smiles,
)


def _filter_dataset(config: Config) -> pd.DataFrame:

    filename = config.filename("raw_library")
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"The file {filename} is missing - cannot proceed without the full template library."
        )

    # Skipping the last header as it is not available in the raw data
    full_data = pd.read_csv(
        filename,
        index_col=False,
        header=0 if config["in_csv_headers"] else None,
        names=None if config["in_csv_headers"] else config["library_headers"][:-1],
        sep=config["csv_sep"],
    )

    if config["reaction_smiles_column"]:
        full_data = split_reaction_smiles(full_data, config)

    if config["remove_unsanitizable_products"]:
        products = full_data[config["column_map"]["products"]].to_numpy()
        idx = np.apply_along_axis(is_sanitizable, 0, [products])
        full_data = full_data[idx]

    template_hash_col = config["column_map"]["template_hash"]
    full_data = full_data.drop_duplicates(subset=config["column_map"]["reaction_hash"])
    template_group = full_data.groupby(template_hash_col)
    template_group = template_group.size().sort_values(ascending=False)
    min_index = template_group[template_group >= config["template_occurrence"]].index
    dataset = full_data[full_data[template_hash_col].isin(min_index)]

    template_labels = LabelEncoder()
    dataset = dataset.assign(
        template_code=template_labels.fit_transform(dataset[template_hash_col])
    )
    dataset.to_csv(
        config.filename("library"),
        mode="w",
        header=config["in_csv_headers"],
        index=False,
        sep=config["csv_sep"],
    )
    return dataset


def _get_config(optional_args: Optional[Sequence[str]] = None) -> Config:
    parser = argparse.ArgumentParser(
        "Tool to pre-process a template library to be used in training a expansion network policy"
    )
    parser.add_argument("config", help="the filename to a configuration file")
    args = parser.parse_args(optional_args)

    return Config(args.config)


def _save_unique_templates(dataset: pd.DataFrame, config: Config) -> None:
    template_hash_col = config["column_map"]["template_hash"]

    if dataset.empty:
        raise ValueError("There are no train examples in the split column")

    template_group = dataset.groupby(template_hash_col, sort=False).size()
    dataset = dataset[
        [config["column_map"]["retro_template"], "template_code"]
        + config["metadata_headers"]
    ]
    if "classification" in dataset.columns:
        dataset["classification"].fillna("-", inplace=True)
    dataset = dataset.drop_duplicates(subset="template_code", keep="first")
    dataset["library_occurrence"] = template_group.values
    dataset.set_index("template_code", inplace=True)
    dataset = dataset.sort_index()
    dataset.rename(
        columns={
            template_hash_col: "template_hash",
            config["column_map"]["retro_template"]: "retro_template",
        },
        inplace=True,
    )
    print(f"Length of unique templates: {len(dataset)}", flush=True)
    dataset.to_hdf(config.filename("unique_templates"), "table")


def main(optional_args: Optional[Sequence[str]] = None) -> None:
    """Entry-point for the preprocess_expansion tool"""
    config = _get_config(optional_args)
    if config["library_headers"][-1] != "template_code":
        config["library_headers"].append("template_code")

    filename = config.filename("library")
    if not os.path.exists(filename):
        dataset = _filter_dataset(config)
    else:
        dataset = pd.read_csv(
            filename,
            index_col=False,
            header=0 if config["in_csv_headers"] else None,
            names=None if config["in_csv_headers"] else config["library_headers"],
            sep=config["csv_sep"],
        )
        if config["reaction_smiles_column"]:
            dataset = split_reaction_smiles(dataset, config)

    # assert that the dataset is has a split column
    if "split" not in dataset.columns:
        raise ValueError("The dataset must have a split column with train, valid, test")

    ########### training
    train  = dataset[dataset["split"] == "train"]
    print("Training Data:", len(train), flush=True)

    print("Dataset filtered/loaded, generating labels...", flush=True)
    labelb = LabelBinarizer(neg_label=0, pos_label=1, sparse_output=True)
    labelb = labelb.fit(train[config["column_map"]["template_hash"]])
    labels = labelb.transform(train[config["column_map"]["template_hash"]])
    save_data_with_defined_split(labels, "labels", config, "training_")

    print("Labels created and split, generating inputs...", flush=True)
    products = train[config["column_map"]["products"]].to_numpy()
    inputs = np.apply_along_axis(smiles_to_fingerprint, 0, [products], config)
    inputs = sparse.lil_matrix(inputs.T).tocsr()
    save_data_with_defined_split(inputs, "inputs", config, "training_")

    print("Inputs created and split, splitting full Dataset...", flush=True)
    save_data_with_defined_split(train, "library", config, "training_")

    ########### Validation
    valid = dataset[dataset["split"] == "valid"]
    print("Validation Data:", len(valid), flush=True)

    # remove all templates that are not in the training set
    in_train_and_valid = valid[config["column_map"]["template_hash"]].isin(train[config["column_map"]["template_hash"]])
    valid = valid[in_train_and_valid]

    print("Validation Data after filtering templates out not in training:", len(valid), flush=True)

    print("Dataset filtered/loaded, generating labels...", flush=True)
    labels = labelb.transform(valid[config["column_map"]["template_hash"]])
    save_data_with_defined_split(labels, "labels", config, "validation_")

    print("Labels created and split, generating inputs...", flush=True)
    products = valid[config["column_map"]["products"]].to_numpy()
    inputs = np.apply_along_axis(smiles_to_fingerprint, 0, [products], config)
    inputs = sparse.lil_matrix(inputs.T).tocsr()
    save_data_with_defined_split(inputs, "inputs", config, "validation_")

    print("Inputs created and split, splitting full Dataset...", flush=True)
    save_data_with_defined_split(valid, "library", config, "validation_")


    # create templates --> does inline stuff
    print("Full Dataset split, creating unique template set", flush=True)
    _save_unique_templates(train, config)



if __name__ == "__main__":
    main()
