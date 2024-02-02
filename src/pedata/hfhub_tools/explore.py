import os
from pathlib import Path
import shutil
from huggingface_hub import DatasetFilter, list_datasets
from datasets import (
    DownloadConfig,
    get_dataset_infos,
)
from dataclasses import dataclass
import argparse

# setting setting the __package__ attribute to solve the relative import proplem when runningn the scripts in the command line
__package__ = "pedata.hfhub_tools"


# ==== dataset exploration ====
@dataclass
class TagFeaturePattern(object):
    """Correpondings between a tag and a feature pattern and whether all patterns must be present or just one of them"""

    tag: str  # tag to be added to the dataset
    feature_pattern: list[str]  # list of patterns to be searched in the features
    all_patterns: bool = (
        False  # whether all patterns must be present or just one of them
    )


dstype_feature_patterns = [
    TagFeaturePattern(
        tag="_MOL",
        feature_pattern=["bnd_type", "atm_type", "bnd_idcs"],
        all_patterns=True,
    ),
    TagFeaturePattern(tag="_PE", feature_pattern=["aa_"]),
    TagFeaturePattern(tag="_DNA", feature_pattern=["dna_"]),
]

target_tags_patterns = [
    TagFeaturePattern(tag="_SOL", feature_pattern=["solubility"]),
    TagFeaturePattern(
        tag="at_least_one_feature_named_TARGET", feature_pattern=["target"]
    ),
]


def get_tag_list(
    tag_pattern: list[TagFeaturePattern], features: list[str]
) -> list[str]:
    """Get a list of tags based on the features of a dataset"""
    tag_list = []
    # going over the tag patterns
    for tp in tag_pattern:
        # go over the patterns in tp.feature_pattern and store the patterns present in the feature list
        feature_pattern_in_ds = []
        for pattern in tp.feature_pattern:
            if any([pattern in feature for feature in features]):
                feature_pattern_in_ds.append(pattern)

        # check if the tag should be added to the list
        # tags that should all be present
        if tp.all_patterns and len(feature_pattern_in_ds) == len(tp.feature_pattern):
            tag_list.append(tp.tag)

        # only one tag present
        elif not tp.all_patterns and len(feature_pattern_in_ds) > 0:
            tag_list.append(tp.tag)

        # no tag present
        else:
            pass

    return tag_list


def save_log(filename, log_data):
    with open(Path(filename), "w") as file:
        file.write("\n".join(log_data))
    print(f"Log saved to {filename}")


def explore_datasets(
    working_dir: str,
    log_filename: str,
    author: str = "Company",
    delete_cache: bool = False,
    just_testing: bool = False,
) -> None:
    """Explore all datasets from a given author and save the results to a log file
    Args:
        working_dir: directory to store the log file
        log_filename: name of the log file
        author: author to explore
        cache_dir: directory to store the cache
        delete_cache: whether to delete the cache after exploration
        just_testing: whether to just test the function
    Returns:
        None
    """
    # make exploration directory
    working_dir = "./huggingface_datasets_info"
    os.makedirs(working_dir, exist_ok=True)
    # cache directory
    cache_dir = Path(f"{working_dir}/.cache")
    # Configure download with a cache directory
    conf = DownloadConfig(cache_dir=Path(cache_dir))
    # List all datasets
    datasets = list_datasets(filter=DatasetFilter(author=author))
    nb_datasets = len(list(list_datasets(filter=DatasetFilter(author=author))))
    # Initialize lists to store data for logs
    log_csv = [
        "dataset_id, tags, target_tags, split_info, fullsize, last_modified, features"
    ]

    # Loop over datasets
    for index, dataset in enumerate(datasets):
        print(
            "---------------------------------------- \n"
            f"{index} of {nb_datasets} - {dataset.id} \n"
        )
        ds_info = get_dataset_infos(dataset.id, download_config=conf)
        features = list(ds_info[list(ds_info.keys())[0]].features)

        # get tags
        ds_tag_list = get_tag_list(dstype_feature_patterns, features)
        target_tag_list = get_tag_list(target_tags_patterns, features)
        last_modified = dataset.last_modified.strftime("%Y-%m-%d_%H:%M:%S")

        datasize_info = ""
        full_size = 0
        for split, values in ds_info[list(ds_info.keys())[0]].splits.items():
            if hasattr(values, "num_examples"):
                datasize_info = (
                    f"{datasize_info} split: {split}: {values.num_examples} examples, "
                )
                full_size += values.num_examples
            else:
                datasize_info = f"{datasize_info} split: {split}: ERROR, "
                full_size = "ERROR"

        datasize_info = f" {datasize_info}"

        if just_testing and index > 2:
            break

        # Print info - gives an idea of progress
        print(
            f"Tags: {ds_tag_list} \n"
            f"Target tags: {target_tag_list} \n"
            f"Features: {features} \n"
            f"Last modified: {last_modified} \n"
            f"Split Info: {datasize_info} \n"
            f"Full size: {full_size} \n"
        )

        # Format for CSV log
        log_csv.append(
            f'"{dataset.id}","{", ".join(ds_tag_list)}","{", ".join(target_tag_list)}","{datasize_info}", {full_size}, "{last_modified}","{", ".join(features)}"'
        )

    # Save logs to files
    save_log(Path(f"{working_dir}/{log_filename}.csv"), log_csv)

    if delete_cache:
        shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(
        description="Explore all datasets from a given author and save the results to a log file"
    )
    # required arguments
    parser.add_argument(
        "--working_dir",
        required=False,
        help="str: the working directory to save the log file to.",
        default="./huggingface_datasets_info",
    )
    parser.add_argument(
        "--log_filename",
        required=False,
        help="str: the name of the log file to save the log to.",
        default="exploration_log",
    )
    # optional arguments
    parser.add_argument(
        "--author",
        required=False,
        help="str: The author of the dataset to pull from Hugging Face.",
        default="Company",
    )
    parser.add_argument(
        "--delete_cache",
        action="store_true",
        help="bool: Whether to delete the cache after exploration.",
    )
    parser.add_argument(
        "--just_testing",
        action="store_true",
        help="bool: Whether to just test the function.",
    )

    args = parser.parse_args()
    print(args)
    # create dataset upload object
    data_upload = explore_datasets(
        args.working_dir,
        args.log_filename,
        author=args.author,
        delete_cache=args.delete_cache,
        just_testing=args.just_testing,
    )
