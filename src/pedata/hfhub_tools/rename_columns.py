# setting setting the __package__ attribute to solve the relative import proplem when runningn the scripts in the command line
__package__ = "pedata.hfhub_tools"

# imports
import argparse
import json
import shutil
from pathlib import Path
from datasets import load_dataset, DownloadConfig


# method
def rename_hub_dataset_column(repo_id: str, column_name_mapping: dict[str:str]) -> None:
    """Rename a column in a dataset on the hub
    Args:
        repo_id: The id of the repo to rename the column in
        column_name_mapping: A dictionary with the old column name as key and the new column name as value
    Notes:
        This function pulls from the hub, renames the column and pushes it back to the hub
    Examples:
        THIS EXAMPLE IS NOT WRITTEN LIKE THIS BECAUSE SHOULD NOT BE RUN AS A DOCTEST - BUT IT WORKS
        from datasets import load_dataset
        from pedata.preprocessing.upload import rename_hub_dataset_column
        rename_hub_dataset_column(
            repo_id="Company/test_example_dataset_ha1",
            column_name_mapping={
                "target kcat per kmol": "target_kcat_per_kmol",
                "aa_unirep_1900": "aa unirep 1900",
            },
        )
    """
    # set cache dir and download config
    cache_directory = Path(f".cache/{repo_id}")
    shutil.rmtree(cache_directory, ignore_errors=True)  # delete cache
    dll_conf = DownloadConfig(cache_dir=cache_directory, force_download=True)

    # pull dataset from hub
    dataset = load_dataset(repo_id, download_config=dll_conf, cache_dir=cache_directory)

    # update column names
    for name, new_name in column_name_mapping.items():
        dataset = dataset.rename_column(name, new_name)

    # push to hub
    dataset.push_to_hub(repo_id)

    shutil.rmtree(cache_directory, ignore_errors=True)  # delete cache


def validate_dataset_columns_names():
    # TODO: implement and use while creating dataset in upload.py and preprocess dataset
    """Check that the column names do not contain any forbidden characters"""
    forbidden_characters = [" ", "/", "."]
    pass


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(
        description=(
            """Rename a column in a dataset on the hub python '
            rename_columns.py --repo "Company/test_example_dataset_ha1" --col_mapping '{"target kcat per kmol": "target_kcat_per_kmol", "aa_unirep_1900": "aa unirep 1900"}'"""
        )
    )
    # required arguments
    parser.add_argument(
        "--repo",
        required=True,
        help="Name of the repository to pull from on Hugging Face.",
    )
    parser.add_argument(
        "--col_mapping",
        required=True,
        help="JSON string representing a dictionary with old column name as key and new column name as value",
    )
    args = parser.parse_args()

    # parse JSON string to dictionary
    col_mapping = json.loads(args.col_mapping)

    # create dataset upload object
    rename_hub_dataset_column(args.repo, col_mapping)

    # Example usage:
    # python src/pedata/Company_datasets/rename_columns.py --repo "Company/test_example_dataset_ha1" --col_mapping '{"target kcat per kmol": "target_kcat_per_kmol", "aa_unirep_1900": "aa unirep 1900"}'
