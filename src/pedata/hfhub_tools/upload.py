""" #FIXME: THIS NEED FULL REFACTORING
Module upload.py

Usage from the command line:

    Required argument:
        --repo, Name of the repository on Hugging Face
        
    None required arguments:
        --commit_hash, Commit hash of the dataset to pull from Hugging Face. default=None,
        --filename, Path to the CSV file for dataset creation. default=None,
        --local_dir, Name of the local directory to save the dataset to. default="./local_datasets",
        --cache_dir, cache directory; default: "./cache"
        --save_locally, Name of the local directory to save the dataset to. default=True,
        --splits_to_combine_as_whole_ds, list: names of the split to combine as 'whole_dataset' ex --splits_to_combine_as_whole_ds whole_dataset
        --update_just_readme, When update - just update the readme or dataset card. default=False,
        --needed_encodings, list: list of encodings for the dataset; ex --needed_encodings aa_seq aa_1hot
        --overwrite_repo, bool: Set to True to overwrite a repo when uploading a new dataset, if the repo already exists


Examples: 
    Uploading a new dataset from a csv file to a new repo
    ```bash
    "python src/pedata/Company_datasets/upload.py --repo Company/test_example_dataset_ha1 --filename local_datasets/datafiles/example_dataset_ha1.csv --needed_encodings 'aa_seq' 'aa_1hot' 'aa_unirep_1900'
    ```

    Overwriting a new repo by uploading a new dataset made from a csv file
    ```bash
    python src/pedata/Company_datasets/upload.py ... same as above ... --overwrite_repo True "
    ```

    Pulling a dataset from huggingface, updating it and uploading it uploading to the same repo
    ```bash
    "python src/pedata/Company_datasets/upload.py --repo Company/test_example_dataset_ha1"

```
"""
# FIXME: refactor this as a upload and update class -
# create a pipeline which includes
# - checking the dataset,
# - adding encodigns,
# - adding index,
# - adding splits
# - execute tag_finder
# - save the data locally
# - (delete the data from the hub - only if update)
# - push the data to the hub
# - update the readme -> check the function from huggingface to update the metadata
# - push the readme to the hub

# setting setting the __package__ attribute to solve the relative import proplem when runningn the scripts in the command line
__package__ = "pedata.hfhub_tools"
import argparse
import os
import shutil
from pathlib import Path
import numpy as np
from typing import Sequence
from datasets import load_dataset, Dataset, concatenate_datasets
from huggingface_hub import (
    repo_exists,
    list_repo_files,
    delete_file,
    metadata_update,
)
from ..transform import transform_pipeline
from ..disk_cache import preprocess_data
from ..util import get_target
from ..visual import plot_target_distributions
from . import ReadMe


class DatasetUpload:  # FIXME: change the name of this class
    """A class to handle dataset creation, update and upload to Hugging Face Hub."""

    def __init__(
        self,
        repo: str,
        commit_hash: str = None,
        local_dir: str | Path = "./local_datasets",
        cache_dir: str | Path = "./cache",
        csv_filename: str | Path = None,
        save_locally: bool = True,
        splits_to_combine_as_whole_ds: list = [],
        update_just_readme: bool = False,
        needed_encodings: list[str] = [],
        overwrite_repo: bool = False,
    ):
        """Initialize the class and run the creation, update and upload pipeline.
        Args:
            repo (str): Hugging Face Hub repository name (format: 'Company/dataset-name').
            local_dir (str): Local directory to save the dataset to.
            cache_dir (str): cache directory
            csv_filename (str): Path to the CSV file for dataset creation.
                if None, the dataset is pulled from Hugging Face Hub and updated. - default: None
            save_locally (bool): Whether to save the dataset to a local directory. - default: True
            splits_to_combine_as_whole_ds: The name of the splits to combine as the whole dataset
                - when updating a dataset which is already on the hub. - default: []
            update_just_readme (bool): When updating a dataset which is already on the hub,
                update just the readme or dataset card. - default: False
            needed_encodings (list): list of encodings for the dataset; default: []
            overwrite_repo (bool): Set to True to overwrite a repo when uploading a new dataset,
                if the repo already exists - default: False

        """

        self._repo = repo
        self._commit_hash = str(commit_hash) if commit_hash is not None else None
        self._local_dir = Path(local_dir) if isinstance(local_dir, str) else local_dir
        self._cache_dir = Path(cache_dir) if isinstance(cache_dir, str) else cache_dir
        self._csv_filename = (
            Path(csv_filename) if isinstance(csv_filename, str) else csv_filename
        )
        self._save_locally = save_locally
        self._whole_split_name = "whole_dataset"
        self._splits_to_combine_as_whole_ds = splits_to_combine_as_whole_ds
        self._update_just_readme = update_just_readme
        self._needed_encodings = needed_encodings  # TODO - add a setter for this
        self._init_process(overwrite_repo)

    def _init_process(self, overwrite_repo: bool = False):
        if self._csv_filename is not None:
            if repo_exists(self._repo, repo_type="dataset") and not overwrite_repo:
                raise Exception(
                    f"repo {self._repo} already exists \n"
                    "Please choose another name or set overwrite_repo=True to overwrite the content of the repo"
                )
            self._dataset = self.create_and_preprocess(
                self._csv_filename, self._needed_encodings
            )
        else:
            self._dataset = self.pull_and_update(
                self._repo,
                commit_hash=self._commit_hash,
                whole_split_name=self._whole_split_name,
                splits_to_combine_as_whole_ds=self._splits_to_combine_as_whole_ds,
                update_just_readme=self._update_just_readme,
                needed_encodings=self._needed_encodings,
                cache_dir=self._cache_dir,
            )

    @property
    def local_path(self) -> str:
        return os.path.join(self._local_dir, self._repo)

    @property
    def figure_path(self) -> str:
        return os.path.join(self.local_path, "figures")

    @property
    def cache_dir(self) -> str:
        return self._cache_dir

    def process(self, verbose: bool = True, readme: bool = False) -> None:
        """Run the creation, update and upload pipeline.
        FIXME - from Ingmar: the usage of this class I've always seen only creation of an object directly followed by a call to this method. If this is the only usage then a simple function definition / function call would be better.
        If the class definition makes implementation better/easier to understand, then this would only mean changing the API, e.g. by introducing a small wrapper function
        """
        if verbose:
            print(self.__repr__())

        # create local directory
        if not os.path.exists(self.local_path):
            os.makedirs(self.local_path)

        if readme:
            # create the figures directory
            os.makedirs(f"{self.local_path}/figures", exist_ok=True)

        # save
        if self._save_locally:
            self.save(self._dataset, self.local_path)

        # cleanup hub before pushing (only when updating)
        if self._csv_filename is None:
            # clear the datafiles in the repo
            self._clear_hub_dataset_files()

            # update the metadata in the dataset card with nothing in it
            metadata_update(
                repo_id=self._repo,
                metadata={"dataset_info": "nothing in it", "configs": []},
                repo_type="dataset",
                overwrite=True,
            )

        # clear the cache before pushing
        shutil.rmtree(self._cache_dir, ignore_errors=True)

        # push
        self.push(self._dataset, repo=self._repo, split=self._whole_split_name)

        if verbose:
            print(self.__repr__())

        if readme:
            self.update_readme()

        # clear the cache
        shutil.rmtree(self._cache_dir, ignore_errors=True)

    def make_read_me_figures(self) -> None:
        plot_target_distributions(self.targets, savedir=self.local_path)

    def update_readme(self) -> None:
        """Update the readme file"""
        readme = ReadMe(local_dir=self.local_path)
        readme.pull_readme_from_hub(repo_id=self._repo)

        readme.update_readme(self.datapoints_n, "datapoints")
        readme.update_readme(self.feature_names, "embeddings")
        readme.update_readme(self.target_names, "targets")
        readme.update_readme(self.available_splits, "available_splits")

        self.make_read_me_figures()

        readme.update_readme_figures(figure_path=self.figure_path)
        readme.push_figures_to_hub()
        readme.push_readme_to_hub(repo_id=self._repo)

    def __repr__(self) -> None:
        """Print the processing to be done or the processing done."""

        def print_list(l):
            return "\n".join([f" - {item}" for item in l])

        if "_dataset" not in self.__dict__:
            return f"""
------------------------------------
DatasetUpload - Processing to be done
------------------------------------
- repo={self._repo} 
- local_dir={self._local_dir}
- csv_filename={self._csv_filename}
- save_locally={self._save_locally}
            """
        else:
            return f"""
-------------------------------
DatasetUpload - Processing done
-------------------------------
Saved locally: {self.local_path}
Pushed to the huggingface repository: {self._repo}
Available features:
{print_list(self.feature_names)}
Available targets: 
{print_list(self.target_names)}
Available splits:
{print_list(self.available_splits)}
            """

    @property
    def target_names(self) -> list[str]:
        """get all target names"""
        targets = get_target(self._dataset, as_dict=True)
        return list(targets.keys())

    @property
    def targets(self) -> dict[Sequence[str], np.ndarray]:
        """Getting all targets
        Returns:
            Dictionary of targets with the target names as keys and the target values as values
        """
        return get_target(self._dataset, as_dict=True)

    @property
    def available_splits(self) -> list[str]:
        """get all available splits"""
        return [col for col in self._dataset.column_names if "split" in col]

    @property
    def feature_names(self) -> list[str]:
        """get all features names"""
        return [
            col
            for col in self._dataset.column_names
            if not (
                col in self.target_names
                or col in self.available_splits
                or col in ["index"]
            )
        ]

    @property
    def datapoints_n(self) -> list[int]:
        """get the number of datapoints"""
        return [self._dataset.num_rows]

    @staticmethod
    def create_and_preprocess(
        csv_filename: str | Path, needed_encodings: list[str]
    ) -> Dataset:
        """Create a dataset from a CSV file and push it to Hugging Face.
        Args:
            filename (str): Path to the CSV file.
            needed_encodings: list of encodings
        Returns:
            ds.Dataset: Dataset created from the CSV file.
        """
        # Convert CSV to Hugging Face dataset
        return preprocess_data(
            csv_filename,
            add_index=True,
            add_splits=True,
            needed_encodings=needed_encodings,
        )

        # print(f"Dataset '{repo}' successfully pushed to Hugging Face.")

    @staticmethod
    def pull_and_update(
        repo: str,
        commit_hash: str = None,
        whole_split_name: str = "whole_dataset",
        splits_to_combine_as_whole_ds: list = [],
        update_just_readme: bool = False,
        needed_encodings: list[str] = [],
        cache_dir="./cache",
    ) -> Dataset:
        """Pull a dataset from Hugging Face, update it.
        Args:
            repo (str): Hugging Face Hub repository name (format: 'Company/dataset-name').
        returns:
            ds.Dataset: Dataset pulled from Hugging Face.
        """
        # Pull dataset from Hugging Face
        # concatenate_datasets makes sure that the dataset is a dataset and not a dataset dictionary (which is the case when pulling from the hub)
        dataset_dict = load_dataset(
            f"{repo}",
            download_mode="force_redownload",
            cache_dir=Path(cache_dir),
            revision=commit_hash,
        )
        splits_already_in_dataset = list(dataset_dict.keys())
        # return the dataset in the a dataset dictionary with the whole dataset as one split named 'whole_dataset'

        if len(dataset_dict) > 1 and whole_split_name not in splits_already_in_dataset:
            raise ValueError(
                f"DatasetDict has more than one split and does not have a split named {whole_split_name}."
                "Use splits_to_combine_as_whole_ds as argument to specify which splits to combine."
            )

        # if splits_to_combine_as_whole_ds and there is only one split in the dataset
        if splits_to_combine_as_whole_ds == [] and len(dataset_dict) == 1:
            splits_to_combine_as_whole_ds = splits_already_in_dataset

        # convert the DatasetDict to a Dataset
        dataset = concatenate_datasets(
            [dataset_dict[ds] for ds in splits_to_combine_as_whole_ds]
        )

        if update_just_readme:
            return dataset
        else:
            return transform_pipeline(dataset, needed_encodings)

    @property
    def files_in_repo(self) -> list[str]:
        """Get all files in a HuggingFace dataset repository folder.
        Args:
            repo : Hugging Face Hub repository name (format: 'Company/dataset-name').
        Returns:
            list: List of files in the data/ folder of the repository.
        """
        return list_repo_files(repo_id=self._repo, repo_type="dataset")

    def _get_hub_data_folder_files(self) -> list[str]:
        """Get all files in a HuggingFace dataset repository data/ folder.
        Args:
            repo : Hugging Face Hub repository name (format: 'Company/dataset-name').
        Returns:
            list: List of files in the data/ folder of the repository.
        """
        return [file for file in self.files_in_repo if "data/" in file]

    def _repo_get_file_list_split_cleanup(self) -> list[str]:
        """Delete all files in a HuggingFace dataset repository.
        Checks the difference between the list of files in the repo and the list of files in the dataset directory using the load_dataset methods.
        Args:
            repo : Hugging Face Hub repository name (format: 'Company/dataset-name').
        """
        dataset_files = self._get_hub_data_folder_files()
        if dataset_files == []:
            return []

        dataset_dict_keys = list(load_dataset(f"{self._repo}").keys())
        list_of_files_to_delete = []  # FIXME write this as a list comprehension
        for dataset_dict_key in dataset_dict_keys:
            for dataset_file in dataset_files:
                if dataset_dict_key in dataset_file:
                    list_of_files_to_delete.append(dataset_file)

        return list_of_files_to_delete

    def _clear_hub_dataset_files(self, list_of_files_to_delete: list = []) -> None:
        """Delete all files in a HuggingFace dataset repository data/ folder.
        Args:
            dataset (ds.Dataset): Dataset to save and push.
            list_of_files_to_delete (list): List of files to delete from the hub.
                default: [] -> delete all datafiles not corresponding to the splits present in self.dataset
        """
        if list_of_files_to_delete == []:
            list_of_files_to_delete = self._repo_get_file_list_split_cleanup()

        if list_of_files_to_delete != []:
            print(f"{list_of_files_to_delete} will be deleted from {self._repo}")

            for file_to_delete in list_of_files_to_delete:
                print(f"Deleting {file_to_delete}")
                delete_file(
                    path_in_repo=file_to_delete,
                    repo_id=self._repo,
                    repo_type="dataset",
                )

        else:
            print(f"No dataset files to delete from {self._repo}")

    @staticmethod
    def save(dataset: Dataset, local_path: str) -> None:
        """Save a dataset to a local directory and push it to Hugging Face.
        Args:
            dataset (ds.Dataset): Dataset to save and push.
            local_dir (str): Local directory to save the dataset to."""

        # Save to a local directory
        dataset.save_to_disk(local_path)

    @staticmethod
    def push(dataset: Dataset, repo: str, split="whole_dataset") -> None:
        """Save a dataset to a local directory and push it to Hugging Face.
        Args:
            dataset (ds.Dataset): Dataset to save and push.
            repo (str): Hugging Face Hub repository name (format: 'Company/dataset-name').
            split (str): The name of the split to push to the hub. Defaults to "whole_dataset"
        """
        # push to hub
        dataset.push_to_hub(
            repo,
            private=True,
            split=split,
            embed_external_files=False,
        )


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(
        description="Create and push a dataset to Hugging Face."
    )
    # required arguments
    parser.add_argument(
        "--repo",
        required=True,
        help="Name of the repository to pull from on Hugging Face.",
    )
    # optional arguments
    parser.add_argument(
        "--commit_hash",
        required=False,
        help="str:The commit hash of the dataset to pull from Hugging Face.",
        default=None,
    )
    parser.add_argument(
        "--filename",
        required=False,
        help="Path to the CSV file for dataset creation.",
        default=None,
    )
    parser.add_argument(
        "--local_dir",
        required=False,
        help="Name of the local directory to save the dataset to.",
        default="./local_datasets",
    )
    parser.add_argument(
        "--cache_dir",
        required=False,
        help="cache directory",
        default="./cache",
    )
    parser.add_argument(
        "--save_locally",
        required=False,
        help="Name of the local directory to save the dataset to.",
        default=True,
    )

    parser.add_argument(
        "--update_just_readme",
        required=False,
        help="When update - just update the readme or dataset card.",
        default=False,
    )

    parser.add_argument(
        "--splits_to_combine_as_whole_ds",
        required=False,
        help="list: names of the split to combine as 'whole_dataset'",
        nargs="+",
        default=[],
    )

    parser.add_argument(
        "--needed_encodings",
        required=False,
        help="list: list of encodings for the dataset; default: []",
        nargs="+",
        default=[],
    )

    parser.add_argument(
        "--overwrite_repo",
        required=False,
        help="bool: Set to True to overwrite a repo when uploading a new dataset, if the repo already exists",
        default=False,
    )

    args = parser.parse_args()
    print(args)
    # create dataset upload object
    data_upload = DatasetUpload(
        args.repo,
        commit_hash=args.commit_hash,
        local_dir=args.local_dir,
        cache_dir=args.cache_dir,
        csv_filename=args.filename,
        save_locally=args.save_locally,
        splits_to_combine_as_whole_ds=args.splits_to_combine_as_whole_ds,
        update_just_readme=args.update_just_readme,
        needed_encodings=args.needed_encodings,
        overwrite_repo=args.overwrite_repo,
    )

    # process dataset
    data_upload.process(verbose=True)

    # exmaple usage:
    # python examples/dataset_upload.py --repo Company/test_example_dataset_ha1 --filename examples/datasets/test_example_dataset_ha1.csv
