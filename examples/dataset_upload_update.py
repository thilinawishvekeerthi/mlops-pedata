from pedata.hfhub_tools import DatasetUpload
from pedata import RegressionToyDataset
from huggingface_hub import delete_repo, repo_exists
from datasets import load_dataset, Dataset
import os
import shutil
from pedata import save_dataset_as_csv


def delete_cache_and_repo(cache_dir: str, repo_name: str) -> None:
    """Delete the cache and the repo
    Args:
        cache_dir: cache dir to delete
        repo_name: repo name to delete
    """
    shutil.rmtree(cache_dir, ignore_errors=True)
    if repo_exists(repo_id=repo_name, repo_type="dataset"):
        delete_repo(repo_id=repo_name, repo_type="dataset")


def regr_dataset(needed_encodings: list[str]) -> Dataset:
    """Regression dataset - full dataset
    Args:
        needed_encodings (list): list of encodings needed for the model. Default: []
    Returns:
        dataset: train split"""
    ds = RegressionToyDataset(needed_encodings)
    return ds.full_dataset


if __name__ == "__main__":
    repo_name = "Company/TestDataset_upload_update_delete"
    cache_dir = "./cache"
    csv_filename = "TestDataset_upload_update_delete.csv"
    needed_encodings = ["aa_seq", "aa_1hot"]
    updated_encodings = ["aa_unirep_1900"]
    dataset = regr_dataset(needed_encodings)

    """processing an csv file,
    creating a dataset acccording to Company standards
    and uploading a dataset to a huggingface repository"""

    # cleanup in case test failed before
    delete_cache_and_repo(cache_dir, repo_name)

    # create a csv file containing the dataset
    save_dataset_as_csv(dataset, csv_filename)

    ### 1 ==== uploading the dataset
    upload = DatasetUpload(
        repo=repo_name,
        save_locally=False,
        csv_filename=csv_filename,
        needed_encodings=needed_encodings,
    )
    upload.process()

    # load the dataset from huggingface and check that it has the correct encodings
    dll_ds = load_dataset(
        repo_name, download_mode="force_redownload", cache_dir=cache_dir
    )
    assert sorted(list(dll_ds["whole_dataset"].info.features)) == sorted(
        [
            "aa_mut",
            "target_kcat_per_kmol",
            "aa_seq",
            "aa_1hot",
            "index",
            "random_split_train_0_8_test_0_2",
            "random_split_10_fold",
        ]
    )

    # cleanup
    os.remove(csv_filename)  # csvfile
    shutil.rmtree(cache_dir)  # cache dir

    ### 2 ==== Update the dataset with the encodings per default
    update = DatasetUpload(repo=repo_name, needed_encodings=updated_encodings)
    update.process()

    ## load the dataset from huggingface and check that it has the correct encodings
    dll_ds = load_dataset(
        repo_name, download_mode="force_redownload", cache_dir=cache_dir
    )

    assertion_statement = sorted(list(dll_ds["whole_dataset"].info.features)) == sorted(
        [
            "aa_mut",
            "target_kcat_per_kmol",
            "aa_seq",
            "aa_1hot",
            "index",
            "random_split_train_0_8_test_0_2",
            "random_split_10_fold",
            "aa_unirep_1900",
            "aa_unirep_final",
        ]
    )
    assert assertion_statement

    ## delete the cache
    shutil.rmtree(cache_dir, ignore_errors=True)

    # 3 ==== Just update the readme ====
    update = DatasetUpload(
        repo=repo_name,
        update_just_readme=True,
    )
    update.process()
    assert assertion_statement

    # 4 ======== Specifying splits to combine as a whole dataset
    update = DatasetUpload(
        repo=repo_name,
        splits_to_combine_as_whole_ds=["whole_dataset"],
    )
    update.process()
    assert assertion_statement

    ## delete the cache and repo
    delete_cache_and_repo(cache_dir, repo_name)
