import shutil
import subprocess

import requests
from huggingface_hub import delete_repo, repo_exists
from pytest import fixture


# check if we are online and can access huggingface
def huggingface_hub_access():
    try:
        # Try to access the Hugging Face Hub
        response = requests.get("https://huggingface.co/")
        response.raise_for_status()  # Raise an HTTPError for bad responses
        # If the request was successful, print a message
        print("Connected to Hugging Face Hub!")
        return True
    except requests.exceptions.RequestException as e:
        # Handle exceptions related to connection issues or bad responses
        print(f"Error accessing Hugging Face Hub: {e}")
        return False


if huggingface_hub_access():
    # ========== HELPER FUNCTIONS ==========
    def delete_cache_and_repo(cache_dir: str, repo_name: str) -> None:
        """Delete the cache and the repo
        Args:
            cache_dir: cache dir to delete
            repo_name: repo name to delete
        """
        shutil.rmtree(cache_dir, ignore_errors=True)
        if repo_exists(repo_id=repo_name, repo_type="dataset"):
            delete_repo(repo_id=repo_name, repo_type="dataset")

    def run_command(command):
        subprocess.run(command, shell=True, check=True)

    # ========== FIXTURES ==========
    @fixture(scope="module")
    def repo_name():
        return "Company/TestDataset_upload_update_delete"

    @fixture(scope="module")
    def cache_dir():
        return "./cache"

    @fixture(scope="module")
    def csv_filename():
        return "TestDataset_upload_update_delete.csv"

    @fixture(scope="module")
    def needed_encodings():
        return ["aa_seq", "aa_1hot"]

    @fixture(scope="module")
    def updated_encodings():
        return ["aa_unirep_1900"]

    # ========== TESTS ==========
    # def test_Dataset_upload_and_update(
    #     regr_dataset,
    #     needed_encodings,
    #     csv_filename,
    #     updated_encodings,
    #     repo_name,
    #     cache_dir,
    # ):
    #     """Testing the Upload Class"""

    #     # cleanup in case test failed before
    #     delete_cache_and_repo(cache_dir, repo_name)

    #     # create a csv file containing the dataset
    #     save_dataset_as_csv(regr_dataset, csv_filename)

    #     # 1 ==== uploading the dataset
    #     upload = DatasetUpload(
    #         repo=repo_name,
    #         save_locally=False,
    #         csv_filename=csv_filename,
    #         needed_encodings=needed_encodings,
    #         cache_dir=cache_dir,
    #     )
    #     upload.process()

    #     # load the dataset from huggingface and check that it has the correct encodings
    #     dll_ds = load_dataset(
    #         repo_name, download_mode="force_redownload", cache_dir=cache_dir
    #     )
    #     assert sorted(list(dll_ds["whole_dataset"].info.features)) == sorted(
    #         [
    #             "aa_mut",
    #             "target_kcat_per_kmol",
    #             "aa_seq",
    #             "aa_1hot",
    #             "index",
    #             "random_split_train_0_8_test_0_2",
    #             "random_split_10_fold",
    #         ]
    #     )

    #     # cleanup
    #     os.remove(csv_filename)  # csvfile
    #     shutil.rmtree(cache_dir)  # cache dir

    #     # 2 ==== Update the dataset with the encodings per default
    #     update = DatasetUpload(
    #         repo=repo_name, needed_encodings=updated_encodings, cache_dir=cache_dir
    #     )
    #     update.process()

    #     # load the dataset from huggingface and check that it has the correct encodings
    #     dll_ds = load_dataset(
    #         repo_name, download_mode="force_redownload", cache_dir=cache_dir
    #     )

    #     assertion_statement = sorted(
    #         list(dll_ds["whole_dataset"].info.features)
    #     ) == sorted(
    #         [
    #             "aa_mut",
    #             "target_kcat_per_kmol",
    #             "aa_seq",
    #             "aa_1hot",
    #             "index",
    #             "random_split_train_0_8_test_0_2",
    #             "random_split_10_fold",
    #             "aa_unirep_1900",
    #             "aa_unirep_final",
    #         ]
    #     )
    #     assert assertion_statement

    #     ## delete the cache
    #     shutil.rmtree(cache_dir)

    #     # 3 ==== Just update the readme ====
    #     update = DatasetUpload(
    #         repo=repo_name,
    #         update_just_readme=True,
    #     )
    #     update.process()
    #     assert assertion_statement

    #     # 4 ======== Specifying splits to combine as a whole dataset
    #     update = DatasetUpload(
    #         repo=repo_name,
    #         splits_to_combine_as_whole_ds=["whole_dataset"],
    #     )
    #     update.process()
    #     assert assertion_statement

    #     # delete the cache and repo
    #     delete_cache_and_repo(cache_dir, repo_name)

    # =======  These tests are testing additional features and also test running in the command line

    # def test_upload_csv_to_huggingface_repo_already_exists():
    #     """trying to upload to a repo which already exist"""

    #     # raising error because repo already exists
    #     wrong_command = (
    #         "python "
    #         "src/pedata/hfhub_tools/upload.py "
    #         "--repo Company/test_example_dataset_ha1 "
    #         "--filename local_datasets/datafiles/example_dataset_ha1.csv "  # optional
    #         "--needed_encodings 'aa_seq' 'aa_1hot' 'aa_unirep_1900' "  # optional
    #     )
    #     with raises(Exception):
    #         run_command(wrong_command)

    #     # forcing overwrite
    #     command = (
    #         "python "
    #         "src/pedata/hfhub_tools/upload.py "
    #         "--repo Company/test_example_dataset_ha1_2 "
    #         "--filename local_datasets/datafiles/example_dataset_ha1.csv "  # optional
    #         "--needed_encodings 'aa_seq' 'aa_1hot' 'aa_unirep_1900' "  # optional
    #         "--overwrite_repo True "
    #     )
    #     run_command(command)

    #     # pulling a dataset from huggingface, updating it and uploading it uploading to the same repo"""

    #     command = (
    #         "python "
    #         "src/pedata/hfhub_tools/upload.py "
    #         "--repo Company/test_example_dataset_ha1 "
    #         "--needed_encodings 'aa_seq' 'aa_1hot' 'aa_unirep_1900' "  # optional
    #     )
    #     run_command(command)
