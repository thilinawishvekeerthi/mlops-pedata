import subprocess
import shutil
import requests
from datasets import load_dataset, DownloadConfig
from pedata.hfhub_tools import (
    rename_hub_dataset_column,
)


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


if huggingface_hub_access:

    def run_command(command):
        subprocess.run(command, shell=True, check=True)

    def test_rename_hub_dataset_column():
        """testing the rename_hub_dataset_column function"""

        # helper functions
        def cleanup():
            shutil.rmtree(cache_directory, ignore_errors=True)

        def get_feat_list():
            cleanup()
            inf = load_dataset(
                "Company/test_example_dataset_ha1",
                download_config=dll_conf,
                download_mode="force_redownload",
                cache_dir=cache_directory,
            )
            cleanup()
            return list(inf[list(inf.keys())[0]].features)

        # clear cache
        cache_directory = "./cachebleeeeh"
        dll_conf = DownloadConfig(cache_dir=cache_directory, force_download=True)
        cleanup()

        ## creates a repo with the original names
        command = (
            "python "
            "src/pedata/hfhub_tools/upload.py "
            "--repo Company/test_example_dataset_ha1 "
            "--filename local_datasets/datafiles/example_dataset_ha1.csv "  # optional
            "--needed_encodings 'aa_seq' 'aa_1hot' 'aa_unirep_1900' "  # optional
            "--overwrite_repo True "
        )
        run_command(command)

        # changing the column name
        rename_hub_dataset_column(
            repo_id="Company/test_example_dataset_ha1",
            column_name_mapping={
                "aa_mut": "bleeeeh",
            },
        )

        # clear cache and get features list
        list_feat = get_feat_list()
        assert "bleeeeh" in list_feat
        assert "aa_mut" not in list_feat
