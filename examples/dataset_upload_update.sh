
# processing and csv file and uploading a dataset to a huggingface repository
python src/pedata/hfhub_tools/upload.py --repo Exazyme/test_example_dataset_ha1 --filename local_datasets/datafiles/example_dataset_ha1.csv

# pulling a dataset from huggingface, updating it and upload to/replacing the same repo
python src/pedata/hfhub_tools/upload.py --repo Exazyme/test_example_dataset_ha1



# 2 updates done on 2024/01/09
python src/pedata/hfhub_tools/upload.py --repo "Exazyme/IRED_selectivity_REGR" --commit_hash d969b84f91342c81cd6fd5b4073ad82be21a3a1d
python src/pedata/hfhub_tools/upload.py --repo "Exazyme/BetaLact_survival_REGR"
python src/pedata/hfhub_tools/upload.py --repo "Exazyme/CrHydA1_PE_REGR"