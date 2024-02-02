import datasets
from pedata.preprocessing import (
    DatasetSplitterRandomTrainTest,
    DatasetSplitterRandomKFold,
)

dataset_dict = {
    "aa_seq": [
        "MLGLYITR",
        "MAGLYITR",
        "MLYLYITR",
        "RAGLYITR",
        "MLRLYITR",
        "RLGLYITR",
    ],
    "target a": [1, 2, 3, 5, 4, 6],
    "aa_mut": [
        "A2L",
        "wildtype",
        "A2L_G3Y",
        "M1R",
        "A2L_G3R",
        "A2L_M1R",
    ],
}

print("-------- Input dataset being a (HF) Dataset --------")
toy_dataset = datasets.Dataset.from_dict(dataset_dict)

dataset = DatasetSplitterRandomTrainTest().split(toy_dataset)
print(dataset)

print("-------- Input dataset being a (HF) DatasetDict --------")
toy_dataset_dict = datasets.DatasetDict(
    {"all": datasets.Dataset.from_dict(dataset_dict)}
)
dataset = DatasetSplitterRandomTrainTest().split(toy_dataset_dict)
print(dataset)

print("-------- random train/test split and return a concatenated Dataset --------")

dataset = DatasetSplitterRandomTrainTest().split(
    toy_dataset_dict, return_dataset_dict=False
)
print(dataset)

print(
    "-------- Splitting a dataset containing the split column returns the same dataset --------"
)
dataset_out = DatasetSplitterRandomTrainTest().split(
    toy_dataset_dict, return_dataset_dict=False
)
print(dataset.to_pandas())
print(dataset_out.to_pandas())
assert (
    dataset["aa_seq"] == dataset_out["aa_seq"]
    and dataset["random_split_train_0_8_test_0_2"]
    == dataset_out["random_split_train_0_8_test_0_2"]
)

print("-------- random train/test split and return a DatasetDict -----------------")
dataset = DatasetSplitterRandomTrainTest().split(
    toy_dataset_dict, return_dataset_dict=True
)
print(dataset)

print(
    "-------- random train/test and return a concatenated Dataset with only a selected split -----------------"
)
splitter = DatasetSplitterRandomTrainTest()
dataset = splitter.split(toy_dataset, return_dataset_dict=True)
assert isinstance(dataset, datasets.DatasetDict)
dataset_cat = splitter.concatenated_dataset(dataset, split_list=["train"])
print(dataset_cat)


print("-------- k-fold split -----------------")
splitter = DatasetSplitterRandomKFold(k=3)
dataset = splitter.split(toy_dataset)
print(dataset)


print("-------- k-fold split - return all train_test_sets -----------------")
k = 0
for train_test_set in splitter.yield_all_train_test_sets(dataset):
    k += 1
    print(f"- train_test_{k} -")
    print(train_test_set)


print(
    "-------- k-fold split - return all train_test_sets - combining 2 splits for generating test set-----------------"
)
splitter = DatasetSplitterRandomKFold(k=6)
dataset = splitter.split(toy_dataset)

for split_n, split in enumerate(
    splitter.yield_all_train_test_sets(dataset, combined_n=2)
):
    print(f"- split_combination_{split_n} -")
    print(split["test"]["random_split_6_fold"])
