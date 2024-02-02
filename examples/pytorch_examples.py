from pedata.pytorch_dataloaders import Dataloader
from datasets import load_dataset
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"======== Using device: {device} =======")
dataset = load_dataset("Exazyme/TemStaPro-Minor-bal65", split="testing[:50]")

# train mod
dataloader = Dataloader(
    dataset=dataset,
    embedding_names=["ankh", "aa_1hot"],
    targets=["growth_temp"],
    batch_size=8,
    device=device,
    shuffle=True,
)
num_samples = 0
for X, y in dataloader:
    num_samples += len(X["ankh"])
else:
    print(f"loop finished with {num_samples} samples")

assert num_samples == len(dataset)

# no target test mod
dataloader = Dataloader(
    dataset=dataset,
    embedding_names=["ankh", "aa_1hot"],
    batch_size=8,
    device=device,
    shuffle=True,
)
num_samples = 0
for X in dataloader:
    num_samples += len(X["ankh"])
else:
    print(f"loop finished with {num_samples} samples")

assert num_samples == len(dataset)
