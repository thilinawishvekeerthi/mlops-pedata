"""
# ideas for visualizations
Amino acid sequences
- length of the sequence
- wild-type or base sequence representation according to the amino-acid properties (hydrophobicity / hydrophilicity / charge / size / polarity / aromaticity)
- corresponding linear map of the total amount of mutations from Cterm to Nterm
- distance map (calculated from AA properties) of the different mutated sequence from the wild-type sequence (check blossum matrix)

Visualization of the target - DONE
- histogram of the target values 

Target vs distance map
- plot the target values against the distance map

# other ideas - for later
motifs and domains / database annotation

residues in the active sites:
- residues relevant for substrate binding
- residues relevant for co-factor binding
- good to see if we have mutations there and what they do

homology: highly conserved residues could be critical for the folding and modifying there could be a bad idea - but not necessarily

"""
import matplotlib.pyplot as plt
import numpy as np
import os

2


def plot_target_distribution(
    target: np.array,
    label: str = "target",
    savedir: str = None,
) -> None:
    """plot the target distribution

    Args:
        target (np.array): target values
        transformed (bool, optional): if the target is transformed. Defaults to False.
        label (str, optional): label for the target. Defaults to "target".

    Returns:
        None

    Example:
        >>> import numpy as np
        >>> target = np.random.normal(0, 1, 1000)
        >>> plot_target_distribution(target, label="norm_random_target")
    """
    if not isinstance(target, np.ndarray):
        raise TypeError("`target` should be a numpy array")

    if savedir is not None:
        os.makedirs(f"{savedir}/figures", exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.hist(target, bins=50, color="black")
    plt.xlabel(label)
    plt.ylabel("occurence")
    plt.title(f"{len(target)} datapoints")
    plt.tight_layout()
    # plt.show()
    if savedir is not None:
        plt.savefig(f"{savedir}/figures/distribution_{label}.png")
    plt.close()


def plot_target_distributions(target_dict, savedir=None, label=None):
    """
    Runs plot_target_distribution for all targets in the target_dict

    Args:
        target_dict: dictionary of targets. From by pedata.util.get_target(dataset, as_dict=True)
        savedir: directory to save the plots
        label: label for the targets

    Returns:
        None

    Example:
        >>> target_1 = np.random.normal(0, 1, 1000)
        >>> target_2 = np.random.normal(0, 1, 1000)
        >>> targets = {
        ...    "target_1": target_1,
        ...    "target_2": target_2,
        ... }
        >>> plot_target_distributions(
        ...    targets, label="pytest_2_norm_random_targets", savedir="code_health"
        ... )
    """
    if label is None:
        label = ""

    for t_name, t_data in target_dict.items():
        plot_target_distribution(t_data, label=f"{label}_{t_name}", savedir=savedir)


if __name__ == "__main__":
    pass
