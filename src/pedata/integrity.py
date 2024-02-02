import datasets as ds
from .constants import Mut, Mut_with_no_targ


def check_dataset(df: ds.Dataset):
    """
    Function checks if the dataset is valid.

    Args:
        df (ds.Dataset): Dataset to check. It should be an instance of ds.Dataset.

    Raises:
        TypeError: If the input dataset is not an instance of ds.Dataset.
        KeyError: If the dataset is missing a required column or contains a non-allowed column.

    Example:
        >>> dataset = ds.Dataset.from_dict({"aa_seq": [None, None]})
        KeyError: Columns are missing in the dataset. In particular: aa_mut.

    """

    # FIXME - add this step for the Molecule datasets

    if isinstance(df, ds.Dataset):
        feature_keys = df.features.keys()
    else:
        raise TypeError("Mutation should be a dataset")
    df = df.with_format("pandas")
    missing_cols = []

    # Check if either "aa_seq" or "dna_seq" column is missing
    if "aa_seq" not in feature_keys and "dna_seq" not in feature_keys:
        missing_cols.append('"aa_seq" or "dna_seq"')

    else:
        # Calculate the number of missing values in either "aa_seq" or "dna_seq" column
        if "aa_seq" in feature_keys:
            missing_values = df["aa_seq"].isnull().sum()

            # Check if missing values exist and "aa_mut" column is missing
            if missing_values > 0 and "aa_mut" not in feature_keys:
                missing_cols.append("aa_mut")

        if "dna_seq" in feature_keys:
            missing_values = df["dna_seq"].isnull().sum()

            # Check if missing values exist and "aa_mut" column is missing
            if missing_values > 0 and "dna_mut" not in feature_keys:
                missing_cols.append("dna_mut")

    # Check if "target summary variable" column is present
    if "target summary variable" in feature_keys:
        raise KeyError(
            "There was already a column called 'target summary variable' in the data set. This is a special column name reserved for Companys internal use."
        )

    # Check if any column starts with "target"
    if len([k for k in feature_keys if k.lower().startswith("target")]) == 0:
        missing_cols.append('a column starting with "target"')

    # Raise KeyError if any missing columns are found
    if len(missing_cols) > 0:
        raise KeyError(
            f"Columns are missing in the data file. In particular: {', '.join(missing_cols)}."
        )


# Validate namedtuple mutation
def check_mutation_namedtuple(m: Mut):
    """
    Function validates a namedtuple mutation.

    Args:
        m (Mut): The namedtuple mutation to validate.

    Raises:
        TypeError: If the mutation is not a valid namedtuple or if attributes are of incorrect types.

    Example:
        >>> mutation = Mut(pos='2', src='A', targ='C')
        >>> check_mutation_namedtuple(mutation)
        TypeError: Attribute 'pos' should exist in a namedtuple and be an int
    """

    if not (isinstance(m, Mut) or isinstance(m, Mut_with_no_targ)) or len(m) < 2:
        raise TypeError(
            "Invalid format. Each mutation namedtuple should be an instance of a Mut namedtuple with at least two attributes: (pos, src), and atmost 3 attributes: (pos, src, targ)"
        )

    if not hasattr(m, "pos") or not isinstance(m.pos, int):
        raise TypeError("Attribute 'pos' should exist in a namedtuple and be an int")

    if not hasattr(m, "src") or not isinstance(m.src, str):
        raise TypeError("Attribute 'src' should exist in a namedtuple and be a string")

    if len(m) == 3:
        if not hasattr(m, "targ") or not isinstance(m.targ, str):
            raise TypeError(
                "If a namedtuple has length of 3, attribute 'targ' should exist as a string"
            )
