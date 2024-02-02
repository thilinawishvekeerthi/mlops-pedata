import os
import sys
from pathlib import Path
from datasets import Dataset, DatasetDict
import pandas as pd
import jax.numpy as np  # FIXME can we use only numpy or is jax.numpy fully necessary here?
import numpy as onp
from .integrity import check_dataset
from .config import alphabets, paths, add_encodings
from .mutation.mutation import Mutation
from .transform import transform_pipeline
import fsspec


def _read_csv_ignore_case(file_path: str) -> pd.DataFrame:
    """Reads a CSV file with a case-insensitive match

    Args:
        file_path: path to the file

    Returns:
        The dataframe

    Raises:
        FileNotFoundError if no file is matching
    """
    directory, file_name = os.path.split(file_path)
    if len(directory) == 0:
        directory = os.getcwd()
    # List all files in the directory
    files_in_directory = os.listdir(directory)

    # Find the file with a case-insensitive match
    matching_files = [
        file for file in files_in_directory if file.lower() == file_name.lower()
    ]

    if not matching_files:
        raise FileNotFoundError(f"No file found matching: {file_name}")

    # Use the first matching file (in case there are multiple matches)
    matching_file_path = os.path.join(directory, matching_files[0])

    # Use pd.read_csv with the found file path
    return pd.read_csv(matching_file_path)


def read_dataset_from_file(filename: str) -> Dataset:
    """Reads a CSV, Excel or Parquet file and return a HuggingFace dataset.
    Args:
        filename: File name of the source CSV file.

    Returns:
        A HuggingFace dataset.

    Raises:
        TypeError: If the input type is not a CSV, Excel or Parquet file.

    """
    filename = str(filename).lower()

    # Check file format
    if filename.endswith("csv"):
        df = _read_csv_ignore_case(filename)

    elif filename.endswith("xls") or filename.endswith("xlsx"):
        df = pd.read_excel(filename, 0)

    elif filename.endswith("parquet"):
        df = pd.read_parquet(filename)

    else:
        raise TypeError("Invalid input: input either a csv, excel or parquet file")

    return Dataset.from_pandas(df)


def save_dataset_as_csv(
    dataset: DatasetDict | Dataset | pd.DataFrame,
    filename: str | Path,
) -> None:
    """Saves the dataset as csv

    Args:
        dataset: dataset to save
        filename: filename to save the dataset
    """
    if isinstance(dataset, DatasetDict):
        for split in dataset.keys():
            save_dataset_as_csv(
                dataset[split], filename=f"{filename.split('.')[0]}_{split}.csv"
            )
    else:
        if isinstance(filename, str):
            filename = Path(os.path.abspath(filename))

        if isinstance(dataset, Dataset):
            # convert Dataset to pandas dataframe
            dataset = dataset.to_pandas()

        # save as csv
        dataset.to_csv(filename, index=False)


def get_missing_values(dataset: Dataset, feature: str) -> list[bool]:
    """get missing values in the `feature` column

    Args:
        dataset: dataset to check
        feature: which column to check for missing values

    Returns:
        True for the indices corresponding to the missing values

    """
    df = dataset.to_pandas()
    return df.loc[:, feature].isna() | df.loc[:, feature].isnull()


def fill_missing_sequences(dataset: Dataset, feature: str) -> Dataset:
    """Fill missing values in the `feature` column

    Args:
        dataset: dataset to check
        feature: which column to fill for missing values

    Returns:
        Dataset: dataset with filled missing values
    """
    missing_values = get_missing_values(dataset, feature)
    df = dataset.to_pandas()
    if missing_values.sum() > 0:
        # Apply mutations to fill in missing values in 'dna_seq' column
        df.loc[missing_values, feature] = pd.DataFrame(
            {feature: Mutation.apply_all_mutations(dataset)}
        ).loc[missing_values, feature]

    return Dataset.from_pandas(df)


def hfds_from_pydict(
    dataset_dict: dict,
    needed_encodings: list[str] = [],
    as_DatasetDict: bool = False,
) -> Dataset | DatasetDict:
    """Returns a Dataset or DatasetDict from a dataset_dict

    Args:
        dataset_dict: the python dictionnary containing regression data
        as_DatasetDict: whether to return DatasetDictionary with one split or a Dataset
        needed_encodings:

    Returns:
        a Dataset or DatasetDictionary with the data from the input dictionnary
    """
    dataset = Dataset.from_dict(dataset_dict)
    # Perform dataset integrity check
    check_dataset(dataset)

    if "aa_seq" in dataset.column_names:
        # Check for missing values in the 'aa_seq' column
        dataset = fill_missing_sequences(dataset, "aa_seq")

    elif "dna_seq" in dataset.column_names:
        # Check for missing values in the 'dna_seq' column
        dataset = fill_missing_sequences(dataset, "dna_seq")

    # Add encodings to dataset
    dataset = add_encodings(dataset, needed=needed_encodings)

    # Return the processed dataset
    if as_DatasetDict:
        return dataset.DatasetDict({"whole_dataset": dataset})
    else:
        return dataset


def preprocess_data(
    filename: str | Path,
    save_to_path: str | None = None,
    filesystem: fsspec.AbstractFileSystem | None = None,
    needed_encodings: list[str] | set[str] = [],
    add_index: bool = True,
    add_splits: bool = True,
) -> Dataset:
    """Transforms a data file (CSV or Excel) into a Hugging Face dataset and computes all available features.

    Args:
        filename: File name of the source CSV or Excel file.
        save_to_path: Path to save the Hugging Face dataset. Defaults to None (not saved).
        filesystem: File system to use for saving. Defaults to None (local filesystem).
        needed_encodings : List of encodings needed for the model. Defaults to [].
        add_index: Add an index column to the dataset. Defaults to True.
        add_splits: Add a split column to the dataset. Defaults to True.

    Returns:
        The dataset with all precomputed features.

    Example:
        >>> import pandas as pd
        >>> data = pd.DataFrame({"aa_mut": ["wildtype", "T8M", "P3G"],"aa_seq": ["GMPKSEFTHC", None, None],"target foo": [1, 2, 3]})
        >>> csv_file = "test_data.csv"
        >>> data.to_csv(csv_file, index=False)
        >>> dataset = preprocess_data(csv_file)
        >>> print(dataset)
        Dataset({
            features: ['aa_mut', 'aa_seq', 'target foo', 'target summary variable', 'aa_unirep_1900', 'aa_unirep_final', 'aa_len', 'aa_1gram', 'aa_ankh_base', 'aa_esm2_t6_8M', 'aa_1hot'],
            num_rows: 3
        })

    Note:
        The example above converts a dataframe into a CSV file and saves it in the current directory as "test_data.csv".
        The CSV file is then passed to the preprocess_data() function and is preprocessed to compute more features, and return a new HuggingFace dataset.
    """

    # Read the dataset from the specified file
    dataset = read_dataset_from_file(filename)

    # Perform dataset integrity check
    check_dataset(dataset)

    if "aa_seq" in dataset.column_names:
        # Check for missing values in the 'aa_seq' column
        dataset = fill_missing_sequences(dataset, "aa_seq")

    elif "dna_seq" in dataset.column_names:
        # Check for missing values in the 'dna_seq' column
        dataset = fill_missing_sequences(dataset, "dna_seq")

    # add the basics to the dataset
    dataset = transform_pipeline(
        dataset,
        needed_encodings=needed_encodings,
    )

    # Save the data to a specified path, if provided
    if save_to_path is not None:
        dataset.save_to_disk(save_to_path, storage_options=filesystem.storage_options)

    # Return the processed dataset
    return dataset


def load_similarity(
    alphabet_type: str,
    similarity_name: str | list[str],
    replace_existing: bool = False,
) -> tuple[list[str], np.ndarray]:
    """
    Load similarity matrices.
        Loads similarity matrices based on the specified alphabet type and similarity names. It provides the capability to load
        multiple similarity matrices simultaneously and preprocesses them into usable similarity matrices. The function also supports
        optional caching of the loaded matrices for improved performance in subsequent operations.

    Args:
        alphabet_type: Specifies the type of alphabet used in the similarity calculation. It can be either "aa" for amino acids or "dna" for DNA sequences.
        similarity_name: The name or list of names of the similarity matrix/matrixes to be loaded.
        replace_existing: Determines if an existing matrix should be overwritten. Defaults to False.

    Returns:
        A tuple containing the alphabet used for the similarity calculation and the preprocessed similarity matrix.

    Example:
    >>> similarity_names = ['name1', 'name2']
    >>> alphabet, similarity_matrix = load_similarity('aa', similarity_names)
    >>> print(similarity_matrix)

    Raises:
        ValueError: If the specified alphabet type is invalid
        ValueError: If the similarity matrix dimensions are not valid. FIXME not tested
        ValueError: If the similarity matrix contains superfluous entries.
        Exception: If the similarity matrix is missing entries. FIXME not tested

    """

    # check which alphabet to use
    if alphabet_type == "aa":
        alph = onp.array(alphabets.aa_alphabet)
    elif alphabet_type == "dna":
        alph = onp.array(alphabets.dna_alphabet)
    else:
        raise ValueError(f"Invalid alphabet type: {alphabet_type}")

    rval = []  # Stores preprocessed similarity matrix as return value

    # Ensure similarity_name is a list
    if isinstance(similarity_name, str):
        similarity_name = [similarity_name]

    for s in similarity_name:
        # Prepare file paths
        similarity_filename, file_extension = alphabet_type + "_" + s, ".txt"
        output_file_path = os.path.join(
            paths.path_simil, f"{similarity_filename}_ordered.csv"
        )

        if paths.data_exists(output_file_path) and not replace_existing:
            # Load the similarity matrix from cache
            print(
                f"\n--- Existing disk cache ---\nFile: {output_file_path}\nStatus: Existing file will not be replaced\n---\n",
                file=sys.stderr,
            )
            rval.append(onp.loadtxt(output_file_path, delimiter=","))

        else:
            # Open the similarity matrix file and read its lines
            with open(
                os.path.join(paths.path_simil, similarity_filename + file_extension)
            ) as matrix_file:
                lines = matrix_file.readlines()

            header = None  # Variable to store the header of the similarity matrix
            col_header = []  # List to store the column header of the similarity matrix
            similarity_matrix = []  # List to store the similarity matrix entries

            for idx, row in enumerate(lines):
                # Skip commented lines and empty lines
                if row[0] == "#" or len(row) == 0:
                    continue

                # Strip leading and trailing whitespace from the row
                row = row.strip()

                # Split the row into individual entries
                entries = row.split()

                if header is None:
                    # First non-comment and non-empty line represents the header
                    header = entries
                    continue

                else:
                    # The first entry in each subsequent line is the column header
                    col_header.append(entries.pop(0))

                    # Convert the remaining entries to floats and append them to the similarity matrix
                    similarity_matrix.append(list(map(float, entries)))

            # Convert the header and column header to numpy arrays
            header, col_header = onp.array(header), onp.array(col_header)

            # Convert the similarity matrix to a jax numpy array
            similarity_matrix = np.array(similarity_matrix)

            # Check the dimensions and consistency of the matrix
            if not np.all(header == col_header):  # FIXME: missing test
                raise ValueError(
                    "Inconsistent header and column header in the similarity matrix: "
                    "The values in the header and column header do not match."
                )

            if (
                len(header) != similarity_matrix.shape[0]
                or similarity_matrix.shape[0] != similarity_matrix.shape[1]
            ):
                raise ValueError("Dimensions of the similarity matrix are not valid.")

            # ?? Replace the missing value placeholder in the header if present
            if header[-1] == "*":
                header[-1] = alphabets.stop_codon_enc

            # Check for superfluous entries in the similarity matrix
            superfluous_entries = set(header).difference(alph)
            if len(superfluous_entries) > 0:
                print(
                    f"Similarity matrix contains superfluous entries {superfluous_entries}"
                )

            # Check for missing entries in the similarity matrix
            missing_entries = set(alph).difference(header)
            if len(missing_entries) != 0:  # FIXME: missing test
                raise Exception(f"Similarity matrix doesn't contain {missing_entries}")

            # Reorder the similarity matrix based on the alphabet order
            reorder = np.argmax(header[:, None] == alph[None, :], 0)

            # Append the reordered similarity matrix to the result list
            rval.append(similarity_matrix[reorder, :][:, reorder])

            # Save the reordered similarity matrix to disk for future use
            onp.savetxt(
                output_file_path,
                rval[-1],
                delimiter=",",
                header=", ".join(alph),
                fmt="%.2f",
            )

    return (
        alph,
        onp.array(rval).squeeze(),
    )  # Return the alphabet and the preprocessed similarity matrix
