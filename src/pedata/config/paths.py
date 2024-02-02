"""Specify folder structure for the project"""
import os
from pathlib import Path


path_simil = (
    Path(os.path.realpath(os.path.join(os.path.dirname(__loader__.path), "..")))
    / "static"
    / "similarity_matrices"
)

# define base folder
DEFAULT_BASEDIR = "~/Devel/Protein_engineering"
PE_BASE_DIR = Path(os.path.expanduser(os.getenv("PE_BASE_DIR", DEFAULT_BASEDIR)))
# define subfolders
PE_DATA_DIR = Path(os.path.expanduser(os.getenv("PE_DATA_DIR", PE_BASE_DIR / "data")))
PE_RESULTS_DIR = Path(
    os.path.expanduser(os.getenv("PE_RESULTS_DIR", PE_BASE_DIR / "results"))
)
PE_CHECKPOINTS_DIR = Path(
    os.path.expanduser(os.getenv("PE_CHECKPOINTS_DIR", PE_BASE_DIR / "checkpoints"))
)


def get_filename(filename: str):
    """Separate file name and ending from a given filename.

    Args:
        filename (str): name of the file

    Returns:
        filename (str): name of the file without ending
        ending (str): ending of the filename with .

    """
    return filename[:-4], filename[-4:]


def data_exists(filename: str) -> bool:
    """Check if a file exists.

    Args:
        filename (str): name of the file

    Returns:
        bool: True if file exists

    """
    return os.path.isfile(filename)
