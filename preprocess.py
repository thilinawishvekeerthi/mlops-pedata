from datetime import date
from itertools import product

from pathlib import Path
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
import jax.numpy as np
import pandas as pd
import numpy as onp
import re
from typing import List, Union, Any
import types
import os
import sys
import json
from pedata.disk_cache import load_similarity, preprocess_data
from pedata.config.paths import data_exists, get_filename
from pedata.config import encodings

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", default="", help="filename")
parser.add_argument("-aa", default=None, help="Amino acid sequence col")
parser.add_argument("-dna", default=None, help="DNA sequence col")
parser.add_argument("-t", default="", help="target col")
parser.add_argument("--similarities", dest="similarities", action="store_true")
parser.add_argument("--no-similarities", dest="similarities", action="store_false")
parser.set_defaults(similarities=False)
parser.add_argument("--replace", dest="replace_existing", action="store_true")
parser.add_argument("--no-replace", dest="replace_existing", action="store_false")
parser.set_defaults(replace_existing=False)
args = parser.parse_args()


def get_all_encodings(
    filename: str,
    aa_seqname: str,
    dna_seqname: str,
    target: str,
    replace_existing: bool = False,
):
    # get all encodings for the dataset

    seq_types = set()
    if aa_seqname is not None:
        seq_types.add("aa")
    if dna_seqname is not None:
        seq_types.add("dna")

    assert len(seq_types) > 0, "Not both aa_seqname and dna_seqname can be None"

    name, end = get_filename(filename)

    if data_exists(name + "_reduced" + end) and not replace_existing:
        print(
            f"Reduced file for {filename} already exists and will not be replaced",
            file=sys.stderr,
        )
    else:
        print("Reducing")
        if "," in target:
            target = target.split(",")
        reduce_data(filename, aa_seqname, target, dna_seqname=dna_seqname)


if __name__ == "__main__":
    if args.similarities:
        load_similarity(
            "aa",
            ["BLOSUM62", "BLOSUM90", "IDENTITY", "PAM20"],
            write_cache=True,
            replace_existing=args.replace_existing,
        )
        load_similarity(
            "dna",
            ["Identity", "Simple"],
            write_cache=True,
            replace_existing=args.replace_existing,
        )
    assert (
        args.similarities is not None or args.aa is not None or args.dna is not None
    ), "At least one of -aa or -dna must be provided."
    if len(args.f) > 1:
        get_all_encodings(
            args.f, args.aa, args.dna, args.t, replace_existing=args.replace_existing
        )
