from pedata.preprocessing.tag_finder import TagFinder
from pedata.util import load_full_dataset

# Example of applying TagFinder to ProteineaSolubility
tf = TagFinder()
ds = load_full_dataset("Company/ProteineaSolubility")
no_tag_ds = tf.clean_dataset(dataset=ds, seq_col_name="aa_seq", num_proc=1)
