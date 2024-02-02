import pedata.encoding.embeddings_transform as et
import numpy as np
from pedata.static.example_data.data import RegressionToyDataset

if __name__ == "__main__":
    dataset_train = RegressionToyDataset(needed_encodings=["aa_1hot", "aa_len"]).train

    # reshape 1hot
    emb_list_1hot_train = et.reshape_1hot(
        one_hot_array=dataset_train["aa_1hot"],
        len_seq=np.unique(dataset_train["aa_len"])[0],
        len_aa=21,
    )

    # Compute rkhs kmer embeddings
    kmer_size = 100
    nystrom_size = 256
    rkhs_kmer_emb_list = et.rkhs_kmer_embeddings(
        kernel_fn=et.linear_kernel,
        emb_list=emb_list_1hot_train,
        kmer_size=kmer_size,
        nystrom_size=nystrom_size,
    )
    assert rkhs_kmer_emb_list.shape == (16, 256)
    print(rkhs_kmer_emb_list)
