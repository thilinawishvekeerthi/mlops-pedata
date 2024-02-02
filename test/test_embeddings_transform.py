import numpy as np
from pytest import fixture

from pedata.encoding.embeddings_transform import (
    kmer_embeddings_centers,
    kmers_mean_embeddings,
    linear_kernel,
    reshape_1hot,
)


@fixture(scope="module")  # fixture for regr_dataset_test
def needed_encodings():
    return ["aa_len", "aa_1hot"]


@fixture
def lin_kernel():
    x = np.random.randn(10, 5)
    y = np.random.randn(20, 5)
    return linear_kernel(x, y)


@fixture()
def regr_dataset_seq_len(regr_dataset_test):
    return np.max(regr_dataset_test["aa_len"])


@fixture()
def nb_aa():
    return 21


def test_linear_kernel(lin_kernel):
    assert lin_kernel.shape == (10, 20)


def test_reshape_1hot(regr_dataset_test, regr_dataset_seq_len, nb_aa):
    """Reshaping of 1hot encodings to dimensiions (n_seqs, seq_len, 21)"""
    emb_list_1hot = reshape_1hot(
        regr_dataset_test["aa_1hot"], regr_dataset_seq_len, nb_aa
    )
    assert emb_list_1hot.shape == (5, 442, 21)


def test_kmers_mean_embeddings(regr_dataset_test, regr_dataset_seq_len, nb_aa):
    """testing kmers_mean_embeddings"""

    emb_list_1hot = reshape_1hot(
        regr_dataset_test["aa_1hot"], regr_dataset_seq_len, nb_aa
    )
    U, U_norm = kmers_mean_embeddings(emb_list_1hot, kmer_size=100)
    assert U.shape == (5, 2100)
    assert U_norm.shape == (5, 2100)


def test_kmer_embeddings_centers(regr_dataset_test, regr_dataset_seq_len, nb_aa):
    """testing kmers_embeddings_centers"""
    emb_list_1hot = reshape_1hot(
        regr_dataset_test["aa_1hot"], regr_dataset_seq_len, nb_aa
    )
    kmer_list, U1 = kmer_embeddings_centers(emb_list_1hot, kmer_size=100)
    assert kmer_list[0].shape == (343, 2100)
    assert U1.shape == (759, 2100)


# def test_rkhs_kmer_embeddings(regr_dataset_test, regr_dataset_seq_len, nb_aa):
#     # reshape 1hot
#     emb_list_1hot_train = reshape_1hot(
#         one_hot_array=regr_dataset_test["aa_1hot"],
#         len_seq=regr_dataset_seq_len,
#         len_aa=nb_aa,
#     )

#     # Compute rkhs kmer embeddings
#     kmer_size = 100
#     nystrom_size = 256
#     rkhs_kmer_emb_list = rkhs_kmer_embeddings(
#         kernel_fn=linear_kernel,
#         emb_list=emb_list_1hot_train,
#         kmer_size=kmer_size,
#         nystrom_size=nystrom_size,
#     )

#     assert rkhs_kmer_emb_list.shape == (5, 256)
#     assert list(np.round(rkhs_kmer_emb_list[0][:2], 6)) == list([0.279357, 0.132699])
