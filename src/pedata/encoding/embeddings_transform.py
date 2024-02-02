import itertools
import numpy as np
from scipy.linalg import fractional_matrix_power
from sklearn.cluster import KMeans
from typing import Union


def linear_kernel(x: np.array, y: np.array) -> np.array:
    """
    Linear kernel

    Args:
        x (np.array): array of shape (n,d)
        y (np.array): array of shape (m,d)

    Returns:
        np.array: kernel matrix of shape (n,m)

    Examples:
        >>> import numpy as np
        >>> from pedata.encoding.embeddings_transform import linear_kernel
        >>> x = np.random.randn(10,5)
        >>> y = np.random.randn(20,5)
        >>> K = linear_kernel(x,y)
        >>> K.shape
        (10, 20)
    """
    # FIXME: This does probably not belong here and should go into its own *.py file.
    # Also to be married with the gpytorch kernels potentially.
    return x @ y.T


def reshape_1hot(
    one_hot_array: Union[np.ndarray, list], len_seq: int, len_aa: int
) -> np.ndarray:
    """Reshape a numpy array 1hot encoding of shape (len(df),len_seq*len_aa)
    to a numpy array of shape (len(df), len_seq, len_aa)

    Args:
        one_hot_array : 1hot encoding of shape (len(df),len_seq*len_aa)
        len_seq : length of the sequence
        len_aa : number of amino acids

    Returns:
        np.array: dataset with 1hot encoding of shape (len(df), len_seq, len_aa)

    Examples:
        >>> import numpy as np
        >>> from pedata.static.example_data.example_data import RegressionToyDataset
        >>> from pedata.encoding.embeddings_transform import reshape_1hot
        >>> dataset = RegressionToyDataset(needed_encodings=["aa_1hot", "aa_len"]).full_dataset
        >>> aa_len = np.max(dataset["aa_len"])
        >>> nb_aa = 21
        >>> emb_list_1hot = reshape_1hot(dataset["aa_1hot"], aa_len, nb_aa)
        >>> emb_list_1hot.shape
        (21, 442, 21)

    Notes:
        The one hot encoding is reshaped from (len(df),len_seq*len_aa) to (len(df), len_seq, len_aa)
        This works with `df["aa_1hot"].reshape(len(df), len_seq, len_aa)` because the one hot encoding is padded with zeros.
        Padding can be done when turning a string of a sequence into a numpy array of characters in `seq_strings_to_array`.
        `SeqStrOneHot` uses this when turning sequences into one hot encodings and guarantees the same length. The pad character is there translated to an all-zero "1hot"-encoding.
    """
    if not isinstance(one_hot_array, np.ndarray):
        try:
            one_hot_array = np.array(one_hot_array)
        except TypeError:
            raise TypeError(
                "one_hot_array should be a numpy array, a list or a compatible array type"
            )

    return one_hot_array.reshape(len(one_hot_array), len_seq, len_aa)


def kmers_mean_embeddings(emb_list: list, kmer_size: int) -> np.array:
    """
    Compute the mean of kmer embeddings of length k for each sequence
    in the dataset and its l2-norm

    Args:
        emb_list (list): list of embeddings of shape (len_seq, len_aa)
        kmer_size (int): length of the kmer

    Returns:
        U (np.array): mean of kmer embeddings of length k for each sequence in the dataset
        U_norm (np.array): normalized (l2-norm) mean of kmer embeddings of length k for each sequence in the dataset

    Examples:
        >>> import numpy as np
        >>> from pedata.static.example_data.example_data import RegressionToyDataset
        >>> from pedata.encoding.embeddings_transform import reshape_1hot, kmers_mean_embeddings
        >>> dataset = RegressionToyDataset(needed_encodings=["aa_1hot", "aa_len"]).full_dataset
        >>> aa_len = np.max(dataset["aa_len"])
        >>> nb_aa = 21
        >>> emb_list_1hot = reshape_1hot(dataset["aa_1hot"], aa_len, nb_aa)
        >>> U, U_norm = kmers_mean_embeddings(emb_list_1hot, kmer_size=100)
        >>> U.shape
        (21, 2100)
        >>> U_norm.shape
        (21, 2100)
    """
    k = kmer_size
    d = emb_list[0].shape[1]
    U = np.zeros((len(emb_list), k * d))
    for i in range(0, len(emb_list)):
        # print(i)
        n = emb_list[i].shape[0]
        X = np.zeros((n - k + 1, k * d))
        for j in range(0, n - k + 1):
            X[j, :] = emb_list[i][j : j + k, :].reshape(1, -1)
        U[i, :] = np.mean(X, axis=0)
    U_norm = U / np.linalg.norm(U, axis=1)[:, None]
    return U, U_norm


def kmer_embeddings_centers(emb_list_train: list, kmer_size: int) -> np.array:
    """
    Compute the kmer embeddings of length k for each sequence in the dataset
    and the unique kmer embeddings of length k in the dataset



    Args:
        emb_list_train (list): list of embeddings of shape (len_seq, len_aa)
        k (int): length of the kmer

    Returns:
        kmer_list (list): list of kmer embeddings of length k for each sequence in the dataset
        U1 (np.array): unique kmer embeddings of length k in the dataset

    Examples:
        >>> import numpy as np
        >>> from pedata.static.example_data.example_data import RegressionToyDataset
        >>> from pedata.encoding.embeddings_transform import reshape_1hot, kmer_embeddings_centers
        >>> dataset = RegressionToyDataset(needed_encodings=["aa_1hot", "aa_len"]).full_dataset
        >>> aa_len = np.max(dataset["aa_len"])
        >>> nb_aa = 21
        >>> emb_list_1hot = reshape_1hot(dataset["aa_1hot"], aa_len, nb_aa)
        >>> kmer_list, U1 = kmer_embeddings_centers(emb_list_1hot, kmer_size=100)
        >>> print(kmer_list[0]) #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        [[0. 0. 0. ... 0. 0. 0.]
        [1. 0. 0. ... 0. 0. 0.]
        [1. 0. 0. ... 0. 0. 0.]
        ...
        [0. 0. 0. ... 0. 0. 0.]
        [0. 0. 0. ... 0. 0. 0.]
        [0. 0. 0. ... 0. 0. 0.]]
        >>> print(U1.shape)
        (2516, 2100)


    """
    d = emb_list_train[0].shape[1]
    n_train = len(emb_list_train)
    k = kmer_size

    kmer_list = []
    for i in range(0, n_train):
        seq_length = emb_list_train[i].shape[0]
        X = np.zeros((seq_length - k + 1, k * d))
        for j in range(0, seq_length - k + 1):
            X[j, :] = emb_list_train[i][j : j + k, :].reshape(1, -1)
        kmer_list.append(X)

    U = np.unique(np.array(list(itertools.chain.from_iterable(kmer_list))), axis=0)
    # l2 norm
    U1 = U / np.linalg.norm(U, axis=1)[:, None]
    return kmer_list, U1


def rkhs_kmer_embeddings(
    kernel_fn, emb_list: list, kmer_size: int, nystrom_size: int
) -> np.array:
    """
    Compute the RKHS kmer embeddings of length k for each sequence in the dataset
    see https://hal.science/hal-01632912/ for more details

    Args:
        kernel_fn (_type_): kernel function
        emb_list (list): list of embeddings of shape (len_seq, len_aa)
        k_mer_size (int): length of the kmer
        nystrom_size (int): number of centers for the kmeans clustering

    Returns:
        rkhs_kmer_emb_list (np.array): RKHS kmer embeddings of length k for each sequence in the dataset

    Examples:
        >>> import numpy as np
        >>> from pedata.static.example_data.example_data import RegressionToyDataset
        >>> from pedata.encoding.embeddings_transform import linear_kernel, reshape_1hot, rkhs_kmer_embeddings
        >>> dataset = RegressionToyDataset(needed_encodings=["aa_1hot", "aa_len"]).full_dataset
        >>> aa_len = np.max(dataset["aa_len"])
        >>> nb_aa = 21
        >>> emb_list_1hot = reshape_1hot(dataset["aa_1hot"], aa_len, nb_aa)
        >>> rkhs_kmer_emb_list = rkhs_kmer_embeddings(
        ...     linear_kernel,
        ...     emb_list_1hot,
        ...     kmer_size=100,
        ...     nystrom_size=256
        ... )
        >>> print(rkhs_kmer_emb_list.shape)
        (21, 256)
    """

    kmer_list_data, kmer_embedding_unique = kmer_embeddings_centers(emb_list, kmer_size)
    # cluster kmer_embedding_unique
    kmeans = KMeans(n_clusters=nystrom_size, random_state=0, n_init=10).fit(
        kmer_embedding_unique
    )
    # normalize cluster centers
    l2_norm_centers = (
        kmeans.cluster_centers_
        / np.linalg.norm(kmeans.cluster_centers_, axis=1)[:, None]
    )
    # compute kernel matrix
    K = kernel_fn(l2_norm_centers, l2_norm_centers)
    Kn = fractional_matrix_power(K, -0.5)

    # compute rkhs kmer embeddings
    rkhs_kmer_emb_list = np.zeros((len(emb_list), nystrom_size))
    for i in range(len(emb_list)):
        rkhs_kmer_emb_list[i, :] = (
            Kn @ kernel_fn(l2_norm_centers, kmer_list_data[i])
        ).mean(axis=1)
    return rkhs_kmer_emb_list
