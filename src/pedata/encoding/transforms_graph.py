from typing import Optional, Union, Iterable
import jax.numpy as jnp
import numpy as np
import datasets as ds
import pandas as pd

# from rdkit import Chem

# FIXME: write more tests for this module


def get_num_nodes_edges(adjacency_lists: list[jnp.ndarray]) -> tuple:
    """Get the number of nodes and edges for each graph.
        This function takes a list of adjacency lists and calculates the number of nodes and edges for each graph, as well as the maximum number of edges among all graphs.
        Each element l[i] is a tuple of two lists:
        - l[i][0] represents a list of source nodes.
        - l[i][1] represents a list of target nodes.

    Args:
        adjacency_lists (list): A list of adjacency lists.

    Returns:
        tuple: A tuple of three arrays: num_nodes, num_edges, max_edges.
        - num_nodes[i] represents the number of nodes in the i-th graph.
        - num_edges[i] represents the number of edges in the i-th graph.
        - max_edges represents the maximum number of edges among all graphs.

    Example:
        >>> adj_list = [
        ...     jnp.array(([0, 1, 0], [2, 1, 3])),
        ...     jnp.array(([0, 1], [0, 1])),
        ...     jnp.array(([0], [1]))]
        >>> num_nodes, num_edges, max_edges = get_num_nodes_edges(adj_list)
        >>> print(num_nodes)
        [4 2 2]
        >>> print(num_edges)
        [3 2 1]
        >>> print(max_edges)
        3
    """

    if isinstance(adjacency_lists, str):
        raise TypeError(
            "Invalid input: Input a should be a list of adjacency lists, not a string"
        )

    # Verify adjacency lists
    if not isinstance(adjacency_lists, Iterable) or len(adjacency_lists) == 0:
        raise TypeError("Invalid input: Input a non-empty list of adjacency lists")

    num_nodes = []  # Stores the number of nodes for each graph
    num_edges = []  # Stores the number of edges for each graph

    for adj_list in adjacency_lists:
        # Verify shape of an adjacency list
        if adj_list.shape[0] != 2:
            raise ValueError(
                f"Expected adjacency list to be of shape (2, num_edges), but got {adj_list.shape}"
            )

        # Get number of nodes and edges for each graph
        num_nodes.append(len(jnp.unique(jnp.array(adj_list))))
        num_edges.append(len(adj_list[0]))

    # Convert lists to NumPy arrays
    num_nodes = jnp.array(num_nodes, dtype=np.int32)
    num_edges = jnp.array(num_edges, dtype=np.int32)

    # Find maximum number of edges
    max_edges = num_edges.max()

    return num_nodes, num_edges, max_edges


def filter_adj_list_by_edge_type(
    adj_list: jnp.ndarray,
    edge_type: jnp.ndarray,
    unique_edge_types: jnp.ndarray,
) -> list[jnp.ndarray]:
    """Filter the adjacency list by edge type.

    This function takes an adjacency list, along with the corresponding edge types and unique edge types.
    It filters the adjacency list based on the edge types and returns a list of adjacency lists, where each element l[i]
    contains the edges of type unique_edge_types[i].

    Args:
        adj_list (jnp.ndarray): The adjacency list.
        edge_type (jnp.ndarray): The edge types.
        unique_edge_types (jnp.ndarray): Unique edge types. This is used to filter the adjacency list with a fixed order of edge types.

    Returns:
        list: A list of adjacency lists, where each element l[i] is a list of edges of type unique_edge_types[i].

    Example:
        >>> adj_list = [[0, 1], [1, 2], [2, 0]]
        >>> edge_types = [0, 1, 2]
        >>> unique_edge_types = [0, 1, 2]
        >>> result = filter_adj_list_by_edge_type(
        ...     jnp.array(adj_list), jnp.array(edge_types), jnp.array(unique_edge_types)
        ... )
        >>> print(result) #doctest: +NORMALIZE_WHITESPACE
        [Array([[0, 1]], dtype=int32), Array([[1, 2]], dtype=int32), Array([[2, 0]], dtype=int32)]
    """

    # Verify adjacency list
    if len(adj_list) == 0:
        raise ValueError("Invalid input: Input a non-empty adjacency list")

    # Verify list of edge types
    if len(edge_type) == 0:
        raise ValueError("Invalid input: Input a non-empty list of edge types")

    # Verify shape of an adjacency list
    if adj_list.shape[1] != 2:
        raise ValueError(
            f"Expected adjacency list to have shape (num_edges, 2), but got {adj_list.shape}"
        )

    # Get the number of edges in an adjacency list
    num_edges = adj_list.shape[0]

    # filter the adjacency list based on edge types
    edge_type_filtered_adj_list = [
        adj_list[edge_type[:num_edges] == et] for et in unique_edge_types
    ]

    return edge_type_filtered_adj_list


def adj_list_to_adjmatr(
    adj: jnp.ndarray,
    num_nodes: int,
    num_edges: int,
    fill_values: Union[int, float, jnp.ndarray] = 1,
    pad_num_nodes: int = None,
    pad_num_edges: int = None,
    pad_value: Union[int, float] = 0,
    directed: bool = False,
) -> jnp.ndarray:
    """Convert a single adjacency list to an adjacency matrix.

    This function takes a single adjacency list and converts it into an adjacency matrix representation. The adjacency list should be a 2D array where the first row contains the source nodes and the second row contains the destination nodes. The resulting adjacency matrix is a 2D square array with dimensions based on the number of nodes in the graph.

    Args:
        adj (jnp.ndarray): The adjacency list, where the first row contains the source nodes and the second row contains the destination nodes.
        num_nodes (int): Number of nodes in the graph.
        num_edges (int): Number of edges in the graph. Additional entries in `adj` are assumed to be padding and ignored.
        fill_values (Union[int, float, jnp.ndarray], optional): Fill values for the adjacency matrix. If a scalar, the same value is used for all edges. If a vector, the values are used for the edges in the order they appear in `adj`. Defaults to 1.
        pad_num_nodes (int, optional): Size to pad the adjacency matrix to. Defaults to None, which means no padding.
        pad_num_edges (int, optional): Ignored.
        pad_value (Union[int, float], optional): Value to use for padding in the adjacency matrix. Defaults to 0.
        directed (bool, optional): Whether the graph is directed. Defaults to False.

    Raises:
        ValueError: If `adj` does not have the shape (2, num_edges).
        ValueError: If `fill_values` is a vector and its length does not match `num_edges`.

    Returns:
        jnp.ndarray: The adjacency matrix representing the graph.

    Example:
        >>> adj_list = jnp.array([[0, 1, 2], [1, 2, 0]])
        >>> result = adj_list_to_adjmatr(
        ...     adj_list,
        ...     3, # num_nodes
        ...     3, # num_edges
        ...     1, # fill_values
        ...     None, # pad_num_nodes
        ...     None, # pad_num_edges
        ...     0, # pad_value
        ...     False, # directed
        ... )
        >>> print(result) # doctest: +NORMALIZE_WHITESPACE
        [[0. 1. 1.]
            [1. 0. 1.]
            [1. 1. 0.]]

        >>> adj_list = jnp.array([[0, 1], [1, 2]])
        >>> result = adj_list_to_adjmatr(adj_list, 4, 2, jnp.array([2, 3]), 5, None, 0, True)
        >>> print(result) # doctest: +NORMALIZE_WHITESPACE
        [[0. 2. 0. 0. 0.]
            [0. 0. 3. 0. 0.]
            [0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0.]]

    """

    # Verify adjacency list
    if len(adj) == 0:
        raise ValueError("Invalid input: Input a non-empty adjacency list")

    # Ensure the adjacency list has the expected shape
    if adj.shape[0] != 2:
        raise ValueError(
            f"Expected adjacency list to be of shape (2, num_edges), but got {adj.shape}"
        )

    # Transpose adjacency list array to get edges in the first dimension
    # this was how it was done in the original code, transposing is the easiest fix
    adj = adj.T

    # Initialize adjacency matrix with pad_value
    if pad_num_nodes is None:
        pad_num_nodes = num_nodes
    rval = jnp.full((pad_num_nodes, pad_num_nodes), pad_value, dtype=jnp.float32)

    # Check if fill_values is not a single value (int or float)
    if not isinstance(fill_values, (int, float)):
        # Assign the value at index i and raise an error if the length of fill_values doesn't match the number of edges
        if len(fill_values) != num_edges:
            raise ValueError(
                f"Expected fill_values to have length {num_edges}, but got {len(fill_values)}"
            )

    for i, row in enumerate(adj):
        # Extract the source and destination nodes from the current edge
        src, dest = row[0], row[1]

        values = fill_values  # temporarily stores fill_values

        # If fill_values is iterable, select the value at index i
        if isinstance(fill_values, Iterable):
            values = fill_values[i]

        # Set the adjacency matrix entry for the edge between the source and destination nodes
        rval = rval.at[src, dest].set(values)

        # If the graph is undirected, set the adjacency matrix value for the destination-source edge as well
        if not directed:
            rval = rval.at[dest, src].set(values)

    return rval


def adj_list_to_incidence(
    adj: jnp.ndarray,
    num_nodes: int,
    num_edges: int,
    fill_values: Union[int, float, jnp.ndarray] = 1,
    pad_num_nodes: int = None,
    pad_num_edges: int = None,
    pad_value: Union[int, float] = 0,
    directed: bool = False,
) -> jnp.ndarray:
    """Convert a single adjacency list to an incidence matrix.

    Args:
        adj (jnp.ndarray): The adjacency list, where the first row contains the source nodes and the second row contains the destination nodes.
        num_nodes (int): Number of nodes.
        num_edges (int): Number of edges. Additional entries in `adj` and `edge_type` are assumed to be padding and ignored.
        fill_values (Union[int, float, jnp.ndarray], optional): Fill values for the incidence matrix. If a scalar, the same value is used for all edges. If a vector, the values are used for the edges in the order they appear in `adj`. Defaults to 1.
        pad_num_nodes (int, optional): Size to pad node dimension to. Defaults to None, which means no padding.
        pad_num_edges (int, optional): Size to pad edge dimension to. Defaults to None, which means no padding.
        pad_value (Union[int, float], optional): Value to use for padding. Defaults to 0.
        directed (bool, optional): Whether the graph is directed. Defaults to False, in which case the `fill_values` are used for both directions. If True, the `-fill_values` are used for the source node and `fill_values` are used for the destination node.

    Raises:
        ValueError: If `fill_values` is a vector and its length does not match `num_edges`.

    Returns:
        jnp.ndarray: The incidence matrix.
    """
    if adj.shape[0] != 2:
        raise ValueError(
            f"Expected adjacency list to be of shape (2, num_edges), but got {adj.shape}"
        )
    # Transpose adjacency list array to get edges in the first dimension
    # this was how it was done in the original code, transposing is the easiest fix
    adj = adj.T

    if pad_num_nodes is None:
        pad_num_nodes = num_nodes
    if pad_num_edges is None:
        pad_num_edges = num_edges
    # Initialize incidence matrix with pad_value
    rval = jnp.full((pad_num_nodes, pad_num_edges), pad_value, dtype=jnp.float32)

    for i, (src, dest) in enumerate(adj):
        rval = rval.at[dest, i].set(fill_values)
        rval = rval.at[src, i].set(-fill_values if directed else fill_values)

    return rval


def adj_list_conversion(
    adj_list_to_matrix_fn: callable,
    adj: jnp.ndarray,
    num_nodes: int,
    num_edges: int,
    fill_values: Union[int, float, jnp.ndarray] = 1,
    pad_num_nodes: int = None,
    pad_num_edges: int = None,
    pad_value: Union[int, float] = 0,
    edge_type: Optional[jnp.ndarray] = None,
    unique_edge_types: Optional[jnp.ndarray] = None,
    directed: bool = False,
) -> jnp.ndarray:
    """Convert a single adjacency list.

    Args:
        adj (jnp.ndarray): The adjacency list of shape (num_edges, 2).
        num_nodes (int): Number of nodes.
        num_edges (int): Number of edges. Additional entries in `adj` and `edge_type` are assumed to be padding and ignored.
        fill_values (Union[int, float, jnp.ndarray], optional): Fill values for the adjacency matrix. If a scalar, the same value is used for all edges. If a vector, the values are used for the edges in the order they appear in `adj`. Defaults to 1.
        pad_num_nodes (int, optional): Size to pad node dimension to. Defaults to None, which means no padding.
        pad_num_edges (int, optional): Size to pad edge dimension to. Defaults to None, which means no padding.
        pad_value (Union[int, float], optional): Value to use for padding. Defaults to 0.
        edge_type (jnp.ndarray, optional): Types of edges. Defaults to None. If not None, the output is an adjacency matrix for each edge type. In this case, `unique_edge_types` defines the sort order of the edge types and must be provided.
        unique_edge_types (jnp.ndarray, optional): Unique edge types. If edge_type is not None, this must be provided. Defaults to None.
        directed (bool, optional): Whether the graph is directed. Defaults to False.

    Raises:
        ValueError: If `fill_values` is a vector and its length does not match `num_edges`.

    Returns:
        jnp.ndarray: The adjacency matrix/matrices. If `edge_type` is not None, adjacency matrices for the different edge types are stacked along the first axis.
    """
    if pad_num_nodes is None:
        pad_num_nodes = num_nodes
    if pad_num_edges is None:
        pad_num_edges = num_edges
    # adj = jax.lax.dynamic_slice(adj, (0, 0), (num_edges, 2))
    # adj = adj.at[idx].get()

    if edge_type is not None:
        assert unique_edge_types is not None
        edge_type_filtered_adj_list = [
            adj[edge_type[:num_edges] == et] for et in unique_edge_types
        ]
        adj_matrices = jnp.array(
            [
                adj_list_to_matrix_fn(
                    edge_type_filtered_adj_list[i],
                    num_nodes,
                    len(edge_type_filtered_adj_list[i]),
                    fill_values,
                    pad_num_nodes,
                    pad_num_edges,
                    pad_value,
                    directed,
                )
                for i, et in enumerate(unique_edge_types)
            ]
        )
        return adj_matrices

    return adj_list_to_matrix_fn(
        adj,
        num_nodes,
        num_edges,
        fill_values,
        pad_num_nodes,
        pad_num_edges,
        pad_value,
        directed,
    )


def bnd_count_atm_count(df: Union[ds.Dataset, pd.DataFrame]) -> dict[str, list[int]]:
    """A function to count the number of atoms and bonds for each molecule in a dataset.

    Args:
        df (Union[ds.Dataset, pd.DataFrame]): A dataset or dataframe containing the "bnd_idcs" column.

    Returns:
        dict[str, list[int]]: A dictionary containing the number of atoms and bonds for each molecule.
    """
    num_nodes, num_edges, max_edges = get_num_nodes_edges(df["bnd_idcs"])
    return {"atm_count": num_nodes.tolist(), "bnd_count": num_edges.tolist()}


def atm_adj(df: Union[ds.Dataset, pd.DataFrame]) -> dict[str, list[jnp.ndarray]]:
    """A function to convert the atom adjacency list to an adjacency matrix. Symmetric, since the molecule graph is undirected.

    Args:
        df (Union[ds.Dataset, pd.DataFrame]): A dataset or dataframe containing the "bnd_idcs", "atm_count", and "bnd_count" columns.

    Returns:
        dict[str, list[jnp.ndarray]]: A dictionary containing the adjacency matrices for each molecule.
    """
    df = df.with_format("jax")
    return {
        "atm_adj": [
            adj_list_to_adjmatr(
                df["bnd_idcs"][i], df["atm_count"][i], df["bnd_count"][i]
            ).tolist()
            for i in range(len(df))
        ]
    }


def atm_bnd_incid(df: Union[ds.Dataset, pd.DataFrame]) -> dict[str, list[jnp.ndarray]]:
    """A function to convert the atom-bond incidence list to an incidence matrix.

    Args:
        df (Union[ds.Dataset, pd.DataFrame]): A dataset or dataframe containing the "bnd_idcs", "atm_count", and "bnd_count" columns.

    Returns:
        dict[str, list[jnp.ndarray]]: A dictionary containing the incidence matrices for each molecule.
    """
    df = df.with_format("jax")
    return {
        "atm_bnd_incid": [
            adj_list_to_incidence(
                df["bnd_idcs"][i], df["atm_count"][i], df["bnd_count"][i]
            ).tolist()
            for i in range(len(df))
        ]
    }


def return_prob_feat(
    nb_iter: int, df: Union[ds.Dataset, pd.DataFrame]
) -> dict[str, list[jnp.ndarray]]:
    """A function to compute the return probability feature from the RetGk kernel model. See Zhang et al. (2018) "RetGK: Graph Kernels based on Return Probabilities of Random Walks", https://arxiv.org/abs/1809.02670

    Args:
        nb_iter (int): The number of iterations to run the random walk for.
        df (Union[ds.Dataset, pd.DataFrame]): A dataset or dataframe containing the "atm_adj" column.

    Returns:
        dict[str, list[jnp.ndarray]]: A dictionary containing the return probability feature for each molecule.
    """
    adj_matrices = df["atm_adj"]
    T = []
    for i in range(len(adj_matrices)):
        P = np.diag(np.sum(adj_matrices[i], axis=1) ** -1).dot(adj_matrices[i])
        Ptld = 0.5 * (np.eye(len(P)) + P)
        H = np.eye(len(P))
        U = np.zeros((len(P), nb_iter))
        for j in range(nb_iter):
            H = H.dot(Ptld)
            U[:, j] = np.diag(H)
        T.append(U.tolist())
    return {"atm_retprob100": T}
