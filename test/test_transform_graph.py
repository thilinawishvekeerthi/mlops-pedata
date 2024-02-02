import jax.numpy as jnp
import pedata.encoding.transforms_graph as tg
import datasets as ds
import pytest


def test_get_num_nodes_edges():
    # Test case 1: Single graph with 4 nodes and 3 edges
    adj_list = [jnp.array(([0, 1, 0], [2, 1, 3]))]
    num_nodes, num_edges, max_edges = tg.get_num_nodes_edges(adj_list)
    assert jnp.array_equal(
        num_nodes, jnp.array([4])
    ), f"Test case 1: Expected num_nodes = [4], but got {num_nodes}"
    assert jnp.array_equal(
        num_edges, jnp.array([3])
    ), f"Test case 1: Expected num_edges = [3], but got {num_edges}"
    assert max_edges == 3, f"Test case 1: Expected max_edges = 3, but got {max_edges}"

    # Test case 2: Single graph with 2 nodes and 2 edges
    adj_list = [jnp.array(([0, 1], [0, 1]))]
    num_nodes, num_edges, max_edges = tg.get_num_nodes_edges(adj_list)
    assert jnp.array_equal(
        num_nodes, jnp.array([2])
    ), f"Test case 2: Expected num_nodes = [2], but got {num_nodes}"
    assert jnp.array_equal(
        num_edges, jnp.array([2])
    ), f"Test case 2: Expected num_edges = [2], but got {num_edges}"
    assert max_edges == 2, f"Test case 2: Expected max_edges = 2, but got {max_edges}"

    # Test case 3: 2 graphs with different nodes and edges
    adj_list = [jnp.array(([0, 1, 0], [2, 1, 3])), jnp.array(([0, 1], [0, 1]))]
    num_nodes, num_edges, max_edges = tg.get_num_nodes_edges(adj_list)
    assert jnp.all(
        num_nodes == jnp.array([4, 2])
    ), f"Test case 3: Expected num_nodes = [4, 2], but got {num_nodes}"
    assert jnp.all(
        num_edges == jnp.array([3, 2])
    ), f"Test case 3: Expected num_edges = [3, 2], but got {num_edges}"
    assert max_edges == 3, f"Test case 3: Expected max_edges = 3, but got {max_edges}"

    # Test case 4: Multiple graphs with different numbers of nodes and edges
    adj_list = [
        jnp.array(([0, 1, 0], [2, 1, 3])),
        jnp.array(([0, 1], [0, 1])),
        jnp.array(([0], [1])),
    ]
    num_nodes, num_edges, max_edges = tg.get_num_nodes_edges(adj_list)
    assert jnp.array_equal(
        num_nodes, jnp.array([4, 2, 2])
    ), f"Test case 4: Expected num_nodes = [4, 2, 2], but got {num_nodes}"
    assert jnp.array_equal(
        num_edges, jnp.array([3, 2, 1])
    ), f"Test case 4: Expected num_edges = [3, 2, 1], but got {num_edges}"
    assert (
        max_edges == 3
    ), f"Test case 4: Expected max_edges = [4, 2, 2], but got {max_edges}"

    # Test case 5: Empty adjacency list
    adj_list = []
    with pytest.raises(TypeError):
        tg.get_num_nodes_edges(adj_list)

    # Test case 6: Incorrect shape of adjacency list
    adj_list = [jnp.array(([0, 1], [1, 2], [3, 4]))]
    with pytest.raises(ValueError):
        tg.get_num_nodes_edges(adj_list)


def test_filter_adj_list_by_edge_type():
    # Test case 1: Multiple graphs with multiple edge types
    adj_lists = [([[0, 2], [1, 1], [0, 3]]), ([[0, 0], [1, 1]])]
    edge_types = [[0, 1, 2], [0, 1]]
    filter_adj_list_by_edge_type = [
        tg.filter_adj_list_by_edge_type(
            jnp.array(adj), jnp.array(types), jnp.unique(jnp.array(types))
        )
        for adj, types in zip(adj_lists, edge_types)
    ]
    expected_filter_adj_list_by_edge_type = [
        [[[0, 2]], [[1, 1]], [[0, 3]]],
        [[[0, 0]], [[1, 1]]],
    ]
    for adj_list, expected_adj_list in zip(
        filter_adj_list_by_edge_type, expected_filter_adj_list_by_edge_type
    ):
        for adj, expected_adj in zip(adj_list, expected_adj_list):
            assert jnp.array_equal(
                adj, expected_adj
            ), f"Expected {expected_adj}, but got {adj}"

    # Test case 2: Single graph with multiple edge types
    adj_list = [[0, 1], [1, 2], [2, 0]]
    edge_types = [0, 1, 2]
    unique_edge_types = [0, 1, 2]
    result = tg.filter_adj_list_by_edge_type(
        jnp.array(adj_list), jnp.array(edge_types), jnp.array(unique_edge_types)
    )
    expected_result = [
        jnp.array([[0, 1]]),
        jnp.array([[1, 2]]),
        jnp.array([[2, 0]]),
    ]
    assert len(result) == len(
        expected_result
    ), f"Expected {len(expected_result)} adjacency lists, but got {len(result)}"
    for adj, expected_adj in zip(result, expected_result):
        assert jnp.array_equal(
            adj, expected_adj
        ), f"Expected {expected_adj}, but got {adj}"

    # Test case 3: Invalid input (Empty adjacency list)
    adj_list = []
    edge_types = [0, 1, 2]
    unique_edge_types = [0, 1, 2]
    with pytest.raises(ValueError):
        tg.filter_adj_list_by_edge_type(
            jnp.array(adj_list), jnp.array(edge_types), jnp.array(unique_edge_types)
        )

    # Test case 4: Invalid input (Empty list of edge types )
    adj_list = [[0, 1], [1, 2], [2, 0]]
    edge_types = []
    unique_edge_types = [0, 1, 2]
    with pytest.raises(ValueError):
        tg.filter_adj_list_by_edge_type(
            jnp.array(adj_list), jnp.array(edge_types), jnp.array(unique_edge_types)
        )

    # Test case 5: Graph with missing edge types
    adj_list = [[0, 1], [1, 2], [2, 0]]
    edge_types = [0, 1, 0]  # Missing edge type 2
    unique_edge_types = [0, 1, 2]
    result = tg.filter_adj_list_by_edge_type(
        jnp.array(adj_list), jnp.array(edge_types), jnp.array(unique_edge_types)
    )
    expected_result = [
        jnp.array([[0, 1], [2, 0]]),
        jnp.array([[1, 2]]),
        jnp.array([]),
    ]
    assert len(result) == len(
        expected_result
    ), f"Expected {len(expected_result)} adjacency lists, but got {len(result)}"
    for adj, expected_adj in zip(result, expected_result):
        if len(adj) == 0 and len(expected_adj) == 0:
            continue  # Skip comparison for empty arrays
        assert jnp.array_equal(
            adj, expected_adj
        ), f"Expected {expected_adj}, but got {adj}"

    # Test case 6: Graph with additional edge types
    adj_list = [[0, 1], [1, 2], [2, 0]]
    edge_types = [0, 1, 2, 3]  # Extra edge type 3
    unique_edge_types = [0, 1, 2]
    result = tg.filter_adj_list_by_edge_type(
        jnp.array(adj_list), jnp.array(edge_types), jnp.array(unique_edge_types)
    )
    expected_result = [
        jnp.array([[0, 1]]),
        jnp.array([[1, 2]]),
        jnp.array([[2, 0]]),
    ]
    assert len(result) == len(
        expected_result
    ), f"Expected {len(expected_result)} adjacency lists, but got {len(result)}"
    for adj, expected_adj in zip(result, expected_result):
        assert jnp.array_equal(
            adj, expected_adj
        ), f"Expected {expected_adj}, but got {adj}"

    # Test case 7: Large graph with multiple edge types
    adj_list = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [0, 5], [5, 6], [6, 7], [7, 0]]
    edge_types = [0, 1, 2, 1, 0, 1, 2, 1, 0]
    unique_edge_types = [0, 1, 2]
    result = tg.filter_adj_list_by_edge_type(
        jnp.array(adj_list), jnp.array(edge_types), jnp.array(unique_edge_types)
    )
    expected_result = [
        jnp.array([[0, 1], [4, 0], [7, 0]]),
        jnp.array([[1, 2], [3, 4], [0, 5], [6, 7]]),
        jnp.array([[2, 3], [5, 6]]),
    ]
    assert len(result) == len(
        expected_result
    ), f"Expected {len(expected_result)} adjacency lists, but got {len(result)}"
    for adj, expected_adj in zip(result, expected_result):
        assert jnp.array_equal(
            adj, expected_adj
        ), f"Expected {expected_adj}, but got {adj}"


def test_adj_list_to_adjmatr():
    # Test case 1: Basic test with single edge
    adj_list = jnp.array([[0], [1]])
    num_nodes = 2
    num_edges = 1
    fill_values = 2
    result = tg.adj_list_to_adjmatr(adj_list, num_nodes, num_edges, fill_values)
    expected_result = jnp.array([[0, 2], [2, 0]])
    assert jnp.array_equal(
        result, expected_result
    ), f"Expected {expected_result}, but got {result}"

    # Test case 2: Basic test with multiple edges
    adj_list = jnp.array([[0, 1, 2], [1, 2, 0]])
    num_nodes = 3
    num_edges = 3
    fill_values = 1
    result = tg.adj_list_to_adjmatr(adj_list, num_nodes, num_edges, fill_values)
    expected_result = jnp.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    assert jnp.array_equal(
        result, expected_result
    ), f"Expected {expected_result}, but got {result}"

    # Test case 3: Directed graph with with a list of fill_values
    adj_list = jnp.array([[0, 1], [1, 2]])
    num_nodes = 4
    num_edges = 2
    fill_values = jnp.array([2, 3])
    directed = True
    result = tg.adj_list_to_adjmatr(
        adj_list, num_nodes, num_edges, fill_values, directed=directed
    )
    expected_result = jnp.array(
        [[0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    )
    assert jnp.array_equal(
        result, expected_result
    ), f"Expected {expected_result}, but got {result}"

    # Test case 4: Directed graph with a single fill_value (int)
    adj_list = jnp.array([[0, 1], [1, 2]])
    fill_values = 2
    pad_num_nodes = 5
    pad_num_edges = None
    pad_value = 0
    directed = True
    result = tg.adj_list_to_adjmatr(
        adj_list,
        num_nodes,
        num_edges,
        fill_values,
        pad_num_nodes,
        pad_num_edges,
        pad_value,
        directed,
    )
    expected_result = jnp.array(
        [
            [0.0, 2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    assert jnp.array_equal(
        result, expected_result
    ), f"Expected {expected_result}, but got {result}"

    # Test case 5: Padded adjacency matrix
    adj_list = jnp.array([[0, 1, 2], [1, 2, 0]])
    num_nodes = 3
    num_edges = 3
    fill_values = 1
    pad_num_nodes = 5
    pad_value = -1
    result = tg.adj_list_to_adjmatr(
        adj_list, num_nodes, num_edges, fill_values, pad_num_nodes, pad_value=pad_value
    )
    expected_result = jnp.array(
        [
            [-1, 1, 1, -1, -1],
            [1, -1, 1, -1, -1],
            [1, 1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
        ]
    )
    assert jnp.array_equal(
        result, expected_result
    ), f"Expected {expected_result}, but got {result}"

    # Test case 6: Empty adjacency list
    adj_list = jnp.array([[], []])
    num_nodes = 0
    num_edges = 0
    fill_values = 1
    result = tg.adj_list_to_adjmatr(adj_list, num_nodes, num_edges, fill_values)
    expected_result = jnp.array([])
    for result, expected_result in zip(result, expected_result):
        if len(result) == 0 and len(expected_result) == 0:
            continue  # Skip comparison for empty arrays
        assert jnp.array_equal(
            result, expected_result
        ), f"Expected {expected_result}, but got {result}"


def test_atm_adj():
    adj_list = ds.Dataset.from_dict(
        {
            "bnd_idcs": [
                jnp.array([[0, 1, 0], [2, 1, 3]]),
                jnp.array([[0, 1], [0, 1]]),
            ]
        }
    )
    atm_adj = tg.atm_adj(
        adj_list.with_format("jax").map(tg.bnd_count_atm_count, batched=True)
    )
    expected_atm_adj = [
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        [[1.0, 0.0], [0.0, 1.0]],
    ]
    for i, (adj, expected_adj) in enumerate(zip(atm_adj["atm_adj"], expected_atm_adj)):
        assert jnp.array_equal(
            adj, expected_adj
        ), f"Expected {expected_adj}, but got {adj} in element {i}"
