import numpy as np
from pedata.visual import (
    plot_target_distribution,
    plot_target_distributions,
)


def test_plot_target_distribution():
    """Test that the plot_target_distribution function works"""
    target = np.random.normal(0, 1, 1000)
    plot_target_distribution(
        target, label="pytest_norm_random_target", savedir="code_health"
    )


def test_plot_target_distributions():
    """Test that the plot_target_distributions function works with multiple targets"""
    target_1 = np.random.normal(0, 1, 1000)
    target_2 = np.random.normal(0, 1, 1000)
    targets = {
        "target_1": target_1,
        "target_2": target_2,
    }
    plot_target_distributions(
        targets, label="pytest_2_norm_random_targets", savedir="code_health"
    )
