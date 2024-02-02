from pedata import (
    RegressionToyDataset,
    ClassificationToyDataset,
)


def test_regr_dataset_get_full_dataset():
    """Test that the Regression Dataset has all needed attributes and encodings"""
    TestRegressionData = RegressionToyDataset()
    assert hasattr(TestRegressionData, "train")
    assert hasattr(TestRegressionData, "test")
    assert hasattr(TestRegressionData, "train_test_split_dataset")
    assert hasattr(TestRegressionData, "full_dataset")
    dataset = TestRegressionData.full_dataset
    assert sorted(list(dataset.features.keys())) == sorted(
        [
            "aa_1gram",
            "aa_1hot",
            "aa_len",
            "aa_mut",
            "aa_seq",
            "aa_unirep_1900",
            "aa_unirep_final",
            "target_kcat_per_kmol",
        ]
    )


def test_class_dataset_get_full_dataset():
    """Test that the Classification Dataset has all needed attributes and encodings"""
    TestClassificationData = ClassificationToyDataset()
    assert hasattr(TestClassificationData, "train")
    assert hasattr(TestClassificationData, "test")
    assert hasattr(TestClassificationData, "train_test_split_dataset")
    assert hasattr(TestClassificationData, "full_dataset")
    dataset = TestClassificationData.full_dataset
    assert sorted(list(dataset.features.keys())) == sorted(
        [
            "aa_mut",
            "target_high_low",
            "aa_seq",
            "aa_unirep_1900",
            "aa_unirep_final",
            "aa_1gram",
            "aa_1hot",
            "aa_len",
        ]
    )
