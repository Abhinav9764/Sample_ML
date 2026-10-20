import pandas as pd

from preprocessing import build_preprocessor, preprocess_data


def sample_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "survived": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "age": [22.0, None, 38.0, 26.0, 35.0, 27.0, 54.0, 2.0, 27.0, 14.0],
            "fare": [7.25, 71.28, 7.92, 53.1, 8.05, 11.13, 51.86, 21.07, 11.13, 30.07],
            "sex": ["male", "female"] * 5,
            "class": pd.Categorical(["Third", "First"] * 5),
            "alone": [False, False, True, False, True, True, False, False, True, True],
        }
    )


def test_preprocessor_handles_all_supported_column_types() -> None:
    features = sample_data().drop(columns="survived")
    transformed = build_preprocessor(features).fit_transform(features)
    assert transformed.shape[0] == len(features)
    assert transformed.shape[1] > 2


def test_preprocess_data_splits_and_transforms_data() -> None:
    x_train, x_test, y_train, y_test, _ = preprocess_data(sample_data())
    assert x_train.shape[0] == len(y_train)
    assert x_test.shape[0] == len(y_test)
    assert x_train.shape[1] == x_test.shape[1]
