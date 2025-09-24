import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

class TabularData:
    """
    Tabular version of ImageData.
    """
    
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
    COLUMNS = [
        "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
        "gill-attachment", "gill-spacing", "gill-size", "gill-color",
        "stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring",
        "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color",
        "ring-number", "ring-type", "spore-print-color", "population", "habitat"
    ]

    def __init__(self, data_dir=None, onehot=True, format='N', test_size=0.2, random_state=0):
        """
        data_dir kept for API symmetry; not used (data read from URL).
        format: only 'N' supported for tabular data.
        """
        assert format in ['N'], "MushroomData supports only 'N' (tabular) format."
        self.name = 'mushroom'
        self.onehot_labels = onehot
        self.format = format

        df = pd.read_csv(self.URL, names=self.COLUMNS)

        # Random target function
        y_raw = df['class'].map({'e': 0, 'p': 1}).to_numpy()
        X_raw = df.drop(columns=['class'])

        X_train_df, X_test_df, y_train_raw, y_test_raw = train_test_split(
            X_raw, y_raw, test_size=test_size, random_state=random_state, stratify=y_raw
        )

        # One-hot encode categorical features
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_train = self.encoder.fit_transform(X_train_df)
        X_test  = self.encoder.transform(X_test_df)

        X_train = X_train.astype(np.float32)
        X_test  = X_test.astype(np.float32)

        # Per-feature centering and scaling (train stats)
        mu = X_train.mean(axis=0, keepdims=True)
        sigma = X_train.std(axis=0, keepdims=True)
        #force variance
        sigma[sigma == 0] = 1.0

        X_train = (X_train - mu) / sigma
        X_test  = (X_test  - mu) / sigma

        X_norm = np.linalg.norm(X_train)  # ||X_train||_F
        if X_norm == 0:
            scale = 1.0
        else:
            scale = np.sqrt(X_train.shape[0]) / X_norm
        X_train *= scale
        X_test  *= scale

        if self.onehot_labels:
            n_classes = int(np.max(y_raw)) + 1
            y_train = np.eye(n_classes, dtype=np.float32)[y_train_raw]
            y_test  = np.eye(n_classes, dtype=np.float32)[y_test_raw]
        else:
            y_train = y_train_raw.astype(np.float32)[:, None]
            y_test  = y_test_raw.astype(np.float32)[:, None]

        self.train_X, self.train_y = X_train, y_train
        self.test_X,  self.test_y  = X_test,  y_test

    def get_dataset(self, n, get="train", rng=None, binarize=False, centered=False, feature_normalize=False, **datasetargs):
        """
        Matches your ImageData.get_dataset API:

        - n: number of samples
        - get: "train" or "test"
        - binarize: for labels in {-1, +1} when onehot=True, returns one-hot in {-1, +1};
                    when onehot=False, returns shape (N,1) with labels in {-1, +1}.
        - centered: re-center features (on-the-fly) using the subset mean (not typical for tabular, but preserved)
        - normalize: re-center and per-sample L2-normalize (again, preserved for API symmetry)
        """
        assert int(n) == n and n > 0
        assert get in ["train", "test"]

        full_X, full_y = (self.train_X, self.train_y) if get == "train" else (self.test_X, self.test_y)

        def center_data(X: np.ndarray):
            return X - X.mean(axis=0, keepdims=True)

        X = full_X
        y = full_y

        if binarize:
            y = 2.0 * y - 1.0

        if centered:
            X = center_data(X)

        if feature_normalize:
            X = center_data(X)
            # Per-sample L2 norm across feature dimension
            denom = np.linalg.norm(X, axis=1, keepdims=True)
            denom[denom == 0] = 1.0
            X = X / denom

        # subset
        if rng is None:
            idxs = slice(n)
            X_out, y_out = X[:n].copy(), y[:n].copy()
        else:
            idxs = rng.choice(len(X), size=n, replace=False)
            X_out, y_out = X[idxs].copy(), y[idxs].copy()

        assert len(X_out) == n
        return X_out.astype(np.float32), y_out.astype(np.float32)
