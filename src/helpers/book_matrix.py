from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix


@dataclass
class MatrixSplit:
    X_train: csr_matrix
    X_val: csr_matrix
    X_test: csr_matrix
    y: np.ndarray


class S21UserBookMatrixBuilder:
    def __init__(self, ratings_path, users_path) -> None:
        ratings = pd.read_csv(ratings_path)
        ratings = ratings.dropna(subset=["User-ID", "ISBN", "Book-Rating"])

        users = pd.read_csv(users_path, usecols=["User-ID", "Age"])

        self.ratings = ratings[ratings["Book-Rating"] > 0].copy()
        self.users = users.set_index("User-ID")

    # filter top-n books to reduce the number of total calculations
    def _filter_top_books(self, top_n_books: int | None) -> None:
        if top_n_books is None:
            return

        if top_n_books <= 0:
            raise ValueError("top_n_books must be a positive integer")

        counts = self.ratings["ISBN"].value_counts()
        top_n_books = min(top_n_books, counts.shape[0])
        keep_isbn = counts.nlargest(top_n_books).index
        self.ratings = self.ratings[self.ratings["ISBN"].isin(keep_isbn)].copy()


    # converts ratings to CSR sparse matrix
    def _build_sparse_matrix(self) -> "S21UserBookMatrixBuilder":
        ratings = self.ratings
        user_codes = ratings["User-ID"].astype("category")
        book_codes = ratings["ISBN"].astype("category")

        row = user_codes.cat.codes.to_numpy()
        col = book_codes.cat.codes.to_numpy()
        data = ratings["Book-Rating"].to_numpy(dtype="float32")

        self.user_index = user_codes.cat.categories
        self.book_index = book_codes.cat.categories

        self.matrix = coo_matrix(
            (data, (row, col)),
            shape=(self.user_index.size, self.book_index.size),
            dtype=np.float32,
        ).tocsr()

        self.matrix.eliminate_zeros()
        self.matrix.sort_indices()

        return self


    # splits data to train/val/test (by interactions)
    def _split_matrix(self, train_ratio, val_ratio, test_ratio, seed) -> "S21UserBookMatrixBuilder":
        self._build_sparse_matrix()
        matrix = self.matrix

        ratios = np.array([train_ratio, val_ratio, test_ratio], dtype=float)
        if not np.isclose(ratios.sum(), 1.0):
            raise ValueError("Sum of ratios must be 1")

        ratings_coo = matrix.tocoo()
        nnz = ratings_coo.nnz
        indices = np.arange(nnz)

        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

        train_end = int(ratios[0] * nnz)
        val_end = train_end + int(ratios[1] * nnz)

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        def build(idx: np.ndarray) -> csr_matrix:
            if idx.size == 0:
                return csr_matrix(ratings_coo.shape, dtype=ratings_coo.data.dtype)
            return coo_matrix(
                (ratings_coo.data[idx], (ratings_coo.row[idx], ratings_coo.col[idx])),
                shape=ratings_coo.shape,
            ).tocsr()

        self._split_raw = (
            build(train_idx),
            build(val_idx),
            build(test_idx),
        )

        return self


    # builds MatrixSplit
    def build_split(
        self,
        ratios,
        seed=None,
        *,
        min_age: float = 5,
        max_age: float = 100,
        top_n_books: int | None = None,
    ) -> MatrixSplit:
        if len(ratios) != 3 or any(r < 0 for r in ratios):
            raise ValueError("Enter valid ratios")

        train, val, test = ratios
        self._filter_top_books(top_n_books)
        self._split_matrix(train, val, test, seed)

        X_train, X_val, X_test = self._split_raw

        y_full = self.users["Age"].reindex(self.user_index).to_numpy("float64")
        mask = np.isfinite(y_full) & (y_full >= min_age) & (y_full <= max_age)
        y = y_full[mask]

        self.matrix_split = MatrixSplit(
            X_train=X_train[mask],
            X_val=X_val[mask],
            X_test=X_test[mask],
            y=y,
        )

        return self.matrix_split