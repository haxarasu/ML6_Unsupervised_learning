from __future__ import annotations
from typing import Optional
import numpy as np
from sklearn.decomposition import PCA


class FaceExpressionPCA:
    # configure PCA for face compression
    def __init__(self, n_components: int = 50, random_state: Optional[int] = 42) -> None:
        self.n_components = n_components
        self.random_state = random_state
        self._model: Optional[PCA] = None


    # return fitted PCA model
    @property
    def model(self) -> PCA:
        if self._model is None:
            raise ValueError("PCA model is not fitted.")

        return self._model


    # fit PCA on flattened images
    def fit(self, X: np.ndarray) -> None:
        self._model = PCA(n_components=self.n_components, random_state=self.random_state)
        self._model.fit(X)


    # project data to latent space
    def transform(self, X: np.ndarray) -> np.ndarray:
        result = self.model.transform(X)

        return result


    # reconstruct data from latent vectors
    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        result = self.model.inverse_transform(Z)

        return result