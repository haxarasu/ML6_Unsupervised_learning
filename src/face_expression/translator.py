from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from model import FaceExpressionPCA


class EmotionTranslator:
    # store PCA model and label vocabulary
    def __init__(self, model: FaceExpressionPCA, label_names: Optional[List[str]] = None) -> None:
        self.model = model
        self.label_names = label_names or []
        self.label_to_index = {name: idx for idx, name in enumerate(self.label_names)}
        self.latent_means: Dict[int, np.ndarray] = {}


    # compute mean latent vectors for each label
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        Z = self.model.transform(X)
        self.latent_means = {}
        for label in np.unique(y):
            self.latent_means[int(label)] = Z[y == label].mean(axis=0)


    # encode one image vector to latent space
    def encode(self, x: np.ndarray) -> np.ndarray:
        flat = x.reshape(1, -1)
        result = self.model.transform(flat)[0]

        return result


    # decode one latent vector to image space
    def decode(self, z: np.ndarray, image_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        x = self.model.inverse_transform(z.reshape(1, -1))[0]
        if image_shape:
            result = x.reshape(image_shape)

            return result

        return x


    # shift latent representation from source emotion to target emotion
    def translate(
        self,
        x: np.ndarray,
        source_label: Union[int, str],
        target_label: Union[int, str],
        image_shape: Optional[Tuple[int, int]] = None,
        clip: bool = True,
        alpha: float = 1.0,
    ) -> np.ndarray:
        src = self._label_index(source_label)
        tgt = self._label_index(target_label)
        z = self.encode(x)
        direction = self.latent_means[tgt] - self.latent_means[src]
        z_new = z + alpha * direction
        x_new = self.decode(z_new, image_shape=image_shape)
        if clip:
            result = np.clip(x_new, 0.0, 1.0)

            return result

        return x_new


    # convert label name or index to integer index
    def _label_index(self, label: Union[int, str]) -> int:
        if isinstance(label, int):
            result = label

            return result
        if label not in self.label_to_index:
            raise KeyError(f"Unknown label: {label}")

        return self.label_to_index[label]