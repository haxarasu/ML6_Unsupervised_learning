from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
from PIL import Image


@dataclass
class DatasetOutput:
    X: np.ndarray
    y: np.ndarray
    image_shape: Tuple[int, int]
    label_names: List[str]


class FaceExpressionDataset:
    # store dataset configuration
    def __init__(
        self,
        data_dir: str = "datasets/data/face_expression",
        split: str = "train",
        image_size: Tuple[int, int] = (48, 48),
        grayscale: bool = True,
        normalize: bool = True,
        max_per_class: Optional[int] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.grayscale = grayscale
        self.normalize = normalize
        self.max_per_class = max_per_class


    # load images from class subfolders and return flattened arrays
    def load(self) -> DatasetOutput:
        data = self._load_from_folders(self.data_dir / "images" / self.split)

        return data


    # read images from class folders
    def _load_from_folders(self, path: Path) -> DatasetOutput:
        label_paths = sorted([p for p in path.iterdir() if p.is_dir()])
        label_names = [p.name for p in label_paths]
        label_to_index = {name: idx for idx, name in enumerate(label_names)}

        images: List[np.ndarray] = []
        labels: List[int] = []
        counts: Dict[int, int] = {idx: 0 for idx in range(len(label_names))}

        for label_path in label_paths:
            label_idx = label_to_index[label_path.name]
            for image_path in self._iter_images(label_path):
                if self.max_per_class and counts[label_idx] >= self.max_per_class:
                    break
                images.append(self._read_image(image_path))
                labels.append(label_idx)
                counts[label_idx] += 1

        X = np.stack(images, axis=0)
        X = X.reshape(X.shape[0], -1)
        y = np.array(labels, dtype=np.int64)

        return DatasetOutput(
            X=X,
            y=y,
            image_shape=self.image_size,
            label_names=label_names,
        )


    # yield image files from a directory tree
    def _iter_images(self, root: Path) -> Iterable[Path]:
        yield from root.rglob("*.jpg")


    # read and preprocess a single image as a flattened vector
    def _read_image(self, path: Path) -> np.ndarray:
        mode = "L" if self.grayscale else "RGB"
        image = Image.open(path).convert(mode).resize(self.image_size)
        array = np.asarray(image, dtype=np.float32)
        array = self._normalize(array)

        return array


    # normalizes pixel values to [0, 1] if configured
    def _normalize(self, array: np.ndarray) -> np.ndarray:
        if not self.normalize:
            result = array.astype(np.float32)

            return result

        result = (array / 255.0).astype(np.float32)

        return result