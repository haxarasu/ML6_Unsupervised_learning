import gzip
import struct
import umap
import joblib
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE, LocallyLinearEmbedding
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, silhouette_score
from typing import Optional, Tuple


class S21MnistOperations:
    def __init__(self, images_path: str, labels_path: str) -> None:
        self.images_path: str = images_path
        self.labels_path: str = labels_path
        self.images: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.pca_model = None
        self.X_small = None


    # loads MNIST images from .gz and returns a matrix
    def _load_mnist_images(self, path: str) -> np.ndarray:
        with gzip.open(path, "rb") as f:
            magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
            data = np.frombuffer(f.read(), dtype=np.uint8)
            images = data.reshape(n, rows * cols).astype(np.float32) / 255.0

        return images


    # loads MNIST labels from .gz and returns a vector of labels
    def _load_mnist_labels(self, path: str) -> np.ndarray:
        with gzip.open(path, "rb") as f:
            magic, n = struct.unpack(">II", f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)

        return labels


    # returns loaded images
    def _get_images(self) -> np.ndarray:
        if self.images is None:
            self.images = self._load_mnist_images(self.images_path)

        return self.images


    # gets smaller amount of images for use of resource-demanding algorithms
    def _get_x_small(self, n_samples: int = 5000) -> np.ndarray:
        if self.X_small is None:
            if self.images is None:
                raise RuntimeError(
                    "Images are not loaded yet. Call `load_dataset()` before using X_small."
                )
            self.X_small = self.images[:n_samples]

        return self.X_small


    # returns loaded labels
    def _get_labels(self) -> np.ndarray:
        if self.labels is None:
            self.labels = self._load_mnist_labels(self.labels_path)

        return self.labels

    
    # loads all images and labels into attributes and returns them
    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        self.images = self._load_mnist_images(self.images_path)
        self.labels = self._load_mnist_labels(self.labels_path)
        self.X_small = self._get_x_small()

        return self.images, self.labels 


    # PCA
    def transform_pca(self) -> np.ndarray:
        X_small = self.X_small

        pca = PCA(n_components=2, random_state=42)
        self.X_pca_2d = pca.fit_transform(X_small)
        # keep the trained model so it can be saved later
        self.pca_model = pca

        return self.X_pca_2d


    # saves trained pca model for task from submission
    def save_trained_pca(self, filename: str = "pca_model.pkl") -> None:
        if self.pca_model is None:
            raise RuntimeError(
                "PCA model is not trained yet. Call `transform_pca()` before saving."
            )

        joblib.dump(self.pca_model, filename)


    # SVD    
    def transform_svd(self) -> np.ndarray:
        X_small = self.X_small

        svd = TruncatedSVD(n_components=2, random_state=42)
        self.X_svd_2d = svd.fit_transform(X_small)

        return self.X_svd_2d


    # randomized-SVD
    def transform_random_svd(self) -> np.ndarray:
        X_small = self.X_small

        r_svd = TruncatedSVD(
            n_components=2,
            n_iter=5,
            random_state=42
        )
        self.X_rand_svd_2d = r_svd.fit_transform(X_small)
        
        return self.X_rand_svd_2d


    # TSNE
    def transform_tsne(self) -> np.ndarray:
        X_small = self.X_small

        tsne = TSNE(
            n_components=2,
            perplexity=30,
            learning_rate='auto',
            init='random',
            random_state=42
        )
        self.X_tsne_2d = tsne.fit_transform(X_small)
        
        return self.X_tsne_2d

    
    # UMAP
    def transform_umap(self) -> np.ndarray:
        X_small = self.X_small

        reducer = umap.UMAP(
            n_components=2
        )
        self.X_umap_2d = reducer.fit_transform(X_small)
        
        return self.X_umap_2d


    # LLE
    def transform_lle(self) -> np.ndarray:
        X_small = self.X_small

        lle = LocallyLinearEmbedding(
            n_neighbors=10,
            n_components=2,
            method="standard",
            random_state=42,
        )
        self.X_lle_2d = lle.fit_transform(X_small)

        return self.X_lle_2d

    
    # evaluates how well different 2D embeddings separate digit classes
    def evaluate_2d(self) -> None:
        X_small = self.X_small
        labels = self.labels
        y_small = labels[:len(X_small)]

        # compute 2D embeddings, keeping a consistent subset size
        embeddings = {
            "pca": self.transform_pca(),
            "svd": self.transform_svd(),
            "rand_svd": self.transform_random_svd(),
            "tsne": self.transform_tsne(),   
            "umap": self.transform_umap(),
            "lle": self.transform_lle(),   
        }

        metrics: dict[str, dict[str, float]] = {}

        for name, X_emb in embeddings.items():
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X_emb, y_small)
            y_pred = knn.predict(X_emb)

            acc = accuracy_score(y_small, y_pred)
            f1 = f1_score(y_small, y_pred, average="macro")
            sil = silhouette_score(X_emb, y_small)

            metrics[name] = {
                "accuracy": acc,
                "f1_macro": f1,
                "silhouette": sil,
            }

        for name, vals in metrics.items():
            print(f"{name.upper()}:")
            for metric_name, value in vals.items():
                print(f"  {metric_name}: {value:.4f}")
            print()