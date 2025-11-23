from typing import List, Tuple
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD


class S21Images:
    def __init__(self, path: str) -> None:
        self.path = path
        self.image_matrix = None
        self.svd_result = None


    # loads a single image and converts it into a 2D matrix
    def load_image(self, file_name: str) -> np.ndarray:
        img = Image.open(f'{self.path}/{file_name}').convert("L")
        self.image_matrix = np.array(img)

        return self.image_matrix

    
    # transforms image matrix using svd
    def transform_svd(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = self.image_matrix
        if X is None:
            raise RuntimeError("Matrix has not been created, call load_image() first")

        self.svd_result = np.linalg.svd(X, full_matrices=False)

        return self.svd_result

    
    # plots the singular value spectrum
    def plot_spectrum(self) -> None:
        if self.svd_result is None:
            raise RuntimeError("SVD matrix has not been created, call transform_svd() first")

        U, S, Vt = self.svd_result
        
        plt.figure(figsize=(8, 4))
        plt.plot(np.arange(1, len(S) + 1), S)
        plt.xlabel("Mode index")
        plt.ylabel("Singular value")
        plt.title("Singular value spectrum")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    # plots cumulative explained variance versus rank
    def plot_explained_variance(self) -> None:
        if self.svd_result is None:
            raise RuntimeError("SVD matrix has not been created, call transform_svd() first")

        U, S, Vt = self.svd_result
        variance = S**2
        explained_variance_ratio = variance / variance.sum()
        cumulative_explained_variance = np.cumsum(explained_variance_ratio)

        plt.figure(figsize=(8, 4))
        plt.plot(np.arange(1, len(S) + 1), cumulative_explained_variance)
        plt.xlabel("Rank (number of modes)")
        plt.ylabel("Cumulative explained variance")
        plt.title("Cumulative explained variance vs rank")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    # builds image from svd matrix
    def reconstruct(self, k: int) -> np.ndarray:
        svd_result = self.svd_result
        if svd_result is None:
            raise RuntimeError("SVD matrix has not been created, call transform_svd() first")
        
        U, S, Vt = svd_result

        Uk = U[:, :k]
        Sk = np.diag(S[:k])
        Vtk = Vt[:k, :]

        Xk = Uk @ Sk @ Vtk

        return Xk

    
    # converts to picture
    def plot_ranks(self, ranks: tuple | list) -> None:
        plt.figure(figsize=(12, 6))

        for i, k in enumerate(ranks, start=1):
            Xk = self.reconstruct(k)

            plt.subplot(2, 3, i)
            plt.imshow(Xk, cmap="gray", vmin=0, vmax=255)
            plt.title(f"rank = {k}")
            plt.axis("off")

        plt.tight_layout()
        plt.show()