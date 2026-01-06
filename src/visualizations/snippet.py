import gzip
import joblib
import numpy as np

# Loads MNIST images from IDX3 .gz and return (n, 784) float32 in [0, 1]
def load_mnist_images_gz(filename):
    with gzip.open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        if magic != 2051: # MNIST format
            raise ValueError(f"Invalid MNIST magic {magic} in: {filename}")
            
        n_images = int.from_bytes(f.read(4), 'big')
        n_rows = int.from_bytes(f.read(4), 'big')
        n_cols = int.from_bytes(f.read(4), 'big')

        buf = f.read(n_images * n_rows * n_cols)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(n_images, n_rows * n_cols).astype(np.float32) / 255.0

    return data


# Loads trained PCA model
def load_trained_pca(filename):
    model = joblib.load(filename)

    return model


# Apply PCA to one flattened MNIST image and return a (2,) embedding
def apply_pca(model, image):
    return model.transform(image.reshape(1, -1))[0]


# Compute embedding for a fixed MNIST index using the loaded model
def main(mnist_file, model_file, idx):
    data = load_mnist_images_gz(mnist_file)
    model = load_trained_pca(model_file)
    z = apply_pca(model, data[idx])

    return z


if __name__ == "__main__":
    MNIST_FILE = "../../datasets/data/visualizations/train-images-idx3-ubyte.gz"
    MODEL_FILE = "pca_model.pkl"
    IDX = 0
    z = main(MNIST_FILE, MODEL_FILE, IDX)
    print("LOADED:", z) # output must be the same as in the .ipynb