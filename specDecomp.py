import numpy as np
from PIL import Image
import os

def load_images(folder):
    image_vectors = []
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            img = Image.open(file_path).convert('L')
            img_vector = np.array(img).flatten()
            image_vectors.append(img_vector)
        except (IOError, ValueError):
            print(f"skipping {filename}: not a valid image file.")
    return np.array(image_vectors)

def compute_mean_vector(images):
    return np.mean(images, axis=0)

def compute_covariance_matrix(images, mean_vector):
    centered_images = images - mean_vector
    return np.cov(centered_images, rowvar=False), centered_images

def spectral_decomposition(cov_matrix):
    return np.linalg.eigh(cov_matrix)

def reconstruct_image(test_image, mean_vector, eigenvectors, m):
    centered_image = test_image - mean_vector
    top_eigenvectors = eigenvectors[:, -m:]
    weights = np.dot(centered_image, top_eigenvectors)
    return mean_vector + np.dot(weights, top_eigenvectors.T)

def load_test_image(test_folder, filename):
    img = Image.open(os.path.join(test_folder, filename)).convert('L')
    return np.array(img).flatten()

def visualize_reconstructed_images(reconstructed_images, m_values, image_shape):
    for m, reconstructed in zip(m_values, reconstructed_images):
        img_array = reconstructed.reshape(image_shape)
        img = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
        img.show(title=f'reconstructed with m={m}')
        img.save(f'reconstructed_m_{m}.png')

def main():
    train_folder = "C:/Users/siddu/OneDrive/Desktop/hw2 ML/#4/train"
    test_folder = "C:/Users/siddu/OneDrive/Desktop/hw2 ML/#4/test"
    test_image_name = "subject15.normal"

    images = load_images(train_folder)
    if images.size == 0:
        raise ValueError("no images found in the specified folder or the folder path is incorrect.")
    print(f"number of images loaded: {images.shape[0]}")
    print(f"dimensions of each image vector: {images.shape[1]}")

    mean_vector = compute_mean_vector(images)
    cov_matrix, _ = compute_covariance_matrix(images, mean_vector)

    eigenvalues, eigenvectors = spectral_decomposition(cov_matrix)

    test_image = load_test_image(test_folder, test_image_name)
    image_shape = (60, 80)

    m_values = [2, 10, 100, 1000, 4000]
    reconstructed_images = []
    for m in m_values:
        reconstructed = reconstruct_image(test_image, mean_vector, eigenvectors, m)
        reconstructed_images.append(reconstructed)

    visualize_reconstructed_images(reconstructed_images, m_values, image_shape)

if __name__ == "__main__":
    main()
