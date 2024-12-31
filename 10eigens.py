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
            print(f"Skipping {filename}: Not a valid image file.")
    return np.array(image_vectors)

def compute_mean_vector(images):
    return np.mean(images, axis=0)

def compute_covariance_matrix(images, mean_vector):
    centered_images = images - mean_vector
    covariance_matrix = np.cov(centered_images, rowvar=False)
    return covariance_matrix, centered_images

def spectral_decomposition(cov_matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    return eigenvalues, eigenvectors

def visualize_eigenvectors(eigenvectors, image_shape, num_images=10):
    top_eigenvectors = eigenvectors[:, -num_images:]
    for i in range(num_images):
        eigenvector = top_eigenvectors[:, i]
        normalized_vector = (eigenvector - np.min(eigenvector)) / (np.max(eigenvector) - np.min(eigenvector)) * 256
        eigen_image = normalized_vector.reshape(image_shape)
        img = Image.fromarray(eigen_image.astype(np.uint8))
        img.show(title=f'Eigenvector {i + 1}')
        img.save(f'eigenvector_{i + 1}.png')

def main():
    train_folder = "C:/Users/siddu/OneDrive/Desktop/hw2 ML/#4/train"
    
    # Step 1: Load training images
    images = load_images(train_folder)
    if images.size == 0:
        raise ValueError("No images found in the specified folder or the folder path is incorrect.")
    print(f"Number of images loaded: {images.shape[0]}")
    print(f"Dimensions of each image vector: {images.shape[1]}")

    # Step 2: Calculate mean vector and covariance matrix
    mean_vector = compute_mean_vector(images)
    cov_matrix, centered_images = compute_covariance_matrix(images, mean_vector)

    # Step 3: Perform spectral decomposition
    eigenvalues, eigenvectors = spectral_decomposition(cov_matrix)

    # Assuming the dimensions of each image are (height, width), calculated earlier
    image_shape = (60, 80)  # Example dimensions, replace with actual dimensions

    # Step 4: Visualize the top 10 eigenvectors as images
    visualize_eigenvectors(eigenvectors, image_shape, num_images=10)

# Execute the main function
if __name__ == "__main__":
    main()
