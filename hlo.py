import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def convert_pickle_to_image(pickle_file, output_image_file):
    # Load the embeddings from the pickle file
    with open(pickle_file, 'rb') as f:
        embeddings = pickle.load(f)
    
    # Ensure embeddings are in a proper format (1D array)
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings)
    
    if embeddings.ndim == 1:
        # Handle 1D embeddings (no PCA needed)
        plt.figure(figsize=(10, 4))
        plt.plot(embeddings, marker='o', linestyle='-', color='blue')
        plt.title("1D Embedding Visualization")
        plt.xlabel("Dimension Index")
        plt.ylabel("Value")
        plt.grid(True)
    elif embeddings.ndim == 2:
        # Use PCA for dimensionality reduction if embeddings are 2D
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings)

        plt.figure(figsize=(6, 6))
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c='blue', label='Embedding')
        plt.title("2D Embedding Visualization (PCA)")
        plt.xlabel("PCA Dimension 1")
        plt.ylabel("PCA Dimension 2")
        plt.legend()
        plt.grid(True)
    else:
        raise ValueError("Unsupported embedding dimensions. Expected 1D or 2D array.")

    # Save the image as a .jpg file
    plt.savefig(output_image_file, format='jpg')
    print(f"Embedding visualization saved to {output_image_file}.")
    plt.close()  # Close the plot to avoid displaying it

# Example Usage
pickle_file_path = 'db/Tharukesh.pickle'  # Replace with your pickle file path
output_image_path = './db/Tharukesh_embedding_visualization.jpg'  # Output file path
convert_pickle_to_image(pickle_file_path, output_image_path)
