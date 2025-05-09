import os
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

def load_clustered_data():
    with open('data/latent_space_clustered.json', 'r') as f:
        data = pd.read_json(f)
    return data

def save_clustered_images(data):
    base = 'images/clustered_images'
    if os.path.exists(base):
        shutil.rmtree(base)
    os.makedirs(base)

    for cluster in sorted(data['cluster'].unique()):
        cluster_dir = os.path.join(base, f'cluster_{cluster}')
        os.makedirs(cluster_dir, exist_ok=True)

        cluster_data = data[data['cluster'] == cluster]
        for img_path in cluster_data['path']:
            src = f'images/original_images/{img_path}.jpg'
            dest = os.path.join(cluster_dir, f'{img_path}.jpg')
            shutil.copy(src, dest)
        print(f'Cluster {cluster}: Saved {len(cluster_data)} images.')

def display_clusters(data):
    counts = data['cluster'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts.index, y=counts.values, palette="viridis")
    plt.title('Cluster Counts')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.show()

def main():
    data = load_clustered_data()
    save_clustered_images(data)
    display_clusters(data)

if __name__ == "__main__":
    main()