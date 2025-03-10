from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

def main():
    NUM_CLUSTERS = 10

    df = pd.read_hdf('data/latent_space.h5')
    latent_space = np.stack(df['latent_space'].values)

    # Creating a KMeans model with chosen number of clusters
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=0, n_init=10)

    # Fitting the model to your data
    kmeans.fit(latent_space)

    # Getting the cluster assignments for each image
    df['cluster'] = kmeans.labels_

    # Saving data
    df.to_hdf('data/latent_space.h5', key='df_items', mode='w')

if __name__ == '__main__': 
    main()