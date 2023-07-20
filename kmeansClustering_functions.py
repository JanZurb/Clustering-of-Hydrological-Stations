from sklearn.cluster import KMeans
import math 
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
import pandas as pd

def kmeans_clustering (features_df, normalized_data, lower_bound, num_clusters):
    cluster_labels_kmeans = pd.DataFrame()
    cluster_labels_kmeans['Stationsnummer'] = features_df['Stationsnummer']

    normalized_data = normalized_data.drop(['Stationsnummer'], axis=1)

    # create data frame for where each number of clusters and the davies bouldin score and the silhouette score will be stored
    cluster_scores_kmeans = pd.DataFrame(columns=['num_clusters'])
    cluster_scores_kmeans['num_clusters'] = range(lower_bound, lower_bound + num_clusters)


    for cluster in range(lower_bound, lower_bound + num_clusters):
        km = KMeans(n_clusters=cluster, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
        assignment = km.fit_predict(normalized_data)
        #add cluster label to the normalized data in a new column called num_clusters clusters
        cluster_labels_kmeans[str(cluster) + ' clusters']= assignment

        #calculate scores and save them
        
        score = silhouette_score(normalized_data, assignment)
        cluster_scores_kmeans.loc[cluster - lower_bound, 'silhouette_score'] = score

        score = davies_bouldin_score(normalized_data, assignment)
        cluster_scores_kmeans.loc[cluster - lower_bound, 'davies_bouldin_score'] = score
    for cluster in range(lower_bound, num_clusters + 1):
        cluster_labels_kmeans[str(cluster) + ' clusters'] = cluster_labels_kmeans[str(cluster) + ' clusters']

    return cluster_labels_kmeans, cluster_scores_kmeans