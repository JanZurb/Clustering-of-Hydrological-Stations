from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
import pandas as pd
import numpy as np


def hierarchical_clustering(features_df, normalized_data, lower_bound, num_clusters):
    #create data frame for where each station and the cluster assignment will be stored
    cluster_labels_hierachical = pd.DataFrame(columns=['Stationsnummer'])
    cluster_labels_hierachical['Stationsnummer'] = features_df['Stationsnummer']

    normalized_data = normalized_data.drop(columns=['Stationsnummer'])

    # create data frame for where each number of clusters and the davies bouldin score and the silhouette score will be stored
    cluster_scores_hierachical = pd.DataFrame(columns=['num_clusters'])
    cluster_scores_hierachical['num_clusters'] = range(lower_bound, lower_bound + num_clusters)

    

    # hierarchical clustering
    for k in range(lower_bound, num_clusters + lower_bound):
        agg = AgglomerativeClustering(linkage="ward", n_clusters=k)
        assignment = agg.fit_predict(normalized_data)

        #save assignment to df cluster_labels_hierachical
        cluster_labels_hierachical[str(k) + ' clusters']= assignment

        #score of hierarchical clustering
        score = silhouette_score(normalized_data, assignment)
        cluster_scores_hierachical.loc[k - lower_bound, 'silhouette_score'] = score

        score = davies_bouldin_score(normalized_data, assignment)
        cluster_scores_hierachical.loc[k - lower_bound, 'davies_bouldin_score'] = score

    return cluster_labels_hierachical, cluster_scores_hierachical
