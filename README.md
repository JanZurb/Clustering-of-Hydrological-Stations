# Clustering-of-Hydrological-Stations
With the help of machine learning clustering algorithms hydrological stations are clustered in coherent groups 

## Purpose of the Project
The purpose of this project is to cluster hydrological stations in switzerland. Based on their discharge behaviour and their temperature course over the last fourty years.
This clustering can be used for optimization in training of a graph based deep neural network and may be of help to account for missing data.

## Method
To cluster the stations, a visual analysis of the behaviour of discharge and temperature course was performed, to get meaningful features. With these features clustering algorithms namely a hierarchical and a kmeans algorithm were applied to get groups of similar behaviour. 

## Evaluation 
Due to the fact that clustering is an unsupervised learning approach the number of clusters was not given. to get the optimal number of clusters two metrics were used. The davies bouldin and the silhouette score. They both account coherent groups with good separation.