# Kmeans_clustering_of_Stocks
Clustering stocks together using K-means clustering
Introducing the k-means algorithm:
The term k-means was first used by MacQueen in 1967, although the idea dates back to Steinhaus in 1957. K-means is an unsupervised classification (clustering) algorithm that groups objects into k groups based on their characteristics.

Clustering is done by minimizing the sum of distances between each object and the centroid of its group or cluster. Quadratic distance is often used. The algorithm consists of three steps:

Initialization:
once the number of groups, k, has been chosen, k centroids are established in the data space, for example, choosing them randomly.
Assign objects to centroids: each data object is assigned to its nearest centroid.

Centroid update: 
the position of the centroid of each group is updated, taking as the new centroid the position of the average of the objects belonging to said group.
Steps 2 and 3 are repeated until the centroids do not move, or move below a threshold distance at each step.

Clustering of stocks by return and volatility
We analyze the S&P 500 index to cluster stocks based on return and volatility. This index comprises 500 large-cap US companies from various sectors, traded on NYSE or Nasdaq. Due to its representation of the US’s largest publicly traded firms, it serves as a suitable dataset for algorithmic k-means clustering.
Load Data
We calculate the annual average return and volatility for each company by obtaining their adjusted closing prices during 01/02/2020–12/02/2022 and inserting them into a dataframe, which is then annualized (assuming 252 market days per year).
Determine the optimal number of clusters
The Elbow curve method is a technique used to determine the optimal number of clusters for K-means clustering. The method works by plotting the sum of squared errors (SSE) for different values of k (number of clusters). The optimal number of clusters is the value of k at which the SSE starts to decrease at a slower rate. The optimal number of clusters is determined by finding the elbow or the point at which the SSE reaches its minimum value. In this case, the optimal number of clusters is 4.
K-means clustering
Once the optimum number of clusters has been defined, we proceed to create them. In the first instance, the centroids are defined using the sklearn library. For the creation of 4 groups of actions, the K-means algorithm iteratively assigns data points to the groups based on their similarity of characteristics, or “features”, in this case, Average Annualized Return and Average Annualized Volatility.
The algorithm initially randomly assigns the data points to the clusters and then calculates the centroid of each cluster, which is the mean of all the data points within the cluster. Then, it compares the data points to the centroid and reassigns them to groups accordingly. This process is repeated until the centroid of each cluster remains relatively stable, at which point the algorithm stops and each cluster is assigned a label. The end result is a set of 4 groups, each containing stocks that have similar returns and volatility.
Clustering of shares by price-earnings ratio and dividend rateFollowing this conceptual line, it is possible to apply a clustering similar to the one carried out previously, exchanging the variables Average Annualized Return and Average Annualized Volatility for PER (Price-Earnings Ratio) and Dividend Rate (Dividend Yield). In this way, we could differentiate between “value” companies and “growth” companies.

Load Data
The trailing price-to-earnings (P/E) ratio is a relative valuation multiple that is based on the last 12 months of actual earnings. It is calculated by taking the current share price and dividing it by the earnings per share (EPS) for the last 12 months. While the dividend rate, or dividend rate, is the amount of cash that a company returns to its shareholders annually as a percentage of the company’s market value.
Determine the optimal number of clusters
Once the Price-Earnings Ratio and Dividend Rate data have been obtained, we can reapply the Elbow method to determine the optimal number of clusters
K-means clustering
Once the optimum number of clusters has been defined, we proceed to create them. In the first instance, the centroids are defined using the sklearn library. For the creation of groups of actions, the K-means algorithm iteratively assigns data points to the groups based on their similarity of characteristics, or “features”, in this case, Price-Earnings Ratio and Dividend Rate.
When making a first approximation clustering by trailing price-to-earnings (P/E) and dividend rate, the presence of outliers and excessive dispersion among the observations is evident, so we proceed to filter the actions and normalize the data to eliminate these distortions.

Outlier treatment
We see as a result a scattered and unclear clustering. Therefore, eliminating outliers and normalizing the data will be necessary to achieve more accurate clusters.

First, we apply a filter to include only stocks with price-to-earnings less than 200 and dividend rate less than 5.
Then, we apply MaxAbsScaler. MaxAbsScaler scales each feature by its maximum absolute value. This estimator scales and translates each feature individually such that the maximal absolute value of each feature in the training set will be 1. It does not shift/center the data and thus does not destroy any sparsity.
Once MaxAbsScaler is applied, we perform the Elbow method again with the normalized variables as input:
Once the pertinent modifications have been made, it is possible to obtain 4 clusters generated by the K-means algorithm according to the trailing price-to-earnings (P/E) and dividend rate of each share.
It is possible to graphically verify that the algorithm assigned a greater weight to the dividend rate variable when creating the clusters. In this way, 4 sets of actions are distinguished: C1 with Null or very low, C2 with low, C3 with a medium-high and C4 with a high dividend rate.

