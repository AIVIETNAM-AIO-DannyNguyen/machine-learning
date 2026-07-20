# K-Means Clustering from Scratch

A NumPy-only implementation of the K-Means clustering algorithm. Includes a side-by-side comparison against scikit-learn's `KMeans`.

## Algorithm overview

K-Means partitions `n` points into `k` clusters by alternating two steps until convergence:

1. **Assignment step** — assign every point to its nearest centroid, using Euclidean distance:

   ```
   d(x, c) = sqrt( sum_j (x_j - c_j)^2 )
   ```

2. **Update step** — recompute each centroid as the mean of the points assigned to it:

   ```
   c_i = (1 / |C_i|) * sum_{x in C_i} x
   ```

The algorithm stops when centroids stop changing (convergence) or a maximum number of iterations is reached.

Clustering quality is measured with **WCSS (Within-Cluster Sum of Squares)**, also called *inertia* — the sum of squared distances between each point and its assigned centroid. Lower WCSS means tighter, more compact clusters.

## Dataset

A small dataset of 9 customers, each with two features:

| Feature       | Description              |
|---------------|---------------------------|
| `age`         | Customer age              |
| `expenditure` | Customer spending amount  |

The goal is to group customers into `k = 3` clusters.

## What's implemented

- Manual K-Means from scratch (initialization, assignment step, update step, convergence check, WCSS) using only NumPy.
- A comparison run using scikit-learn's `KMeans` (`k-means++` initialization, multiple restarts via `n_init`).
- A scatter plot visualizing the final clusters and centroids.

## Notes 

- The manual implementation initializes centroids by slicing the raw data (`X[2:k+2]`) rather than using random or `k-means++` initialization — this is simpler for learning purposes but more sensitive to a poor starting point.
- `max_iters` is set low (2) in the manual version to make each step easy to trace by hand; increasing it (or adding a tolerance-based stopping rule instead of exact equality) would make convergence more robust.
- Next steps for practicing further: implement `k-means++` initialization from scratch, and use the elbow method / silhouette score to choose `k` automatically.


## Requirements

- numpy
- matplotlib
- scikit-learn

