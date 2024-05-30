import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)

# Generate random data for customers
num_customers = 1000
purchase_frequency = np.random.randint(1, 101, size=num_customers)  # Random integers between 1 and 100
total_amount_spent = np.random.uniform(10, 500, size=num_customers)  # Random floats between 10 and 500

# Create DataFrame
customer_data = pd.DataFrame({
    'CustomerID': range(1, num_customers + 1),
    'Purchase Frequency': purchase_frequency,
    'Total Amount Spent': total_amount_spent
})

# Save dataset to CSV
customer_data.to_csv('customer_purchase_history.csv', index=False)

# Display the first few rows of the dataset
print(customer_data.head())

# Load the dataset
data = pd.read_csv('customer_purchase_history.csv')

# Extract features for clustering
X = data[['Purchase Frequency', 'Total Amount Spent']]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Based on the Elbow Method, let's choose 5 clusters
k = 5

# Apply K-means clustering
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to the original dataset
data['Cluster'] = clusters

# Visualize the clusters
plt.figure(figsize=(10, 6))
for cluster in range(k):
    cluster_data = data[data['Cluster'] == cluster]
    plt.scatter(cluster_data['Purchase Frequency'], cluster_data['Total Amount Spent'], label=f'Cluster {cluster}')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids', marker='X')
plt.title('Customer Segmentation based on Purchase History')
plt.xlabel('Purchase Frequency')
plt.ylabel('Total Amount Spent')
plt.legend()
plt.show()
