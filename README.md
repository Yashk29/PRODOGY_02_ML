# PRODOGY_02_ML
TASK 2
# Customer Segmentation using K-means Clustering

This project demonstrates customer segmentation using K-means clustering. The synthetic dataset includes features such as purchase frequency and total amount spent by customers. The goal is to segment the customers into distinct groups based on their purchase behavior.

## Project Structure

- `customer_purchase_history.csv`: The synthetic dataset containing customer purchase history.
- `customer_segmentation.py`: The Python script performing data generation, clustering, and visualization.
- `README.md`: This file, providing an overview of the project.

## Requirements

- Python 3.6 or higher
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

You can install the required Python packages using pip:

```sh
pip install numpy pandas matplotlib scikit-learn
Running the Code
Generate and Save Customer Data:

The script generates random customer data, including purchase frequency and total amount spent, and saves it to customer_purchase_history.csv.

Load and Preprocess Data:

The script then loads the data from the CSV file and standardizes the features using StandardScaler from scikit-learn.

Determine the Optimal Number of Clusters:

The script uses the Elbow Method to determine the optimal number of clusters. It plots the Within-Cluster Sum of Square (WCSS) against the number of clusters to help identify the "elbow point."

Apply K-means Clustering:

Based on the Elbow Method, the script applies K-means clustering with the chosen number of clusters (5 in this case) and adds the cluster labels to the original dataset.

Visualize the Clusters:

Finally, the script visualizes the clusters along with their centroids.
