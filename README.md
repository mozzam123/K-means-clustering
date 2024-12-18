# K-Means Clustering Algorithm

This repository contains a project on K-Means Clustering, which groups individuals based on their age and income. The dataset, code, and visualizations provide a clear understanding of clustering concepts and how to implement them using Python.

---

## Dataset Overview

The dataset used for this project contains information about individuals' ages and their respective incomes. Below is a sample of the dataset:

| Name      | Age | Income($) |
|-----------|-----|-----------|
| Rob       | 27  | 70000     |
| Michael   | 29  | 90000     |
| Mohan     | 29  | 61000     |
| Ismail    | 28  | 60000     |
| Kory      | 42  | 150000    |
| Gautam    | 39  | 155000    |
| ...       | ... | ...       |

---

## Project Steps

### 1. Importing Libraries

- Libraries such as `pandas`, `matplotlib`, and `sklearn` are used.
- `%matplotlib inline` ensures inline plotting for Jupyter notebooks.

### 2. Visualizing the Dataset

- A scatter plot of Age vs. Income is plotted to understand the data distribution.

```python
plt.scatter(df['Age'], df['Income($)'])
```

### 3. Applying K-Means Clustering

- K-Means is applied with an initial `n_clusters=3`.
- Predictions are stored in a new column `cluster`.

```python
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Age', 'Income($)']])
df['cluster'] = y_predicted
```

- Data points are grouped by cluster, and visualized with distinct colors.

```python
plt.scatter(df1.Age, df1['Income($)'], color="green")
plt.scatter(df2.Age, df2['Income($)'], color="red")
plt.scatter(df3.Age, df3['Income($)'], color="pink")
```

### 4. Data Scaling with MinMaxScaler

- MinMaxScaler is used to normalize `Age` and `Income($)` values.

```python
scaler = MinMaxScaler()
scaler.fit(df[['Income($)', 'Age']])
df[['Income($)', 'Age']] = scaler.transform(df[['Income($)', 'Age']])
```

### 5. Reapplying K-Means Clustering

- K-Means is reapplied on the scaled dataset.

```python
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Age', 'Income($)']])
df['cluster'] = y_predicted
```

- Cluster centers are visualized:

```python
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color="yellow", marker='*', label="centroid")
```

### 6. Using the Elbow Method

- The Elbow Method helps determine the optimal number of clusters by plotting Sum of Squared Errors (SSE) for different cluster counts.

```python
k_range = range(1, 10)
sse = []
for k in k_range:
    km = KMeans(n_clusters=k)
    km.fit(df[['Age', 'Income($)']])
    sse.append(km.inertia_)

plt.plot(k_range, sse)
plt.xlabel('K')
plt.ylabel('Sum of Squared Error')
```

---

## Visualizations

- **Initial Scatter Plot:** Visualizing raw data points.
- **Clustered Data:** Scatter plot with clusters distinguished by colors and centroids marked.
- **Elbow Plot:** To determine the optimal number of clusters.

---

## Requirements

- Python 3.7+
- Libraries:
  - pandas
  - matplotlib
  - scikit-learn

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## How to Run

1. Clone the repository:

```bash
git clone <repository_url>
```

2. Navigate to the project directory:

```bash
cd kmeans-clustering
```

3. Run the script:

```bash
python main.py
```

---

## Results

- Successfully clustered individuals into three groups based on their age and income.
- Visualized centroids and optimal cluster count using the Elbow Method.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

