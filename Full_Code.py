import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Load the dataset
df_original = pd.read_csv("AIB 503 data.csv")
print(df_original.head())
print(df_original.shape)


# Q1a) Data Preprocessing & EDA

# Drop unnecessary columns
df_original = df_original.drop(columns=['Unnamed: 0', 'id']) 

# Check for missing values in the dataset
missing_values = df_original.isnull().sum()
print(missing_values)

# Handle missing values in 'Arrival Delay in Minutes' by imputing with the median
df_original['Arrival Delay in Minutes'] = df_original['Arrival Delay in Minutes'].fillna(
    df_original['Arrival Delay in Minutes'].median())
# Check for missing value again
print(df_original.isnull().any())

# Setting up EDA visualizations

# 1. Distribution of numerical features (Histograms)
plt.figure(figsize=(12, 6))
df_original.hist(figsize=(16, 12), bins=20)
plt.suptitle("Histograms of Numerical Features", fontsize=16)
plt.tight_layout()
plt.show()

# 2. Plot Bar chart for selected_categorical_featuress
selected_categorical_features = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ax, feature in zip(axes.flatten(), selected_categorical_features):
    sns.countplot(x=df_original[feature], ax=ax)
    ax.set_title(f'Distribution of {feature}')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

plt.tight_layout()
plt.show()


# 3. Correlation heatmap to understand relationships between features
numeric_df = df_original.select_dtypes(include=['number'])
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Heatmap of Feature Correlations')
plt.tight_layout()
plt.show()


# 4. Pair plot for selected numerical variables
pairplot_cols = ['Age', 'Flight Distance', 
                'Departure Delay in Minutes', 'Arrival Delay in Minutes',
                'satisfaction']  # satisfaction is for hue

sns.pairplot(df_original[pairplot_cols], hue='satisfaction')
plt.show()

# Encoding Categorical Variables
df_original['Gender'] = df_original['Gender'].map({'Female': 0, 'Male': 1})
df_original['Customer Type'] = df_original['Customer Type'].map({'Loyal Customer': 1, 'disloyal Customer': 0})
df_original['Type of Travel'] = df_original['Type of Travel'].map({'Business travel': 1, 'Personal Travel': 0})
df_original['satisfaction'] = df_original['satisfaction'].map({'neutral or dissatisfied': 0, 'satisfied': 1})

# One-Hot Encoding for 'Class'
df_original = pd.get_dummies(df_original, columns=['Class'], drop_first=True)  # Drops 'Business' class as baseline

# CREATE AN UNSCALED COPY FOR POST-CLUSTER INTERPRETATION
df_unscaled = df_original.copy()

# Identify numeric columns to scale (excluding binary ones)
binary_columns = ['Gender', 'Customer Type', 'Type of Travel', 'satisfaction', 'Class_Eco', 'Class_Eco Plus']
numeric_columns = [col for col in df_original.columns if col not in binary_columns]

# Initialize the scaler
scaler = StandardScaler()
df_scaled = df_original.copy()

# Apply StandardScaler to numeric columns
df_scaled[numeric_columns] = scaler.fit_transform(df_scaled[numeric_columns])

# Display first few rows to verify scaling
print(df_scaled.head())


# Q1b) K-means Clustering
# Prepare data for clustering
col_order = numeric_columns + binary_columns
X = df_scaled[col_order]

# Determine optimal number of clusters using Elbow Method 
wcss = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, init= 'k-means++', n_init= 50, max_iter= 300, tol= 0.0001, random_state= 111)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot Elbow Method
plt.plot(range(2,11), wcss, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()

# Fitting the K-Means Model
optimal_k = 6  # Example based on elbow
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=20, max_iter= 300, tol= 0.0001, random_state=111)
kmeans.fit(X)

# ATTACH CLUSTER LABELS TO BOTH SCALED & UNSCALED DATAFRAMES
df_scaled['Cluster'] = kmeans.labels_
df_unscaled['Cluster'] = kmeans.labels_

# Display the data again with clusters
print(df_unscaled.head())


# Get the Silhouette Score for K=6 
labels_6 = kmeans.labels_
silhouette_6 = silhouette_score(df_scaled, labels_6)
print(f"Silhouette Score for k=6: {silhouette_6}")



# Q1c) Interpretation and Recommendations

# Cluster Observation
cluster_sizes = df_unscaled['Cluster'].value_counts()
print(cluster_sizes)
cluster_summary = df_unscaled.groupby('Cluster').mean()
print(cluster_summary) 


# Q1d) Visualizations of Clustering results

# PCA
features = df_scaled.drop('Cluster', axis=1)
labels = df_scaled['Cluster']

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(features)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.xlabel(f'PC1 ({"{:.2f}".format(pca.explained_variance_ratio_[0]*100)}% Variance)')
plt.ylabel(f'PC2 ({"{:.2f}".format(pca.explained_variance_ratio_[1]*100)}% Variance)')
plt.title('K-Means Clusters Visualized via 2D PCA')
plt.colorbar(scatter, label='Cluster')
plt.show()
