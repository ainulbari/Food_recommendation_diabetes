import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from minisom import MiniSom
import matplotlib.pyplot as plt
import streamlit as st

# Load data
df = pd.read_csv('C:\\Users\\ainul\\Downloads\\New folder (2)\\New folder (2)\\main_dataset.csv')


# Data preprocessing
df['GI (per 100 glucose)'] = pd.to_numeric(df['GI (per 100 glucose)'], errors='coerce')
df['Carbohydrates (per 100 g)'] = pd.to_numeric(df['Carbohydrates (per 100 g)'], errors='coerce')
df = df.dropna()

# Clustering
conditions = [
    (df['GI (per 100 glucose)'].between(0, 55)) & (df['Carbohydrates (per 100 g)'].between(0, 15)),
    (df['GI (per 100 glucose)'].between(56, 69)) & (df['Carbohydrates (per 100 g)'].between(16, 30)),
    (df['GI (per 100 glucose)'] >= 70) | (df['Carbohydrates (per 100 g)'] >= 31)
]
cluster_labels = ['Normal', 'Limited', 'Avoidable']
df['Cluster'] = np.select(conditions, cluster_labels)

X = df[['GI (per 100 glucose)', 'Carbohydrates (per 100 g)']].values
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_std)
    wcss.append(kmeans.inertia_)

optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=0)
df['KMeans_Cluster'] = kmeans.fit_predict(X_std)

# Self-Organizing Map (SOM)
som = MiniSom(x=10, y=10, input_len=2, sigma=1.0, learning_rate=0.5, random_seed=42)
som.train_random(X_std, 100)

# Streamlit app
st.title('Food Recommendation App')

# Dropdown for food selection
food_name = st.selectbox('Select a food:', df['Food'].unique())

def get_cluster(food_name):
    food_index = df.index[df['Food'] == food_name].tolist()
    if not food_index:
        return "Food not found in the dataset"

    kmeans_cluster = df.iloc[food_index]['KMeans_Cluster'].values[0]
    return df.iloc[food_index]['Cluster'].values[0], kmeans_cluster

def recommend_foods(kmeans_cluster):
    normal_foods = df[df['KMeans_Cluster'] == kmeans_cluster]
    recommendations = normal_foods['Food'].sample(5).tolist()
    return recommendations


# Display cluster information
cluster, kmeans_cluster = get_cluster(food_name)
if cluster == "Food not found in the dataset":
    st.warning(cluster)
else:
    st.success(f"The food belongs to Cluster: {cluster}")

    if cluster == 'Avoidable':
        # Recommendations for Avoidable cluster
        recommendations = recommend_foods(kmeans_cluster)
        st.info("Hey, instead of this, you might want to try these foods, they are healthier!")
        st.write(recommendations)

# Display scatter plot
fig, ax = plt.subplots()
scatter = ax.scatter(X_std[:, 0], X_std[:, 1], c=df['KMeans_Cluster'], cmap='viridis', alpha=0.8)
legend = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend)
st.pyplot(fig)

# Elbow Method plot
st.subheader("Elbow Method for Optimal K")
st.line_chart(pd.DataFrame({'Number of clusters (K)': range(1, 11), 'WCSS': wcss}).set_index('Number of clusters (K)'))

plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of clusters (K)')
plt.ylabel('WCSS')
plt.show()