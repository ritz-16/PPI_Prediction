import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load data
data_path = '9606.protein.links.v12.0.txt'
data = pd.read_csv(data_path, delimiter=' ')

# Filter data for high-confidence interactions
high_confidence_data = data[data['combined_score'] >= 700]

# Create a graph
G = nx.Graph()
for idx, row in high_confidence_data.iterrows():
    G.add_edge(row['protein1'], row['protein2'], weight=row['combined_score'])

# Network Analysis: Calculate network metrics
betweenness = nx.betweenness_centrality(G)
closeness = nx.closeness_centrality(G)
print("Betweenness Centrality:", betweenness)
print("Closeness Centrality:", closeness)

# Predict new interactions based on common neighbors
def predict_new_interactions(G, threshold=2):
    potential_new_links = []
    for u, v in combinations(G.nodes, 2):
        if not G.has_edge(u, v):
            common_neigh = len(list(nx.common_neighbors(G, u, v)))
            if common_neigh > threshold:
                potential_new_links.append((u, v, common_neigh))
    return potential_new_links

new_interactions = predict_new_interactions(G)
print("Potential new interactions:", new_interactions[:10])

# Machine Learning for Prediction Enhancement: Feature extraction
def extract_features(node, G):
    return {
        'degree': G.degree(node),
        'betweenness': betweenness[node],
        'closeness': closeness[node],
    }

# Prepare data for machine learning
X = []
y = []
for u, v, data in G.edges(data=True):
    X.append([extract_features(u, G)['degree'], extract_features(v, G)['degree']])
    y.append(data['weight'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print("MSE:", mse)

# Visualization of the network
pos = nx.spring_layout(G)
nx.draw(G, pos, node_color='lightblue', with_labels=False, edge_color="#BBBBBB")
plt.title("Protein Interaction Network")
plt.show()
