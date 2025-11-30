import numpy as np
from sklearn.cluster import KMeans

def cluster_edges(edge_nodes, h_clusters, seed=None):
    n = len(edge_nodes)
    if h_clusters <= 0:
        raise ValueError("h_clusters must be > 0")
    features = []
    for ninfo in edge_nodes:
        # cp and avg communication cost as features
        cp = float(ninfo.get('cp', 1.0))
        cm_avg = float(np.mean(list(ninfo.get('cm', {}).values())) if ninfo.get('cm') else 1.0)
        features.append([cp, cm_avg])
    features = np.array(features)
    kmeans = KMeans(n_clusters=h_clusters, random_state=seed).fit(features)
    labels = kmeans.labels_
    cluster_map = [[] for _ in range(h_clusters)]
    for idx, lab in enumerate(labels):
        cluster_map[lab].append(idx)
    return cluster_map
