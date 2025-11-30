def select_node_from_cluster(cluster_nodes, edge_nodes):
    # choose node with minimum utilization
    best = min(cluster_nodes, key=lambda i: edge_nodes[i]['utilization'])
    return int(best)
