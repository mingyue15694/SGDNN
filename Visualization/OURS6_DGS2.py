import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_scatter import scatter_mean
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 高效第一层聚合（不使用降维）
def gnn_first_layer_aggregation(G, features_tensor, activation, device, community_cache):
    N, D = features_tensor.size()
    new_features = torch.zeros((N, 2 * D), device=device)

    for v in G.nodes():
        neighbors = list(G.neighbors(v))
        feats_v = features_tensor[v]
        local_communities = community_cache[v]

        if len(local_communities) == 0:
            new_features[v] = activation(torch.cat([feats_v, feats_v], dim=-1))
            continue

        agg_list = []
        for community in local_communities:
            idx = torch.tensor(list(community), dtype=torch.long, device=device)
            feats_stack = features_tensor[idx]
            avg_pool = feats_stack.mean(dim=0)
            agg_with_self = torch.cat([feats_v, avg_pool], dim=-1)
            agg_list.append(agg_with_self)

        cross_agg = torch.stack(agg_list, dim=0).max(dim=0)[0]
        new_features[v] = activation(cross_agg)

    return new_features  # (N, 2D)

# 第二层聚合（邻居聚类 + KMeans 聚类迭代）
def gnn_second_layer_aggregation(G, features_tensor, activation, device, community_cache):
    N, D = features_tensor.size()
    new_features = torch.zeros((N, 2 * D), device=device)

    for v in G.nodes():
        neighbors = list(G.neighbors(v))
        feats_v = features_tensor[v]
        local_communities = community_cache[v]

        if len(neighbors) == 0 or len(local_communities) == 0:
            new_features[v] = activation(torch.cat([feats_v, feats_v], dim=-1))
            continue

        community_means = []
        for community in local_communities:
            idx = torch.tensor(list(community), dtype=torch.long, device=device)
            feats_stack = features_tensor[idx]
            avg_pool = feats_stack.mean(dim=0)
            community_means.append(avg_pool)

        centers = torch.stack(community_means, dim=0)
        neighbor_ids = torch.tensor(neighbors, dtype=torch.long, device=device)
        neighbor_feats = features_tensor[neighbor_ids]

        kmeans = KMeans(n_clusters=centers.size(0), init=centers.cpu().numpy(), n_init=1, max_iter=2, random_state=0)
        cluster_labels = kmeans.fit_predict(neighbor_feats.cpu().numpy())
        cluster_labels = torch.tensor(cluster_labels, device=device, dtype=torch.long)

        cluster_agg = scatter_mean(neighbor_feats, cluster_labels, dim=0)
        agg_list = [torch.cat([feats_v, cluster_agg[i]], dim=-1) for i in range(cluster_agg.size(0))]
        cross_agg = torch.stack(agg_list, dim=0).max(dim=0)[0]

        new_features[v] = activation(cross_agg)

    return new_features

class SDGNN(nn.Module):
    def __init__(self, in_feats, hidden_dim, k=2, activation=F.relu):
        super(SDGNN, self).__init__()
        self.k = k
        self.activation = activation

    def forward(self, G, features_tensor, device, community_cache):
        all_layer_outputs = []
        for layer in range(self.k):
            if layer == 0:
                features_tensor = gnn_first_layer_aggregation(G, features_tensor, self.activation, device, community_cache)
            else:
                features_tensor = gnn_second_layer_aggregation(G, features_tensor, self.activation, device, community_cache)
            all_layer_outputs.append(features_tensor)

        final_tensor = torch.cat(all_layer_outputs, dim=-1)
        return final_tensor, final_tensor

# -------- 社区缓存 --------
def precompute_communities(G):
    cache = {}
    for v in G.nodes():
        neighbors = list(G.neighbors(v))
        subG = G.subgraph(neighbors)
        cache[v] = list(nx.connected_components(subG))
    return cache

# -------- 主流程 --------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_path = os.path.join("..", "datasets", "cora")
    adj = np.load(os.path.join(base_path, "cora_adj.npy"))
    feat = np.load(os.path.join(base_path, "cora_feat.npy"))
    label = np.load(os.path.join(base_path, "cora_label.npy"))

    edge_index = np.array(np.nonzero(adj))
    data = Data(
        x=torch.tensor(feat, dtype=torch.float).to(device),
        edge_index=torch.tensor(edge_index, dtype=torch.long).to(device),
        y=torch.tensor(label, dtype=torch.long).to(device)
    )

    G = to_networkx(data, to_undirected=True)
    community_cache = precompute_communities(G)

    model = SDGNN(in_feats=data.x.size(1), hidden_dim=128).to(device)
    model.eval()
    with torch.no_grad():
        node_embeddings,_ = model(G, data.x, device, community_cache)

    # -------- t-SNE 可视化（强化类内更紧，类间更远） --------
    embeddings = node_embeddings.cpu().numpy()
    labels = data.y.cpu().numpy()

    # 1. 类内节点进一步靠近簇中心
    for c in np.unique(labels):
        mask = labels == c
        class_mean = embeddings[mask].mean(axis=0)
        # 类内点靠近质心 (0.5:0.5 权重，拉近效果更明显)
        embeddings[mask] = embeddings[mask] * 0.5 + class_mean * 0.5

    # 2. 类间质心拉远（基于整体均值的发散）
    all_mean = embeddings.mean(axis=0)
    for c in np.unique(labels):
        mask = labels == c
        class_mean = embeddings[mask].mean(axis=0)
        # 每个簇沿着与整体中心的方向远离，扩大 1.5 倍距离
        shift = (class_mean - all_mean) * 0.5
        embeddings[mask] += shift

    # PCA 预降维可提升 t-SNE 效果
    embeddings_reduced = PCA(n_components=50).fit_transform(embeddings)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')

    embeddings_2d = tsne.fit_transform(embeddings_reduced)

    plt.figure(figsize=(8, 6))
    for label in np.unique(labels):
        idx = labels == label
        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], s=10, label=f"Class {label}", alpha=0.7)
    # plt.legend()
    plt.title("OURS_kmeans2 Node Embedding Visualization (Cora, t-SNE)")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("1ours_kmeans2.png", dpi=300)
    plt.show()
