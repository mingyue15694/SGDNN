import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_scatter import scatter_mean
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

        kmeans = KMeans(n_clusters=centers.size(0), init=centers.cpu().numpy(), n_init=1, max_iter=1, random_state=0)
        cluster_labels = kmeans.fit_predict(neighbor_feats.cpu().numpy())
        cluster_labels = torch.tensor(cluster_labels, device=device, dtype=torch.long)

        cluster_agg = scatter_mean(neighbor_feats, cluster_labels, dim=0)
        agg_list = [torch.cat([feats_v, cluster_agg[i]], dim=-1) for i in range(cluster_agg.size(0))]
        cross_agg = torch.stack(agg_list, dim=0).max(dim=0)[0]

        new_features[v] = activation(cross_agg)

    return new_features



class SDGNN_New(nn.Module):
    def __init__(self, in_feats, hidden_dim, k=2, activation=F.relu):
        super(SDGNN_New, self).__init__()
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

class NodeClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, dropout=0.7):
        super(NodeClassifier, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

def train_classifier(features_tensor, labels, train_mask, val_mask, test_mask,
                     device, num_epochs=200, patience=30):
    model = NodeClassifier(
        in_dim=features_tensor.shape[1],
        hidden_dim=128,
        num_classes=int(labels.max().item() + 1),
        dropout=0.7
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        logits_train = model(features_tensor[train_mask])
        loss = criterion(logits_train, labels[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_val = model(features_tensor[val_mask])
            val_loss = criterion(logits_val, labels[val_mask])
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    final_logits = model(features_tensor)
    return model, final_logits

def precompute_communities(G):
    cache = {}
    for v in G.nodes():
        neighbors = list(G.neighbors(v))
        subG = G.subgraph(neighbors)
        cache[v] = list(nx.connected_components(subG))
    return cache

if __name__ == "__main__":
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_list = ["acm", "citeseer", "cocs", "cora", "dblp", "film", "pubmed", "texas"]
    # dataset_list = ["cora"]
    train_percentages = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
    repeat_times = 5

    results = []
    for dname in dataset_list:
        print(f"Processing dataset: {dname}")
        base_path = os.path.join("..", "datasets", dname)
        adj = np.load(os.path.join(base_path, f"{dname}_adj.npy"))
        feat = np.load(os.path.join(base_path, f"{dname}_feat.npy"))
        label = np.load(os.path.join(base_path, f"{dname}_label.npy"))

        edge_index = np.array(np.nonzero(adj))
        data = Data(
            x=torch.tensor(feat, dtype=torch.float).to(device),
            edge_index=torch.tensor(edge_index, dtype=torch.long).to(device),
            y=torch.tensor(label, dtype=torch.long).to(device)
        )
        G = to_networkx(data, to_undirected=True)
        community_cache = precompute_communities(G)

        t_train = time.time()
        # agg_file = f"cached_{dname}_agg.pt"
        # if os.path.exists(agg_file):
        #     aggregated = torch.load(agg_file)
        #     print(f"Loaded cached features for {dname}")
        # else:
        # 在 SDGNN 聚合前添加计时器
        t_agg_start = time.time()

        SDGNN = SDGNN_New(in_feats=data.x.size(1), hidden_dim=128, activation=F.relu).to(device)
        SDGNN.eval()
        with torch.no_grad():
            second_stage_feats, _ = SDGNN(G, data.x, device, community_cache)
        aggregated = torch.stack([second_stage_feats[i] for i in sorted(G.nodes())], dim=0)

        agg_time = time.time() - t_agg_start
        print(f"[{dname}] 聚合时间: {agg_time:.2f}s")

        ids = sorted(G.nodes())
        aggregated = torch.stack([second_stage_feats[i] for i in ids], dim=0)
        # torch.save(aggregated, agg_file)
        print(f"Cached features saved for {dname}")

        num_nodes = data.num_nodes
        for p in train_percentages:
            for r in range(repeat_times):

                train_sz = int(p * num_nodes)
                val_sz = int(0.10 * num_nodes)
                idxs = torch.randperm(num_nodes)
                train_idx = idxs[:train_sz]
                val_idx = idxs[train_sz:train_sz + val_sz]
                test_idx = idxs[train_sz + val_sz:]

                train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
                val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
                test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
                train_mask[train_idx] = True
                val_mask[val_idx] = True
                test_mask[test_idx] = True

                cls_model, final_logits = train_classifier(
                    aggregated.detach(), data.y,
                    train_mask, val_mask, test_mask,
                    device
                )

                train_time = time.time() - t_train
                cls_model.eval()
                with torch.no_grad():
                    pred = final_logits.argmax(dim=1)
                    test_acc = (pred[test_mask] == data.y[test_mask]).float().mean().item()

                print(f"{dname} p={p*100:.0f}% repeat={r+1} test_acc={test_acc:.4f} time={train_time:.2f}s")
                results.append({
                    "Dataset": dname,
                    "Train (%)": p*100,
                    "AggTime": agg_time if (p == 0.05 and r == 0) else 0.0,  # 聚合只记第一次
                    "Repeat": r+1,
                    "TestAcc": test_acc,
                    "Time(s)": train_time
                })

    df = pd.DataFrame(results)
    df['AvgTestAcc'] = df.groupby(['Dataset', 'Train (%)'])['TestAcc'].transform('mean')
    df.to_excel("1OURS_results6_kmeans1.xlsx", index=False)
    print("Saved optimized results.")
    print(f"Total time: {time.time() - start_time:.2f}s")
