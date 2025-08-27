import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_scatter import scatter_mean

# ==== GPU 版 SGC (计算 S^K 和 G_new) ====
def compute_propagation_graph_gpu(edge_index, num_nodes, k=2, device="cuda"):
    # 构造 (A + I) 稀疏邻接
    self_loops = torch.arange(num_nodes, device=device).unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, self_loops], dim=1)
    values = torch.ones(edge_index.size(1), device=device)
    A = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes))

    # 计算 D^{-1/2}
    deg = torch.sparse.sum(A, dim=1).to_dense()
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.

    row, col = edge_index
    norm_vals = deg_inv_sqrt[row] * values * deg_inv_sqrt[col]
    S = torch.sparse_coo_tensor(edge_index, norm_vals, (num_nodes, num_nodes))

    # 计算 S^K
    S_dense = S.to_dense()  # 直接转 dense，避免多次稀疏乘
    S_k = torch.matrix_power(S_dense, k)

    # 生成二值新图邻接 (CPU 交给 networkx)
    S_bin = (S_k > 0).float().cpu().numpy()
    G_new = nx.from_numpy_array(S_bin.astype(np.uint8))

    return S_k, G_new

# ==== 社区聚合 (新图 G_new) ====
def gnn_first_layer_aggregation(G, features_tensor, activation, device, community_cache):
    N, D = features_tensor.size()
    new_features = torch.zeros((N, 2 * D), device=device)

    for v in G.nodes():
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

# ==== 主模型 ====
class SDGNN_SGC(nn.Module):
    def __init__(self, activation=F.relu, k=2):
        super(SDGNN_SGC, self).__init__()
        self.activation = activation
        self.k = k

    def forward(self, edge_index, features_tensor, device):
        num_nodes = features_tensor.size(0)
        # 1. GPU 上计算 S^K 和新图
        S_k, G_new = compute_propagation_graph_gpu(edge_index, num_nodes, k=self.k, device=device)

        # 2. 传播特征 (S^K @ X)
        propagated_feats = torch.matmul(S_k.to(device), features_tensor)

        # 3. 新图上做社区划分
        community_cache = {}
        for v in G_new.nodes():
            neighbors = list(G_new.neighbors(v))
            subG = G_new.subgraph(neighbors)
            community_cache[v] = list(nx.connected_components(subG)) if neighbors else []
        enhanced_feats=propagated_feats
        for layer in range(1):
        # 4. 社区聚合 (新图)
            enhanced_feats = gnn_first_layer_aggregation(G_new, enhanced_feats, self.activation, device, community_cache)
        return enhanced_feats

# ==== 简单分类器 ====
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

def train_classifier(x, y, train_mask, val_mask, test_mask, device, epochs=200, patience=30):
    model = NodeClassifier(x.shape[1], 128, int(y.max().item() + 1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    best_loss, patience_counter, best_state = float('inf'), 0, None

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(x[train_mask]), y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x[val_mask]), y[val_mask])
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    with torch.no_grad():
        pred = model(x).argmax(dim=1)
        acc = (pred[test_mask] == y[test_mask]).float().mean().item()
    return acc

# ==== 主程序 ====
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = ["acm", "citeseer", "cocs", "cora", "dblp", "film", "pubmed", "texas"]
    # datasets = [ "cora"]
    train_ratios = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
    repeat_times = 5
    results = []

    for name in datasets:
        path = os.path.join("..", "datasets", name)
        feat = np.load(os.path.join(path, f"{name}_feat.npy"))
        adj = np.load(os.path.join(path, f"{name}_adj.npy"))
        label = np.load(os.path.join(path, f"{name}_label.npy"))

        edge_index = torch.tensor(np.array(np.nonzero(adj)), dtype=torch.long, device=device)
        x = torch.tensor(feat, dtype=torch.float32, device=device)
        y = torch.tensor(label, dtype=torch.long, device=device)

        # === 聚合 ===
        t0 = time.time()
        model = SDGNN_SGC().to(device)
        model.eval()
        with torch.no_grad():
            agg_x = model(edge_index, x, device)
        agg_time = time.time() - t0
        print(f"[{name}] 聚合时间: {agg_time:.2f}s")

        # === 分类训练 ===
        for p in train_ratios:
            for r in range(repeat_times):
                num_nodes = x.size(0)
                idx = torch.randperm(num_nodes, device=device)
                n_train = int(p * num_nodes)
                n_val = int(0.1 * num_nodes)
                train_idx, val_idx, test_idx = idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]

                train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
                val_mask = torch.zeros_like(train_mask)
                test_mask = torch.zeros_like(train_mask)
                train_mask[train_idx], val_mask[val_idx], test_mask[test_idx] = True, True, True

                acc = train_classifier(agg_x, y, train_mask, val_mask, test_mask, device)
                results.append({
                    "Dataset": name,
                    "Train (%)": p * 100,
                    "Repeat": r + 1,
                    "TestAcc": acc,
                    "AggTime": agg_time if (p == 0.05 and r == 0) else 0.0
                })
                print(f"{name} p={p * 100:.0f}% r={r + 1} acc={acc:.4f} Aggtime={agg_time:.4f}")

    df = pd.DataFrame(results)
    df["AvgAcc"] = df.groupby(["Dataset", "Train (%)"])["TestAcc"].transform("mean")
    df.to_excel("1OURS_results6_sgc.xlsx", index=False)
    print("已保存结果")
