import pandas as pd
import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import gc

from dataset import CommunityGraphDataset
from models import LocalGCN, GlobalGCN
from collate import collate_fn
from utils import (
    train_val_split,
    collect_all_distances,
    get_global_distance_stats,
    preprocess_dataset_global_norm,

)

if __name__ == "__main__":
    DATA_PATH = "./data"
    EPOCHS = 1
    TRAINING = False
    # train_df = pd.read_csv(os.path.join(DATA_PATH, "train_set.csv"))
    # test_df = pd.read_csv(os.path.join(DATA_PATH, "test_set.csv"))
    train_df = pd.DataFrame({"protein_id": ["112m_1_A_A_model1",
                                            "110m_1_A_A_model1"], "class_id": [39, 39]})
    test_df = pd.DataFrame({"anonymised_protein_id": ["0.vtk"]})

    train_files = train_df["protein_id"].tolist()

    test_files = test_df["anonymised_protein_id"].tolist()

    all_graphs = sorted(os.listdir(os.path.join(DATA_PATH, "processed_mesh_data")))

    id_to_class = dict(zip(train_df["protein_id"], train_df["class_id"]))

    all_labels = [
        int(id_to_class[graph.split(".")[0]])
        for graph in train_files
    ]

    graphs_train, labels_train, graphs_val, labels_val = train_val_split(all_graphs, all_labels)

    train_dataset = CommunityGraphDataset(graphs_train, labels_train, os.path.join(DATA_PATH, "processed_mesh_data"), preload=True)
    val_dataset = CommunityGraphDataset(graphs_val, labels_val, os.path.join(DATA_PATH, "processed_mesh_data"), preload=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_gcn = LocalGCN(in_channels=11, hidden_channels=128, out_channels=128)
    local_gcn = local_gcn.to(device)
    global_gcn = GlobalGCN(in_channels=128, hidden_channels=128, num_classes=97)
    global_gcn = global_gcn.to(device)

    optimizer = torch.optim.AdamW(list(local_gcn.parameters()) + list(global_gcn.parameters()), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    # Step 1: Collect distances
    all_distances = collect_all_distances(train_dataset)

    # Step 2: Fit histogram and get min/max (e.g., 1st and 99th percentile)
    global_min, global_max = get_global_distance_stats(all_distances)

    print(f"Global Min: {global_min}, Global Max: {global_max}")

    train_dataset = preprocess_dataset_global_norm(train_dataset, global_min, global_max)
    val_dataset = preprocess_dataset_global_norm(val_dataset, global_min, global_max)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)

    if TRAINING:
        gc.collect()
        torch.cuda.empty_cache()

        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []

        for epoch in range(1, EPOCHS + 1):
            local_gcn.train()
            global_gcn.train()

            epoch_train_loss = 0.0
            correct_train = 0

            for pyg_batch, subgraph_batch_ids, edge_index_C_list, edge_attr_C_list, labels in tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False):
                x = pyg_batch.x.to(device)
                edge_index = pyg_batch.edge_index.to(device)
                batch = pyg_batch.batch.to(device)

                # Local GCN forward
                subgraph_feats = local_gcn(x, edge_index, batch)

                # Reconstruct community graphs
                num_communities = max(subgraph_batch_ids) + 1
                C_feats = [[] for _ in range(num_communities)]
                for i, cid in enumerate(subgraph_batch_ids):
                    C_feats[cid].append(subgraph_feats[i])

                outputs, all_labels = [], []
                for feats, edge_index_C, edge_attr_C, label in zip(C_feats, edge_index_C_list, edge_attr_C_list, labels):
                    x_C = torch.stack(feats).to(device)
                    edge_index_C = edge_index_C.to(device)
                    edge_attr_C = edge_attr_C.to(device)
                    out = global_gcn(x_C, edge_index_C, edge_attr_C)
                    outputs.append(out)
                    all_labels.append(torch.tensor(label, dtype=torch.long, device=device))

                outputs = torch.stack(outputs)
                label_tensor = torch.stack(all_labels)

                loss = criterion(outputs, label_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item() * len(labels)
                correct_train += (outputs.argmax(dim=1) == label_tensor).sum().item()

            avg_train_loss = epoch_train_loss / len(train_loader.dataset)
            train_acc = correct_train / len(train_loader.dataset)
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_acc)

            # Validation
            local_gcn.eval()
            global_gcn.eval()
            epoch_val_loss = 0.0
            correct_val = 0

            with torch.no_grad():
                for pyg_batch, subgraph_batch_ids, edge_index_C_list, edge_attr_C_list, labels in tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
                    x = pyg_batch.x.to(device)
                    edge_index = pyg_batch.edge_index.to(device)
                    batch = pyg_batch.batch.to(device)

                    subgraph_feats = local_gcn(x, edge_index, batch)

                    num_communities = max(subgraph_batch_ids) + 1
                    C_feats = [[] for _ in range(num_communities)]
                    for i, cid in enumerate(subgraph_batch_ids):
                        C_feats[cid].append(subgraph_feats[i])

                    outputs, all_labels = [], []
                    for feats, edge_index_C, edge_attr_C, label in zip(C_feats, edge_index_C_list, edge_attr_C_list, labels):
                        x_C = torch.stack(feats).to(device)
                        edge_index_C = edge_index_C.to(device)
                        edge_attr_C = edge_attr_C.to(device)
                        out = global_gcn(x_C, edge_index_C, edge_attr_C)
                        outputs.append(out)
                        all_labels.append(torch.tensor(label, dtype=torch.long, device=device))

                    outputs = torch.stack(outputs)
                    label_tensor = torch.stack(all_labels)

                    loss = criterion(outputs, label_tensor)
                    epoch_val_loss += loss.item() * len(labels)
                    correct_val += (outputs.argmax(dim=1) == label_tensor).sum().item()

            avg_val_loss = epoch_val_loss / len(val_loader.dataset)
            val_acc = correct_val / len(val_loader.dataset)
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_acc)

            print(f"Epoch {epoch:3d}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, "
                f"Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")
            
        torch.save(local_gcn.state_dict(), 'local_gcn_weights_stage_1.pth')
        torch.save(global_gcn.state_dict(), 'global_gcn_weights_stage_1.pth')
    
    local_gcn.load_state_dict(torch.load('local_gcn_weights_stage_1.pth'))
    global_gcn.load_state_dict(torch.load('global_gcn_weights_stage_1.pth'))

    all_preds = []
    all_true = []

    local_gcn.eval()
    global_gcn.eval()

    with torch.no_grad():
        for pyg_batch, subgraph_batch_ids, edge_index_C_list, edge_attr_C_list, labels in tqdm(val_loader, leave=False):
            x = pyg_batch.x.to(device)
            edge_index = pyg_batch.edge_index.to(device)
            batch = pyg_batch.batch.to(device)

            # Local GCN output
            subgraph_feats = local_gcn(x, edge_index, batch)

            num_communities = max(subgraph_batch_ids) + 1
            C_feats = [[] for _ in range(num_communities)]
            for i, cid in enumerate(subgraph_batch_ids):
                C_feats[cid].append(subgraph_feats[i])

            for feats, edge_index_C, edge_attr_C, label in zip(C_feats, edge_index_C_list, edge_attr_C_list, labels):
                x_C = torch.stack(feats).to(device)
                edge_index_C = edge_index_C.to(device)
                edge_attr_C = edge_attr_C.to(device)

                # Global feature extraction with edge attributes
                x1 = global_gcn.conv1(x_C, edge_index_C, edge_attr=edge_attr_C).relu()
                x2 = global_gcn.conv2(x1, edge_index_C, edge_attr=edge_attr_C).relu()

                out = global_gcn.classifier(x2.mean(dim=0).to(device))
                pred = out.argmax(dim=-1).cpu().item()

                all_preds.append(pred)
                all_true.append(label)
    
    pred_df = pd.DataFrame({"graph_id": graphs_val, "pred": all_preds, "true": all_true})
    pred_df.to_csv("pred_stage_1.csv")