import pandas as pd
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import gc
import matplotlib.pyplot as plt

from src.dataset import CommunityGraphDataset
from src.models import LocalGCN, GlobalGCN
from src.collate import collate_fn
from src.utils import (
    train_val_split,
    collect_all_distances,
    get_global_distance_stats,
    preprocess_dataset_global_norm,

)
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Train GCN model on protein graph data.")
    
    parser.add_argument('--data_path', type=str, default='./data', help='Path to the data directory')
    parser.add_argument('--output_path', type=str, default='./output', help='Path to save model outputs')
    parser.add_argument('--train_csv', type=str, default='train_set.csv', help='Path to train set csv file')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--num_classes', type=int, default=97, help='Number of output classes')
    parser.add_argument('--hidden_layers', type=int, default=128, help='Number of hidden units in GCN')
    parser.add_argument('--training', action='store_true', help='Flag to enable training')
    parser.add_argument('--suffix', type=str, default='stage_1', help='Optional suffix for model saving')
    parser.add_argument('--train_all', action='store_true', help='Train on the full dataset')
    parser.add_argument('--final_eval', action='store_true', help='Release predictions on val split')
    parser.add_argument('--protein_id', type=str, default='protein_id', help='Column name for protein ID')
    parser.add_argument('--class_id', type=str, default='class_id', help='Column name for class ID')
    parser.add_argument('--load_local_gcn', type=str, default=None, help='Load pretrained local GCN weights')
    parser.add_argument('--load_global_gcn', type=str, default=None, help='Load pretrained global GCN weights')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    DATA_PATH = args.data_path
    OUTPUT_PATH = args.output_path
    TRAIN_CSV = args.train_csv
    EPOCHS = args.epochs
    NUM_CLASSES = args.num_classes
    HIDDEN_LAYERS = args.hidden_layers
    TRAINING = args.training
    SUFFIX = args.suffix
    TRAIN_ALL = args.train_all
    FINAL_EVAL = args.final_eval
    PROTEIN_ID = args.protein_id
    CLASS_ID = args.class_id
    LOAD_LOCAL_GCN = args.load_local_gcn
    LOAD_GLOBAL_GCN = args.load_global_gcn

    TRAIN_CSV = pd.read_csv(os.path.join(DATA_PATH, TRAIN_CSV))
    # TRAIN_CSV = pd.DataFrame({"protein_id": ["112m_1_A_A_model1",
    #                                         "110m_1_A_A_model1"], "class_id": [39, 39]})

    all_graphs = TRAIN_CSV[PROTEIN_ID].map(lambda x: f"{x.split('.')[0]}.pt").tolist()
    all_labels = TRAIN_CSV[CLASS_ID].tolist()

    if TRAIN_ALL:
        graphs_TRAIN_ALLtrain, labels_train, graphs_val, labels_val = all_graphs, all_labels, [], []
    else:
        graphs_train, labels_train, graphs_val, labels_val = train_val_split(all_graphs, all_labels)

    train_dataset = CommunityGraphDataset(graphs_train, labels_train, os.path.join(DATA_PATH, "processed_mesh_data"), preload=True)
    val_dataset = CommunityGraphDataset(graphs_val, labels_val, os.path.join(DATA_PATH, "processed_mesh_data"), preload=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_gcn = LocalGCN(in_channels=11, hidden_channels=HIDDEN_LAYERS, out_channels=HIDDEN_LAYERS)
    local_gcn = local_gcn.to(device)
    global_gcn = GlobalGCN(in_channels=HIDDEN_LAYERS, hidden_channels=HIDDEN_LAYERS, num_classes=NUM_CLASSES)
    global_gcn = global_gcn.to(device)

    if LOAD_LOCAL_GCN:
        local_gcn.load_state_dict(torch.load(LOAD_LOCAL_GCN, map_location="cpu"))
    
    if LOAD_GLOBAL_GCN:
        checkpoint = torch.load(LOAD_GLOBAL_GCN, map_location="cpu")

        # Get classifier output size from checkpoint
        ckpt_classifier_weight = checkpoint.get('classifier.weight', None)
        ckpt_num_classes = ckpt_classifier_weight.shape[0] if ckpt_classifier_weight is not None else None

        if ckpt_num_classes == NUM_CLASSES:
            print("Loading all weights including classifier.")
            global_gcn.load_state_dict(checkpoint)
        else:
            print(f"Classifier mismatch: checkpoint has {ckpt_num_classes} classes, expected {NUM_CLASSES}.")
            print("Loading all weights except classifier.")

            # Filter out classifier weights
            filtered_state_dict = {k: v for k, v in checkpoint.items() if not k.startswith('classifier')}
            global_gcn.load_state_dict(filtered_state_dict, strict=False)

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
    
    suffix = f"_{SUFFIX}" if SUFFIX is not None else ""

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
        
        if TRAIN_ALL:
            print(f"Epoch {epoch:3d}: Train Loss = {avg_train_loss:.4f}, Train Acc = {train_acc:.4f}")
            continue

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
            
    if not TRAIN_ALL:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(val_accuracies, label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()

        plt.savefig(os.path.join(OUTPUT_PATH, f"loss_history{suffix}.pdf"), dpi=200)
        df_training_stats = pd.DataFrame({"train_loss": train_losses,
                                            "val_loss": val_losses,
                                            "train_acc": train_accuracies,
                                            "val_acc": val_accuracies})
        df_training_stats.to_csv(os.path.join(OUTPUT_PATH, f"loss_history{suffix}.csv"))

    torch.save(local_gcn.state_dict(), os.path.join(OUTPUT_PATH, f'local_gcn_weights{suffix}.pth'))
    torch.save(global_gcn.state_dict(), os.path.join(OUTPUT_PATH, f'global_gcn_weights{suffix}.pth'))

    if FINAL_EVAL:
        all_preds = []

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
        
        pred_df = pd.DataFrame({"protein_id": [item.split('.')[0] for item in graphs_val], "class_id": labels_val})
        pred_df.to_csv(os.path.join(OUTPUT_PATH, f"pred{suffix}.csv"), index=False)