import torch
from src.models import LocalGCN, GlobalGCN
from tqdm import tqdm
import os
import pandas as pd
from torch.utils.data import DataLoader

from src.dataset import CommunityGraphDataset
from src.models import LocalGCN, GlobalGCN
from src.collate import collate_fn
from src.utils import (
    collect_all_distances,
    get_global_distance_stats,
    preprocess_dataset_global_norm,

)

import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Train GCN model on protein graph data.")
    
    parser.add_argument('--data_path', type=str, default='./data', help='Path to the data directory')
    parser.add_argument('--output_path', type=str, default='./output', help='Path to save model outputs')
    parser.add_argument('--test_csv', type=str, default='train_set.csv', help='Path to train set csv file')
    parser.add_argument('--num_classes', type=int, default=97, help='Number of output classes')
    parser.add_argument('--hidden_layers', type=int, default=128, help='Number of hidden units in GCN')
    parser.add_argument('--suffix', type=str, default='stage_1', help='Optional suffix for model saving')
    parser.add_argument('--protein_id', type=str, default='protein_id', help='Column name for protein ID')
    parser.add_argument('--class_id', type=str, default='class_id', help='Column name for class ID')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    DATA_PATH = args.data_path
    OUTPUT_PATH = args.output_path
    TEST_CSV = args.test_csv
    NUM_CLASSES = args.num_classes
    HIDDEN_LAYERS = args.hidden_layers
    SUFFIX = args.suffix
    PROTEIN_ID = args.protein_id
    CLASS_ID = args.class_id

    df = pd.read_csv(os.path.join(DATA_PATH, TEST_CSV))
    # df = pd.DataFrame({"protein_id": ["0.vtk"]})
    
    all_graphs = df[PROTEIN_ID].map(lambda x: f"{x.split('.')[0]}.pt").tolist()
    all_labels = [-1] * len(all_graphs)

    dataset = CommunityGraphDataset(all_graphs, all_labels, os.path.join(DATA_PATH, "processed_mesh_data"), preload=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_gcn = LocalGCN(in_channels=11, hidden_channels=HIDDEN_LAYERS, out_channels=HIDDEN_LAYERS)
    local_gcn = local_gcn.to(device)
    global_gcn = GlobalGCN(in_channels=HIDDEN_LAYERS, hidden_channels=HIDDEN_LAYERS, num_classes=NUM_CLASSES)
    global_gcn = global_gcn.to(device)

    optimizer = torch.optim.AdamW(list(local_gcn.parameters()) + list(global_gcn.parameters()), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    # Step 1: Collect distances
    all_distances = collect_all_distances(dataset)

    # Step 2: Fit histogram and get min/max (e.g., 1st and 99th percentile)
    global_min, global_max = get_global_distance_stats(all_distances)

    print(f"Global Min: {global_min}, Global Max: {global_max}")

    dataset = preprocess_dataset_global_norm(dataset, global_min, global_max)

    loader = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
    
    suffix = f"_{SUFFIX}" if SUFFIX is not None else ""

    all_preds = []

    local_gcn.eval()
    global_gcn.eval()

    with torch.no_grad():
        for pyg_batch, subgraph_batch_ids, edge_index_C_list, edge_attr_C_list, labels in tqdm(loader, leave=False):
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
    
    pred_df = pd.DataFrame({"protein_id": [item.split('.')[0] for item in all_graphs], "class_id": all_preds})
    pred_df.to_csv(os.path.join(OUTPUT_PATH, f"pred{suffix}.csv"), index=False)