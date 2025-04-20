from collections import Counter
import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def train_val_split(all_graphs, all_labels):
    # First, identify classes with only 2 instances
    label_counts = Counter(all_labels)
    rare_classes = [label for label, count in label_counts.items() if count == 2]

    # Initialize empty lists for train/test splits
    graphs_train, graphs_val = [], []
    labels_train, labels_val = [], []

    # Handle rare classes first (exactly 2 instances)
    for rare_class in rare_classes:
        # Find indices of the rare class
        rare_indices = np.where(np.array(all_labels) == rare_class)[0]

        # Put one instance in train, one in test
        graphs_train.append(all_graphs[rare_indices[0]])
        labels_train.append(all_labels[rare_indices[0]])

        graphs_val.append(all_graphs[rare_indices[1]])
        labels_val.append(all_labels[rare_indices[1]])

    # Get remaining data (excluding already distributed rare classes)
    rare_indices_set = set([i for c in rare_classes for i in np.where(np.array(all_labels) == c)[0]])
    remaining_indices = [i for i in range(len(all_labels)) if i not in rare_indices_set]

    remaining_graphs = [all_graphs[i] for i in remaining_indices]
    remaining_labels = [all_labels[i] for i in remaining_indices]

    # Split the remaining data with stratification
    if remaining_graphs:  # Only if there are remaining samples
        rem_graphs_train, rem_graphs_val, rem_labels_train, rem_labels_val = train_test_split(
            remaining_graphs, remaining_labels,
            test_size=0.2,
            random_state=42,
            stratify=remaining_labels
        )

        # Combine the rare class samples with the remaining samples
        graphs_train.extend(rem_graphs_train)
        graphs_val.extend(rem_graphs_val)
        labels_train.extend(rem_labels_train)
        labels_val.extend(rem_labels_val)

    return graphs_train, labels_train, graphs_val, labels_val

def collect_all_distances(dataset):
    all_distances = []
    for _, C, _ in tqdm(dataset):
        all_distances.extend([C[u][v].get("distance", 0) for u, v in C.edges])
    return torch.tensor(all_distances, dtype=torch.float)

def get_global_distance_stats(all_distances, clip_percentiles=(1, 99)):
    # Use percentiles to avoid extreme outliers
    lower = np.percentile(all_distances.numpy(), clip_percentiles[0])
    upper = np.percentile(all_distances.numpy(), clip_percentiles[1])
    return lower, upper

def preprocess_dataset_global_norm(dataset, global_min, global_max):
    processed = []
    for community_data, C, label in tqdm(dataset):
        cid_map = {cid: idx for idx, cid in enumerate(C.nodes)}
        edge_index_C = torch.tensor(
            [(cid_map[u], cid_map[v]) for u, v in C.edges],
            dtype=torch.long
        ).t().contiguous()

        # Extract and globally normalize distances
        distances = [C[u][v].get("distance", 0.0) for u, v in C.edges]
        distances = torch.tensor(distances, dtype=torch.float)

        # Clip and normalize
        distances = torch.clamp(distances, min=global_min, max=global_max)
        if global_max > global_min:
            distances = (distances - global_min) / (global_max - global_min)
        else:
            distances = torch.zeros_like(distances)

        edge_attr = distances.unsqueeze(1)  # shape (num_edges, 1)
        processed.append((community_data, edge_index_C, edge_attr, label))
    return processed