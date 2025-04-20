import torch
from torch.utils.data import Dataset
import os
from tqdm import tqdm

class CommunityGraphDataset(Dataset):
    def __init__(self, graph_paths, labels, data_dir, preload=False):
        self.graph_paths = graph_paths
        self.labels = labels
        self.data_dir = data_dir
        self.preload = preload

        if self.preload:
            self.data = []
            for path, label in tqdm(zip(self.graph_paths, self.labels)):
                full_path = os.path.join(self.data_dir, path)
                with open(full_path, "rb") as f:
                    community_data, C = torch.load(f, weights_only=False)
                self.data.append((community_data, C, label))

    def __len__(self):
        return len(self.graph_paths)

    def __getitem__(self, idx):
        if self.preload:
            return self.data[idx]
        else:
            graph_file = os.path.join(self.data_dir, self.graph_paths[idx])
            label = self.labels[idx]
            with open(graph_file, "rb") as f:
                community_data, C = torch.load(f, weights_only=False)
            return community_data, C, label