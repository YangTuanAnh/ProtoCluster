import os
import torch
import numpy as np
import networkx as nx
import pyvista as pv
from tqdm import tqdm
from torch_geometric.utils import from_networkx
from networkx.algorithms.community import greedy_modularity_communities
import pandas as pd

# Precompute sin/cos values for positional encoding
def create_encoding_functions(num_encoding_dims=12):
    encoders = []
    for i in range(num_encoding_dims // 6):
        for fn_type in [np.sin, np.cos]:
            for j in range(3):  # x, y, z
                freq = 1.0 / (10000 ** (2 * i / (num_encoding_dims // 3)))
                encoders.append((j, fn_type, freq))
    return encoders

# Global encoder functions - compute once
ENCODERS = create_encoding_functions()

def positional_encoding_3d(coord, num_encoding_dims=12):
    encoding = np.zeros(num_encoding_dims)
    for i, (dim, fn, freq) in enumerate(ENCODERS[:num_encoding_dims]):
        encoding[i] = fn(coord[dim] * freq)
    return encoding.tolist()

def build_graph(mesh_file, DATA_PATH, split="train", reduction=0.999):
    vtk_file = os.path.join(DATA_PATH, f"{split}_set_vtk.tar", mesh_file)
    mesh = pv.read(vtk_file)

    # Downsample the mesh if reduction > 0:
    if reduction > 0:
        mesh = mesh.decimate_pro(reduction, preserve_topology=True)

    # Extract data once to avoid repeated dictionary lookups
    points = mesh.points
    normals = mesh.point_data['Normals']
    potential = mesh.point_data['Potential']
    normal_potential = mesh.point_data['NormalPotential']

    # Pre-compute encoded positions for all points
    encoded_positions = np.zeros((len(points), 6))
    encoded_positions[:, 0:3] = np.sin(points)
    encoded_positions[:, 3:6] = np.cos(points)

    # Create graph
    G = nx.Graph()

    # Add nodes more efficiently
    for i, (point, normal, pot, norm_pot) in enumerate(zip(points, normals, potential, normal_potential)):
        G.add_node(i,
                  pos=point.tolist(),
                  encoded_pos=encoded_positions[i].tolist(),
                  normals=normal.tolist(),
                  potential=float(pot),
                  normal_potential=float(norm_pot))

    # Process faces more efficiently
    faces = mesh.faces.reshape((-1, 4))
    # Filter out non-triangular faces first
    triangle_faces = faces[faces[:, 0] == 3]

    # Add edges in bulk
    edges = []
    for face in triangle_faces[:, 1:]:
        a, b, c = face
        edges.extend([(a, b), (b, c), (c, a)])

    G.add_edges_from(edges)
    return G

def build_community_graph(G):
    # 1. Community detection
    communities = list(greedy_modularity_communities(G))
    num_communities = len(communities)

    # 2. Map each node to its community (use array for faster lookup)
    max_node_id = max(G.nodes())
    node_community = np.full(max_node_id + 1, -1, dtype=np.int32)

    for cid, community in enumerate(communities):
        for node in community:
            node_community[node] = cid

    community_subgraphs = {
        cid: G.subgraph(community).copy()
        for cid, community in enumerate(communities)
    }

    # 3. Build community graph
    C = nx.Graph()
    C.add_nodes_from(range(num_communities))

    # 4. Compute aggregated features for each community
    community_centers = {}

    for cid, community in enumerate(communities):
        community_list = list(community)

        # Batch extract node attributes for better performance
        coords = np.array([G.nodes[n]['pos'] for n in community_list])
        potentials = np.array([G.nodes[n]['potential'] for n in community_list])
        normals = np.array([G.nodes[n]['normals'] for n in community_list])
        normal_potentials = np.array([G.nodes[n]['normal_potential'] for n in community_list])

        # Center of mass
        center = coords.mean(axis=0)
        community_centers[cid] = center

        # Assign features to node in C
        C.nodes[cid]['center'] = center.tolist()
        C.nodes[cid]['mean_potential'] = float(potentials.mean())
        C.nodes[cid]['mean_normal'] = normals.mean(axis=0).tolist()
        C.nodes[cid]['mean_normal_potential'] = float(normal_potentials.mean())
        C.nodes[cid]['positional_encoding'] = positional_encoding_3d(center)

    # 5. Add edges between communities
    # First, collect all inter-community edges
    inter_community_edges = {}

    for u, v in G.edges():
        cu = node_community[u]
        cv = node_community[v]
        if cu != cv and cu != -1 and cv != -1:
            edge_key = (min(cu, cv), max(cu, cv))  # Ensure unique edges
            inter_community_edges[edge_key] = True

    # Then add edges with distance calculation
    for cu, cv in inter_community_edges:
        dist = np.linalg.norm(community_centers[cu] - community_centers[cv])
        C.add_edge(cu, cv, distance=dist)

    return C, communities, community_subgraphs

def extract_node_features(G, node_list=None):
    if node_list is None:
        node_list = list(G.nodes)

    features = []
    for n in node_list:
        pos = G.nodes[n]['pos']
        encoded_pos = np.concatenate([np.sin(pos), np.cos(pos)])
        normal = G.nodes[n]['normals']
        potential = G.nodes[n]['potential']
        normal_potential = G.nodes[n]['normal_potential']
        feat = np.concatenate([encoded_pos, normal, [potential], [normal_potential]])
        features.append(feat)

    return torch.tensor(features, dtype=torch.float32)

def process_mesh_data(train_df, DATA_PATH, split="train", file_id="protein_id", batch_size=10):
    # Create output directory if it doesn't exist
    os.makedirs(os.path.join(DATA_PATH, "processed_mesh_data"), exist_ok=True)

    # Process in batches to manage memory better
    for start_idx in tqdm(range(0, len(train_df), batch_size)):
        batch_df = train_df.iloc[start_idx:min(start_idx + batch_size, len(train_df))]

        for _, row in batch_df.iterrows():
            mesh_file = row[file_id]
            if not mesh_file.endswith(".vtk"):
                mesh_file += ".vtk"

            # Skip if already processed
            out_path = os.path.join(DATA_PATH, "processed_mesh_data", row[file_id] + ".pt")
            if os.path.exists(out_path):
                continue

            try:
                G = build_graph(mesh_file, DATA_PATH, split=split)
                C, communities, community_subgraphs = build_community_graph(G)

                community_data = []

                for cid, community in enumerate(communities):
                    # Use the pre-computed subgraph
                    subgraph = community_subgraphs[cid]

                    # Relabel nodes to consecutive integers
                    node_map = {old: new for new, old in enumerate(subgraph.nodes())}
                    relabeled = nx.relabel_nodes(subgraph, node_map, copy=True)

                    # Extract node features
                    x = extract_node_features(relabeled)

                    # Build edge index directly as a tensor
                    edges = list(subgraph.edges())
                    if edges:  # Only process if edges exist
                        # Map old node IDs to new IDs
                        mapped_edges = [(node_map[u], node_map[v]) for u, v in edges]

                        # Create tensor and make undirected
                        edge_index = torch.tensor(mapped_edges, dtype=torch.long).t().contiguous()
                        edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
                    else:
                        # Empty edge tensor with correct shape
                        edge_index = torch.zeros((2, 0), dtype=torch.long)

                    community_data.append({'x': x, 'edge_index': edge_index})

                # Save data
                with open(out_path, "wb") as f:
                    torch.save((community_data, C), f)

            except Exception as e:
                print(f"Error processing {mesh_file}: {e}")

# Usage
if __name__ == "__main__":
    DATA_PATH = "./data"
    # train_df = pd.read_csv(os.path.join(DATA_PATH, "train_set.csv"))
    train_df = pd.DataFrame({"protein_id": ["112m_1_A_A_model1",
                                            "110m_1_A_A_model1"], "class_id": [39, 39]})
    process_mesh_data(train_df, DATA_PATH)

    test_df = pd.DataFrame({"anonymised_protein_id": ["0.vtk"]})
    process_mesh_data(test_df, DATA_PATH, split="test", file_id="anonymised_protein_id")