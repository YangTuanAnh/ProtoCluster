from torch_geometric.data import Batch as PyGBatch

def collate_fn(batch):
    """
    Custom collate function that properly handles community graph data with edge attributes.
    """
    all_subgraphs = []
    subgraph_batch_ids = []
    edge_index_C_list = []
    edge_attr_C_list = []
    labels = []

    for i, (community_data, edge_index_C, edge_attr_C, label) in enumerate(batch):
        # Append all subgraphs for PyG batching
        all_subgraphs.extend(community_data)
        subgraph_batch_ids.extend([i] * len(community_data))

        edge_index_C_list.append(edge_index_C)
        edge_attr_C_list.append(edge_attr_C)
        labels.append(label)

    if len(all_subgraphs) == 0:
        raise ValueError("No subgraphs found in batch")

    # Convert subgraphs to PyG batch
    try:
        pyg_batch = PyGBatch.from_data_list(all_subgraphs)
    except AttributeError:
        from torch_geometric.data import Data
        if isinstance(all_subgraphs[0], dict):
            all_subgraphs = [Data(**sg) for sg in all_subgraphs]
        pyg_batch = PyGBatch.from_data_list(all_subgraphs)

    return pyg_batch, subgraph_batch_ids, edge_index_C_list, edge_attr_C_list, labels