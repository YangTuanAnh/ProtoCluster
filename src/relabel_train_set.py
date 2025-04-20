import json
import os
import pandas as pd

def build_sink_grouped_superclass_map(confusion_groups, num_original_classes):
    superclass_map = {}
    assigned = set()
    superclass_id = 0

    for sink, mistaken in confusion_groups.items():
        group = [sink] + mistaken
        for c in group:
            superclass_map[c] = superclass_id
            assigned.add(c)
        superclass_id += 1

    # Assign remaining classes to their own superclasses
    for c in range(num_original_classes):
        if c not in assigned:
            superclass_map[c] = superclass_id
            superclass_id += 1

    return superclass_map

def relabel_dataset_to_superclasses(labels, superclass_map):
    return [superclass_map[label] for label in labels]

import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Configure paths and class count for processing or training.")

    parser.add_argument('--data_path', type=str, default='./data', help='Path to the input data directory')
    parser.add_argument('--output_path', type=str, default='./output', help='Path to the output directory')
    parser.add_argument('--train_df', type=str, default='train_set.csv', help='Path to train set csv file')
    parser.add_argument('--num_classes', type=int, default=97, help='Number of output classes')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    DATA_PATH = args.data_path
    OUTPUT_PATH = args.output_path
    TRAIN_DF = args.train_df
    NUM_CLASSES = args.num_classes
    
    with open(os.path.join(OUTPUT_PATH, "mistaken_groups.json"), "r") as f:
        mistaken_group = json.load(f)

    train_df = pd.read_csv(os.path.join(DATA_PATH, TRAIN_DF))
    # train_df = pd.DataFrame({"protein_id": ["112m_1_A_A_model1",
    #                                         "110m_1_A_A_model1"], "class_id": [39, 39]})
    
    all_labels = train_df['class_id'].values

    superclass_map = build_sink_grouped_superclass_map(mistaken_group, NUM_CLASSES)
    superclass_labels = relabel_dataset_to_superclasses(all_labels, superclass_map)

    relabeled_df = pd.DataFrame({"protein_id": train_df['protein_id'].values, "class_id": superclass_labels})
    relabeled_df.to_csv(os.path.join(DATA_PATH, f"relabeled_{TRAIN_DF}"), index=False)