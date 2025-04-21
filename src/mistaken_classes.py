from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import json
import os

import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Graph preprocessing or training configuration.")
    
    parser.add_argument('--min_samples', type=int, default=5, help='Minimum number of samples per class or component')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to the data directory')
    parser.add_argument('--output_path', type=str, default='./output', help='Path to save output')
    parser.add_argument('--suffix', type=str, default='stage_1', help='Suffix to distinguish different stages or runs')
    parser.add_argument('--train_csv', type=str, default='train_set.csv', help='Path to train set csv file')
    parser.add_argument('--protein_id', type=str, default='protein_id', help='Column name for protein ID')
    parser.add_argument('--class_id', type=str, default='class_id', help='Column name for class ID')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    MIN_SAMPLES = args.min_samples
    DATA_PATH = args.data_path
    OUTPUT_PATH = args.output_path
    SUFFIX = args.suffix
    TRAIN_CSV = args.train_csv
    PROTEIN_ID = args.protein_id
    CLASS_ID = args.class_id

    suffix = f"_{SUFFIX}" if SUFFIX is not None else ""

    df_pred = pd.read_csv(os.path.join(OUTPUT_PATH, f"pred{suffix}.csv"))
    train_df = pd.read_csv(os.path.join(DATA_PATH, TRAIN_CSV))
    # train_df = pd.DataFrame({PROTEIN_ID: ["112m_1_A_A_model1",
    #                                         "110m_1_A_A_model1"], CLASS_ID: [39, 39]})
    
    all_preds = df_pred[CLASS_ID].values
    # Extract protein IDs from df_pred
    graph_ids = df_pred[PROTEIN_ID].str.split('.').str[0]

    # Filter train_df where protein_id is in the extracted IDs
    all_true = train_df[train_df[PROTEIN_ID].isin(graph_ids)][CLASS_ID]

    num_classes = 97
    all_labels = list(range(num_classes))  # Ensures all 97 classes are included

    cm = confusion_matrix(all_true, all_preds, labels=all_labels, normalize='true')

    mistakes = {}

    for true_class in range(num_classes):
        # Copy the row and zero out the diagonal (correct prediction)
        row = np.copy(cm[true_class])
        row[true_class] = 0.0

        # Get the top mistaken class (or classes if you want top-N)
        most_mistaken_class = np.argmax(row)
        mistake_strength = row[most_mistaken_class]

        # Save only if the mistake is significant (e.g., > 10%)
        if mistake_strength >= 0.5:
            mistakes[true_class] = (most_mistaken_class, mistake_strength)

    # Display results
    for true_cls, (mistaken_cls, strength) in mistakes.items():
        print(f"Class {true_cls} is often confused with class {mistaken_cls} ({strength:.2%} of the time)")

    from collections import defaultdict

    # Reverse mapping: {mistaken_class: [true_classes]}
    mistaken_groups = defaultdict(list)
    for true_cls, (mistaken_cls, _) in mistakes.items():
        mistaken_groups[int(mistaken_cls)].append(true_cls)    

    # class_counts is count of each class_id in train_df
    class_counts = train_df['class_id'].value_counts().to_dict()

    # Filter mistaken_groups to retain only those where all involved classes have enough samples
    filtered_groups = {}
    for mistaken_cls, true_classes in mistaken_groups.items():
        all_classes = true_classes + [mistaken_cls]
        if sum(class_counts.get(c, 0) >= MIN_SAMPLES for c in all_classes):
            filtered_groups[mistaken_cls] = true_classes

    with open(os.path.join(OUTPUT_PATH, "mistaken_groups.json"), "w") as f:
        json.dump(filtered_groups, f)