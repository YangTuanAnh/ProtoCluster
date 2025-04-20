#!/bin/bash

DATA_PATH="./data"
OUTPUT_PATH="./output"
TRAIN_DF="sample_train_set.csv"
TEST_DF="sample_test_set.csv"

echo "Data path is: $DATA_PATH"
echo "Output path is: $OUTPUT_PATH"
echo "Train set csv is: $TRAIN_DF"
echo "Test set csv is: $TEST_DF"

# Preprocessing
py -m src.preprocessing --data_path "$DATA_PATH" --train_df "$TRAIN_DF" --test_df "$TEST_DF"

# Stage 1 - Confusion analysis
py -m src.train \
    --data_path "$DATA_PATH" \
    --output_path "$OUTPUT_PATH" \
    --train_df "$TRAIN_DF" \
    --suffix "stage_1" \
    --training \
    --final_eval

py -m src.mistaken_classes \
    --data_path "$DATA_PATH" \
    --output_path "$OUTPUT_PATH" \
    --suffix "stage_1"

py -m src.relabel_train_set \
    --data_path "$DATA_PATH" \
    --output_path "$OUTPUT_PATH" \
    --train_df "$TRAIN_DF" \
    --num_classes 97

# Stage 2 - pretraining
py -m src.train \
    --data_path "$DATA_PATH" \
    --output_path "$OUTPUT_PATH" \
    --train_df "relabeled_$TRAIN_DF" \
    --suffix "stage_2" \
    --training \
    --final_eval \
    --num_classes 61