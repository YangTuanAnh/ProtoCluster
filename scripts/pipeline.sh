#!/bin/bash

DATA_PATH="./sample_data"
OUTPUT_PATH="./output"
TRAIN_CSV="train_set.csv"
TEST_CSV="test_set.csv"

# ANSI color codes
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
MAGENTA='\033[1;35m'
RESET='\033[0m'
BOLD='\033[1m'

divider() {
    echo -e "${MAGENTA}------------------------------------------------------------${RESET}"
}

section() {
    divider
    echo -e "${BOLD}${CYAN}$1${RESET}"
    divider
}

echo -e "${BOLD}${YELLOW}🔧 Configuration:${RESET}"
echo -e "📁 Data path      : ${GREEN}$DATA_PATH${RESET}"
echo -e "📁 Output path    : ${GREEN}$OUTPUT_PATH${RESET}"
echo -e "📄 Train CSV      : ${GREEN}$TRAIN_CSV${RESET}"
echo -e "📄 Test CSV       : ${GREEN}$TEST_CSV${RESET}"
echo ""

mkdir -p "$DATA_PATH"
mkdir -p "$OUTPUT_PATH"

section "📦 Extracting train_set_vtk.tar.gz"
if [ -f "$DATA_PATH/train_set_vtk.tar.gz" ]; then
    tar --format=posix -xzf "$DATA_PATH/train_set_vtk.tar.gz" -C "$DATA_PATH"
else
    echo -e "${YELLOW}⚠️ train_set_vtk.tar.gz not found in $DATA_PATH${RESET}"
fi

section "⚙️ Preprocessing (Training)"
py -m src.preprocessing --data_path "$DATA_PATH" --input_csv "$TRAIN_CSV"

section "🧪 Stage 1 - Confusion Analysis"
py -m src.train \
    --data_path "$DATA_PATH" \
    --output_path "$OUTPUT_PATH" \
    --train_csv "$TRAIN_CSV" \
    --suffix "stage_1" \
    --training \
    --final_eval

py -m src.mistaken_classes \
    --data_path "$DATA_PATH" \
    --output_path "$OUTPUT_PATH" \
    --suffix "stage_1" \
    --train_csv "$TRAIN_CSV"

py -m src.relabel_train_set \
    --data_path "$DATA_PATH" \
    --output_path "$OUTPUT_PATH" \
    --train_csv "$TRAIN_CSV" \
    --num_classes 97

section "🧠 Stage 2 - Pretraining"
py -m src.train \
    --data_path "$DATA_PATH" \
    --output_path "$OUTPUT_PATH" \
    --train_csv "relabeled_$TRAIN_CSV" \
    --suffix "stage_2" \
    --training \
    --final_eval \
    --num_classes 97

section "🎯 Stage 3 - Finetuning"
py -m src.train \
    --data_path "$DATA_PATH" \
    --output_path "$OUTPUT_PATH" \
    --train_csv "relabeled_$TRAIN_CSV" \
    --suffix "stage_3" \
    --training \
    --final_eval \
    --num_classes 97 \
    --load_local_gcn "$OUTPUT_PATH/local_gcn_weights_stage_2.pth" \
    --load_global_gcn "$OUTPUT_PATH/global_gcn_weights_stage_2.pth"

section "📦 Extracting test_set_vtk.tar.gz"
if [ -f "$DATA_PATH/test_set_vtk.tar.gz" ]; then
    tar --format=posix -xzf "$DATA_PATH/test_set_vtk.tar.gz" -C "$DATA_PATH"
else
    echo -e "${YELLOW}⚠️ test_set_vtk.tar.gz not found in $DATA_PATH${RESET}"
fi

section "📤 Submission Inference"

start_time=$(date +%s)

py -m src.preprocessing \
    --data_path "$DATA_PATH" \
    --input_csv "$TEST_CSV" \
    --split "test" \
    --protein_id "anonymised_protein_id"

py -m src.eval \
    --data_path "$DATA_PATH" \
    --output_path "$OUTPUT_PATH" \
    --test_csv "$TEST_CSV" \
    --suffix "stage_3" \
    --protein_id "anonymised_protein_id"

cp "$OUTPUT_PATH/pred_stage_3.csv" "$OUTPUT_PATH/submission.csv"

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo -e "${GREEN}⏱️ Submission inference completed in ${elapsed} seconds.${RESET}"

echo -e "${GREEN}${BOLD}✅ Done!${RESET} 🚀"
