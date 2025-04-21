#!/bin/bash

# ========== Configuration ==========
DATA_PATH="./sample_data"
OUTPUT_PATH="./output"
WEIGHTS_PATH="./weights"
TRAIN_CSV="train_set.csv"
TEST_CSV="test_set.csv"

# ========== ANSI Color Codes ==========
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

# ========== Summary ==========
echo -e "${BOLD}${YELLOW}🔧 Configuration:${RESET}"
echo -e "📁 Data path   : ${GREEN}$DATA_PATH${RESET}"
echo -e "📁 Output path : ${GREEN}$OUTPUT_PATH${RESET}"
echo -e "📄 Train CSV   : ${GREEN}$TRAIN_CSV${RESET}"
echo -e "📄 Test CSV    : ${GREEN}$TEST_CSV${RESET}"
echo ""

mkdir -p "$DATA_PATH"
mkdir -p "$OUTPUT_PATH"

section "📤 Submission Inference"
start_time=$(date +%s)

python -m src.preprocessing \
    --data_path "$DATA_PATH" \
    --input_csv "$TEST_CSV" \
    --split "test" \
    --protein_id "anonymised_protein_id"

python -m src.eval \
    --data_path "$DATA_PATH" \
    --output_path "$OUTPUT_PATH" \
    --test_csv "$TEST_CSV" \
    --suffix "inference" \
    --protein_id "anonymised_protein_id" \
    --load_local_gcn "$WEIGHTS_PATH/local_gcn_weights_stage_3.pth" \
    --load_global_gcn "$WEIGHTS_PATH/global_gcn_weights_stage_3.pth"

cp "$OUTPUT_PATH/pred_stage_3.csv" "$OUTPUT_PATH/submission.csv"

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo -e "${GREEN}⏱️ Submission inference completed in ${elapsed} seconds.${RESET}"
echo -e "${GREEN}${BOLD}✅ Done!${RESET} 🚀"