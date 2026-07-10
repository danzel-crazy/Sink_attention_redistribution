#!/bin/bash
CUDA_VISIBLE_DEVICES='0'
gpu_list="${1:-3}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

MODEL_PATH=${MODEL_PATH:-"liuhaotian/llava-v1.5-7b"}
DATA_DIR="/project/aimm/pp/MTK/eval"
SPLIT="llava_vqav2_mscoco_test-dev2015"
QUESTION_FILE="${DATA_DIR}/vqav2/${SPLIT}.jsonl"
IMAGE_FOLDER="${DATA_DIR}/vqav2/test2015"
ANSWER_FILE="/tmp2/danzel/attention-bias/SparseVLMs/playground/data/eval/vqav2/answers/192/$CKPT/${CHUNKS}_${IDX}.jsonl"
SUBMISSION_FILE="/tmp2/danzel/attention-bias/SparseVLMs/playground/data/eval/vqav2/answers/192/$CKPT/merge.jsonl"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 -m llava.eval.model_vqa_loader \
        --model-path $MODEL_PATH \
        --question-file $QUESTION_FILE \
        --image-folder $IMAGE_FOLDER \
        --answers-file $ANSWER_FILE \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=$SUBMISSION_FILE

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $ANSWER_FILE >> "$output_file"
done

python3 scripts/convert_vqav2_for_submission.py --split "192" --ckpt $CKPT

