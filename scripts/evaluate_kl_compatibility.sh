#!/bin/bash
# MODEL=bert-base-cased

MAX_DATAPOINTS=1000

export DATASETS_VERBOSITY=error

OUTPUT_DIR=results/kl-compatibility
mkdir -p $OUTPUT_DIR

COMMON_ARGS="--use-gpu --batch-size 256 --max-datapoints $MAX_DATAPOINTS"

# Checks if $BLOCK_CONTIGUOUS is set to true, the mode is enabled. If it is unset or set to something other than `true` (e.g., `false`) it is disabled
block_contiguous_flag=""
block_contiguous_file_flag=""
# compatibility_file="results/compatibility-layer/$MODEL.pair.pkl"
compatibility_file="results/compatibility-layer-additional/backup/$MODEL.pair.bert-top-k.fwd.better-normalization.pkl"
if [[ "$BLOCK_CONTIGUOUS" == "true" ]]; then
    block_contiguous_flag="--block-contiguous"
    block_contiguous_file_flag=".contiguous"
    # compatibility_file="results/compatibility-layer/$MODEL.pair_contiguous.pkl"
    compatibility_file="results/compatibility-layer-additional/backup/$MODEL.pair_contiguous.bert-top-k.fwd.better-normalization.pkl"
fi

# Checks if $JOINT_SIZE is set. If it is unset or set to something other than `true` (e.g., `false`) it is disabled
restricted_flag=""
restricted_file_flag=""
schemes="naive mrf mrf-local hcb-both iter"
if [[ -v JOINT_SIZE ]]; then
    restricted_flag="--joint-size $JOINT_SIZE"
    restricted_file_flag=".top-$JOINT_SIZE"
    schemes="$schemes compatibility"
fi

# NOTE: Must include --export=ALL,MODEL={bert-base-cased,bert-large-cased,roberta-base,roberta-large},DATASET={snli,xsum},BLOCK_CONTIGUOUS={true,false}
echo "Model: $MODEL"
echo "Block contiguous: $BLOCK_CONTIGUOUS"
echo "Dataset: $DATASET"
echo "Joint size: $JOINT_SIZE"

cmd="python evaluate_kl_compatibility.py $MODEL --schemes $schemes $COMMON_ARGS --dataset $DATASET --output-file $OUTPUT_DIR/$MODEL.$DATASET.${MAX_DATAPOINTS}${block_contiguous_file_flag}${restricted_file_flag}.pkl $block_contiguous_flag $restricted_flag --compatibility-layer $compatibility_file"
echo "$cmd"
eval "$cmd"

