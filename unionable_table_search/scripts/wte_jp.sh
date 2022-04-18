#!/bin/bash
set -e

topic="job_posting"
model_name="web_table_embeddings_combo150"

for threshold in $(seq 0.5 0.1 0.9);
    do
        python test_wte_pylon.py \
        --topic=${topic} \
        --query_file="/ssd/congtj/data/pylon_benchmark/v3/${topic}_queries.txt" \
        --top_k=41 \
        --model_path="/ssd/congtj/web_table_embedding_models/${model_name}.bin" \
        --lsh_threshold=${threshold} \
        && \
        python eval_query_results.py \
        --instance="${model_name}_lsh_${threshold}" \
        --topic=${topic} \
        --num_queries=22 \
        --num_gt=41;
    done