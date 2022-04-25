#!/bin/bash
set -e

topic="music"
model_name="web_table_embeddings_combo150"
num_samples=10

for threshold in $(seq 0.7 0.1 0.7);
    do
        python test_wte_pylon.py \
        --topic=${topic} \
        --query_file="/ssd/congtj/data/pylon_benchmark/v3.11/${topic}_queries.txt" \
        --top_k=48 \
        --model_path="/ssd/congtj/web_table_embedding_models/${model_name}.bin" \
        --num_samples=${num_samples} \
        --lsh_threshold=${threshold} \
        && \
        python eval_query_results.py \
        --benchmark_version="v3.11" \
        --instance="${model_name}_sample_${num_samples}_lsh_${threshold}" \
        --topic=${topic} \
        --num_queries=16 \
        --num_gt=48;
    done