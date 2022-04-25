#!/bin/bash
set -e

version="v3.2"
model_name="web_table_embeddings_combo150"
k=350
num_samples=100000

for threshold in $(seq 0.7 0.1 0.7);
    do
        # python test_wte_pylon_topk.py \
        # --source_dir="/ssd/congtj/data/pylon_benchmark/${version}/source/" \
        # --target_dir="/ssd/congtj/data/pylon_benchmark/${version}/source/" \
        # --index_dir="/home/congtj/web-table-embeddings/unionable_table_search/indexes/${version}/" \
        # --output_dir="/home/congtj/web-table-embeddings/unionable_table_search/results/${version}/" \
        # --query_file="/ssd/congtj/data/pylon_benchmark/${version}/all_queries.txt" \
        # --model_path="/ssd/congtj/web_table_embedding_models/${model_name}.bin" \
        # --lsh_threshold=${threshold} \
        # --num_samples=${num_samples} \
        # --top_k=${k} \
        # && \
        python eval_query_results_topk.py \
        --version="${version}" \
        --ground_truth_file="/ssd/congtj/data/pylon_benchmark/${version}/all_ground_truth.pkl" \
        --result_dir="/home/congtj/web-table-embeddings/unionable_table_search/results/${version}/" \
        --instance="${model_name}_sample_${num_samples}_lsh_${threshold}_topk_${k}" \
        --top_k=${k}
    done