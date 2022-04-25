#!/bin/bash
set -e

version="small"
model_name="web_table_embeddings_combo150"
num_samples=1000

for threshold in $(seq 0.7 0.1 0.7);
    do
        python test_wte_tus.py \
        --source_dir="/ssd/congtj/data/table-union-search-benchmark/table_csvs/${version}_benchmark/" \
        --target_dir="/ssd/congtj/data/table-union-search-benchmark/table_csvs/${version}_benchmark/" \
        --index_dir="/home/congtj/web-table-embeddings/unionable_table_search/indexes_tus/${version}/" \
        --output_dir="/home/congtj/web-table-embeddings/unionable_table_search/results_tus/${version}/" \
        --query_file="/ssd/congtj/data/table-union-search-benchmark/table_csvs/${version}_groundtruth/recall_groundtruth.csv" \
        --model_path="/ssd/congtj/web_table_embedding_models/${model_name}.bin" \
        --lsh_threshold=${threshold} \
        --num_samples=${num_samples} \
        && \
        python eval_tus_query_results.py \
        --version="${version}" \
        --ground_truth_file="/ssd/congtj/data/table-union-search-benchmark/${version}_groundtruth.pkl" \
        --result_dir="/home/congtj/web-table-embeddings/unionable_table_search/results_tus/${version}/" \
        --instance="${model_name}_sample_${num_samples}_lsh_${threshold}"
    done