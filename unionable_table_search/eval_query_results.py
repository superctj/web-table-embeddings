import argparse
import logging
import os
import pickle


def main(args):
    # [("paper_publication", 34, 50), ("job_posting", 30, 50), ("paper_publication_restr", 20, 35), ("job_posting_restr", 22, 41)]
    version = args.benchmark_version
    instance = args.instance
    topic = args.topic
    num_queries = args.num_queries
    num_gt = args.num_gt

    result_dir = f"/home/congtj/web-table-embeddings/unionable_table_search/results/{version}/{instance}/{topic}/"
    ground_truth_file = f"/ssd/congtj/data/pylon_benchmark/{version}/{topic}_ground_truth.pkl"
    
    output_file = os.path.join(result_dir, "metric.txt")
    logging.basicConfig(filename=output_file, filemode="w", level=logging.INFO, format="[%(levelname)s] %(message)s")

    with open(ground_truth_file, "rb") as f:
        ground_truth = pickle.load(f)

    all_match = [0] * num_queries
    for i in range(1, num_queries+1):
        result_file = os.path.join(result_dir, f"q{i}.txt")
        with open(result_file, "r") as f:
            query_result = f.readlines()
        
        start = False
        match = 0
        for line in query_result:
            if not start:
                if line.split(" ")[1] == "Target":
                    start = True
                    target = line.split(" ")[-1].rstrip()
                continue
            
            candidate = line.split(" ")[1].rstrip()
            if candidate in ground_truth[target] or candidate == target:
                match += 1
        
        all_match[i-1] = match

    logging.info("Precision/Recall @ Ground Truth")
    num_attempts = num_queries * num_gt
    logging.info(f"{topic}: {sum(all_match) / num_attempts : .2f}")

    for i in range(1, num_queries+1):
        logging.info(f"  Query {i}: {all_match[i-1]} / {num_gt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Query Results",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--benchmark_version", type=str, default="v3", help="")
    parser.add_argument("--instance", type=str, default="web_table_embeddings_combo150_lsh_0.7", help="")
    parser.add_argument("--topic", type=str, default="paper_publication", help="") # "paper_publication"
    parser.add_argument("--num_queries", type=int, default=20, help="")
    parser.add_argument("--num_gt", type=int, default=35, help="")

    main(parser.parse_args())
