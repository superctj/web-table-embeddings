import argparse
import logging
import os
import pickle

import numpy as np


def main(args):
    ground_truth_file = args.ground_truth_file
    result_dir = os.path.join(args.result_dir, args.instance)
    
    output_file = os.path.join(result_dir, "metric.txt")
    logging.basicConfig(filename=output_file, filemode="w", level=logging.INFO, format="[%(levelname)s] %(message)s")

    if args.version == "small" or args.version == "large":
        logging.info(f"TUS {args.version}")
    else:
        logging.info(f"Pylon {args.version}")

    with open(ground_truth_file, "rb") as f:
        ground_truth = pickle.load(f)

    num_queries = len(ground_truth)

    for k in range(0, args.top_k+10, 10):
        if k == 0: k = 1

        all_match = [0] * num_queries
        all_ground_truth = [0] * num_queries

        for i in range(1, num_queries+1):
            result_file = os.path.join(result_dir, f"q{i}.txt")
            with open(result_file, "r") as f:
                query_result = f.readlines()
            
            start = False
            match, count = 0, 0
            for line in query_result:
                if not start:
                    if line.split(" ")[1] == "Target":
                        start = True
                        target = line.split(" ")[-1].rstrip()
                    continue
                
                candidate = line.split(" ")[1].rstrip()
                if candidate in ground_truth[target]["groundtruth"]:
                    match += 1

                count += 1
                if count >= k: # Stop at k
                    break
            
            all_match[i-1] = match
            all_ground_truth[i-1] = ground_truth[target]["recall"]

        logging.info(f"Precision / Recall @{k}")
        precision = sum(all_match) / (k * num_queries)
        recall = np.sum(np.array(all_match) / np.array(all_ground_truth)) / num_queries
        logging.info(f"{precision:.2f} / {recall:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Query Results",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--version", type=str, default="", help="")
    parser.add_argument("--ground_truth_file", type=str, default="", help="")
    parser.add_argument("--result_dir", type=str, default="", help="")
    parser.add_argument("--instance", type=str, default="", help="")
    parser.add_argument("--top_k", type=int, default=0, help="")

    main(parser.parse_args())