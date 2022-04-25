import argparse
import logging
import os
import pickle


def main(args):
    ground_truth_file = args.ground_truth_file
    result_dir = os.path.join(args.result_dir, args.instance)
    
    output_file = os.path.join(result_dir, "metric.txt")
    logging.basicConfig(filename=output_file, filemode="w", level=logging.INFO, format="[%(levelname)s] %(message)s")

    with open(ground_truth_file, "rb") as f:
        ground_truth = pickle.load(f)

    num_queries = len(ground_truth)
    all_match = [0] * num_queries
    all_ground_truth = [0] * num_queries

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
            if candidate in ground_truth[target]["groundtruth"]:
                match += 1
        
        all_match[i-1] = match
        all_ground_truth[i-1] = ground_truth[target]["recall"]

    logging.info("Precision/Recall @ Ground Truth")
    num_attempts = sum(all_ground_truth)
    logging.info(f"TUS {args.version}: {sum(all_match) / num_attempts : .2f}")

    for i in range(1, num_queries+1):
        logging.info(f"  Query {i}: {all_match[i-1]} / {all_ground_truth[i-1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Query Results",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--version", type=str, default="", help="")
    parser.add_argument("--ground_truth_file", type=str, default="", help="")
    parser.add_argument("--result_dir", type=str, default="", help="")
    parser.add_argument("--instance", type=str, default="", help="")

    main(parser.parse_args())