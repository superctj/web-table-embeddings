import argparse
import logging
import os
import shutil

from d3l.indexing.similarity_indexes import NameIndex, FormatIndex, ValueIndex, WtEmbeddingIndex, DistributionIndex
from d3l.input_output.dataloaders import CSVDataLoader
from d3l.querying.query_engine import QueryEngine
from d3l.utils.functions import pickle_python_object, unpickle_python_object
from tqdm import tqdm

from util import custom_logger


def aggregate_func(similarity_scores):
    avg_score = sum(similarity_scores) / len(similarity_scores)
    return avg_score


def main(args):
    # CSV data loader
    dataloader = CSVDataLoader(
        root_path=args.source_dir,
        sep=",", 
        lineterminator="\n" # this is crucial see https://stackoverflow.com/questions/33998740/error-in-reading-a-csv-file-in-pandascparsererror-error-tokenizing-data-c-err
    )

    # name_index_path = os.path.join(args.index_dir, "name.lsh")
    # if os.path.exists(name_index_path):
    #     name_index = unpickle_python_object(name_index_path)
    #     print("Name Index: LOADED!")
    # else:
    #     print("Name Index: STARTED!")
    #     name_index = NameIndex(dataloader=dataloader)
    #     pickle_python_object(name_index, name_index_path)
    #     print("Name Index: SAVED!")

    # if os.path.exists('./indexes_d3l/format.lsh'):
    #     format_index = unpickle_python_object('./indexes_d3l/format.lsh')
    #     print("Format: Loaded!")
    # else:
    #     print("Format: Started!")
    #     format_index = FormatIndex(dataloader=dataloader)
    #     pickle_python_object(format_index, './indexes_d3l/format.lsh')
    #     print("Format: SAVED!")

    # if os.path.exists('./indexes_d3l/value.lsh'):
    #     value_index = unpickle_python_object('./indexes_d3l/value.lsh')
    #     print("Value: Loaded!")
    # else:
    #     print("Value: Started!")
    #     value_index = ValueIndex(dataloader=dataloader)
    #     pickle_python_object(value_index, './indexes_d3l/value.lsh')
    #     print("Value: SAVED!")

    # embedding_index_path = os.path.join(args.index_dir, "embedding.lsh")
    # if os.path.exists(embedding_index_path):
    #     embedding_index = unpickle_python_object(embedding_index_path)
    #     print("Embedding: Loaded!")
    # else:
    #     print("Embedding: Started!")
    #     embedding_index = EmbeddingIndex(dataloader=dataloader, index_cache_dir="./")
    #     pickle_python_object(embedding_index, embedding_index_path)
    #     print("Embedding: SAVED!")
    
    # if os.path.exists('./indexes_d3l/distribution.lsh'):
    #     distribution_index = unpickle_python_object('./indexes_d3l/distribution.lsh')
    #     print("Distribution: Loaded!")
    # else:
    #     print("Distribution: Started!")
    #     distribution_index = DistributionIndex(dataloader=dataloader)
    #     pickle_python_object(distribution_index, './indexes_d3l/distribution.lsh')
    #     print("Distribution: SAVED!")

    model_name = args.model_path.split("/")[-1][:-4]
    embedding_index_name = f"{model_name}_sample_{args.num_samples}_lsh_{args.lsh_threshold}"

    if not os.path.exists(args.index_dir):
        os.makedirs(args.index_dir)
    embedding_index_path = os.path.join(args.index_dir, f"{embedding_index_name}.lsh")
    
    if os.path.exists(embedding_index_path):
        embedding_index = unpickle_python_object(embedding_index_path)
        print(f"{embedding_index_name} Embedding Index: LOADED!")
    else:
        print(f"{embedding_index_name} Embedding Index: STARTED!")

        embedding_index = WtEmbeddingIndex(
            model_path=args.model_path,
            num_samples=args.num_samples,
            dataloader=dataloader,
            index_similarity_threshold=args.lsh_threshold,
            index_cache_dir="./")
        pickle_python_object(embedding_index, embedding_index_path)
        
        print(f"{embedding_index_name} Embedding Index: SAVED!")
    
    query_dataloader = CSVDataLoader(
        root_path=args.target_dir,
        sep=",",
        lineterminator="\n"
    )

    qe = QueryEngine(embedding_index)

    with open(args.query_file, "r") as f:
        query_recall_pairs = f.readlines()[1:] # skip the header
    
    output_dir = os.path.join(args.output_dir, embedding_index_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        # shutil.rmtree(output_dir)
        # os.makedirs(output_dir)
        raise ValueError("Output directory already exists!")

    for i, query_recall in enumerate(tqdm(query_recall_pairs)):
        output_file = os.path.join(output_dir, f"q{i+1}.txt")
        logger = custom_logger(output_file, level=logging.INFO)
        logger.info(args) # For reproduction

        query, recall = query_recall.split(",")
        query = query[:-4]
        recall = int(recall.rstrip())
        
        logger.info(f"Target table: {query}")
        query_table = query_dataloader.read_table(table_name=query)
        results = qe.table_query(table=query_table, aggregator=aggregate_func, k=recall, verbose=False)

        for res in results:
            logger.info(f"{res[0]} {str(res[1])}")


if __name__ == "__main__":
    # "paper_publication", "job_posting"
    parser = argparse.ArgumentParser(description="D3L for Top-K Table Search",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--source_dir", type=str, default="/ssd/congtj/data/table-union-search-benchmark/table_csvs/small_benchmark/", help="")
    parser.add_argument("--target_dir", type=str, default="/ssd/congtj/data/table-union-search-benchmark/table_csvs/small_benchmark/", help="")
    parser.add_argument("--index_dir", type=str, default="/home/congtj/web-table-embeddings/unionable_table_search/indexes_tus/small/", help="")
    parser.add_argument("--output_dir", type=str, default="/home/congtj/web-table-embeddings/unionable_table_search/results_tus/small/", help="") # tus_small_326_clr_sample_10
    parser.add_argument("--query_file", type=str, default="/ssd/congtj/data/table-union-search-benchmark/table_csvs/small_groundtruth/recall_groundtruth.csv", help="")

    parser.add_argument("--model_path", type=str, default="/ssd/congtj/web_table_embedding_models/web_table_embeddings_combo150.bin", help="")
    parser.add_argument("--lsh_threshold", type=float, default=0.7, help="")
    parser.add_argument("--num_samples", type=int, default=5, help="Maximum number of rows to sample from each table to construct embeddings.")
    
    main(parser.parse_args())
