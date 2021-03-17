
import sys
from copy import deepcopy
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter

from feature_generator import FeatureGenerator
from feature_preparation import FeaturePreparation
from web_table_multi_embedding_model import WebTableMultiEmbeddingModel
from fasttext_model import FastTextModel
from random_embedding_model import RandomEmbeddingModel
from classifier import DECOClassifier
from one_vs_all_classifier import DECOClassifierOVA
from random_forest_classifier import RFClassifier
from voting_classifier import VotingClassifier
from table_export import TableExport
from scoring import Scoring

SPLITTING = [0.4, 0.1, 0.5]
DEFAULT_ITERATIONS = 1
RANDOM_FEATURE_DIM = 100
USE_NORMALIZATION = False
OUTPUT_TYPE = 'sqlite'  # json or sqlite


def create_arg_parser():
    parser = ArgumentParser("evaluate_classifier",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve',
                            description='''Evaluates the DECO classifier.''')
    parser.add_argument('-i', '--input',
                        help="path to pickle file with feature data (generated by featureGenerator.py)", required=True, nargs=1)
    parser.add_argument('-e', '--embedding-model',
                        help="path to fastText embedding model", required=False, nargs='*')
    parser.add_argument('-o', '--output',
                        help="path for output txt file", required=False, nargs=1)
    parser.add_argument('-f', '--features',
                        help="types of features: 'embeddings', 'deco', or 'combined'", required=True, nargs='*')
    parser.add_argument('-t', '--embedding-type',
                        help="embedding type: 'web-table', 'fasttext', or 'random'", required=True, nargs=1)
    parser.add_argument('-it', '--iterations',
                        help="number of evaluation iterations", required=False, nargs=1)
    parser.add_argument('-c', '--classifier',
                        help="types of classifiers: 'gnn', 'gnn-ova', or 'random-forest'", required=True, nargs='*')

    return parser

def evaluate(args):
    print('Load embedding model ...')
    embedding_model = None
    if args.embedding_type[0].lower() == 'web-table':
        if args.embedding_model is None:
            print('ERROR: No embedding model file argument')
            quit()
        embedding_model = WebTableMultiEmbeddingModel(args.embedding_model)
    elif args.embedding_type[0].lower() == 'fasttext':
        if args.embedding_model is None:
            print('ERROR: No embedding model file argument')
            quit()
        embedding_model = FastTextModel(args.embedding_model[0])
    elif args.embedding_type[0].lower() == 'random':
        embedding_model = RandomEmbeddingModel(RANDOM_FEATURE_DIM)
    else:
        print('ERROR: Unknown embedding model:', args.embedding_type[0])
        quit()

    print('Load features from python pickle file ...')
    feature_generator = FeatureGenerator(pickle_file_path=args.input[0])
    feature_preparation = FeaturePreparation(
        feature_generator, embedding_model)

    print('Evaluation loop:')
    iterations = int(
        args.iterations[0]) if args.iterations is not None else DEFAULT_ITERATIONS
    for i in range(iterations):
        print('Iteration:', i)
        print('Sample sheets ...')
        [train_sheets, valid_sheets,
            test_sheets], graphs = feature_preparation.sample_sheets(SPLITTING)
        print('Construct DGL graph ...')
        dgl_graph = feature_preparation.construct_dgl_graph(graphs)
        # get labels
        print('Create label lookup for nodes in the DGL graph ...')
        label_lookup, labels = feature_preparation.create_node_label_lookup(
            dgl_graph)
        # create feature and label vectors
        print(
            'Create vectors for node ids and vectors for labels (inputs for classifier) ...')
        train_ids, train_labels, valid_ids, valid_labels, test_ids, test_labels, class_weights = feature_preparation.create_mx_arrays(
            dgl_graph, label_lookup, set(train_sheets), set(valid_sheets), set(test_sheets), downsampling=False)
        classifiers = [None] * len(args.classifier)
        all_features = []
        for j in range(len(args.classifier)):
            classifier_type = args.classifier[j].lower()
            feature_type = args.features[j].lower()
            # get fetaures for graph
            print('Construct feature vectors for DGL graph ...')
            features = feature_preparation.construct_features_for_dgl_graph(
                dgl_graph, feature_type)
            print('Add vector vectors to nodes in the DGL graph ...')
            annotated_dgl_graph = feature_preparation.add_features_to_graph(
                deepcopy(dgl_graph), features, normalization=USE_NORMALIZATION)
            # create classifier
            print('Create', classifier_type, 'classifier object ...')
            if classifier_type == 'gnn':
                classifiers[j] = DECOClassifier(
                    annotated_dgl_graph, features, len(labels), class_weights)
            elif classifier_type == 'gnn-ova':
                classifiers[j] = DECOClassifierOVA(
                    annotated_dgl_graph,  features, len(labels))
            elif classifier_type == 'random-forest':
                classifiers[j] = RFClassifier(features, len(labels))
            else:
                print('ERROR: Unkown classifier type:', classifier_type)
            # training
            print('Train classifier ...')
            classifiers[j].train(train_ids, train_labels,
                                 valid_ids, valid_labels)
            all_features.append(features)
        classifier_names = list(args.classifier)
        if len(classifiers) > 1:
            # add voting classifier
            classifiers.append(VotingClassifier(
                list(classifiers), all_features, feature_preparation, labels))
            classifier_names.append('VotingClassifier')
        for j, classifier in enumerate(classifiers):
            # classify
            print('Evaluate classification accuracy on test data ...')
            _, score = classifier.evaluate(valid_ids, valid_labels)
            print('Valid Accuracy (' + classifier_names[j] + '):', score)
            pred, score = classifier.evaluate(test_ids, test_labels)
            print('Test Accuracy (' + classifier_names[j] + '):', score)
            scoring = Scoring(test_ids, test_labels, pred, labels)
            scores = scoring.get_scores()
            if (args.output is not None) and (len(args.output) == 1):
                table_exporter = TableExport(
                    test_ids, pred, dgl_graph, feature_generator.node_attributes, feature_generator.features, labels, scores, labeling=test_labels)
                if OUTPUT_TYPE == 'json':
                    export_filename = args.output[0] + '-' + \
                        classifier_names[j] + str(i) + '.json'
                    table_exporter.export_tables(export_filename)
                elif OUTPUT_TYPE == 'sqlite':
                    db_filename = args.output[0] + '-' + \
                        classifier_names[j] + '.sqlite'
                    if i == 0:
                        table_exporter.clear_sqlite_db(db_filename)
                    table_exporter.export_tables_as_sqlite(db_filename, i)
                else:
                    print('ERROR: Unknown output type:', OUTPUT_TYPE)
    return

def main():
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()
    if len(args.classifier) != len(args.features):
        print('ERROR: number of classifiers has to be equal to number of feature types')
        quit()

    evaluate(args)
    return


if __name__ == "__main__":
    main()