"""Main application to run experiments"""

import logging
from logger.tqdm_logging_handler import TqdmLoggingHandler

logging.basicConfig(level=logging.INFO, handlers=[TqdmLoggingHandler()])
logger = logging.getLogger(__name__)

import sys
import getopt
import pandas as pd
import os
import json

from configuration.configuration_reader import ConfigurationReader
from evaluation.information_loss import calculate_normalized_certainty_penalty
from evaluation.partition import get_partition_lengths, calculate_mean_partition_size, calculate_std_partition_size, get_partition_split_share
from kernel.anonymization_kernel import AnonymizationKernel
from nlp.sensitive_terms_recognizer import SensitiveTermsRecognizer
from preprocessing.data_reader import DataReader
from preprocessing.preprocessor import Preprocessor
from pathlib import Path


def main(argv):
    """Main entrypoint for the anonymization tool"""

    # Default parameters
    configuration_file = ''
    input_file = ''
    use_cache = True
    weight = 0.5
    strategy = "gdf"
    result_dir = None

    # Read and set tool parameters
    try:
        opts, _ = getopt.getopt(argv, "c:i:r:w:v", ["config=", "input=", "weight=", "result_dir=", "verbose"])
    except getopt.GetoptError:
        logger.error('experiment_runner.py -c <config_file> -i <input_file> -w <relational_weight>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-c", "--config"):
            configuration_file = arg
        if opt in ("-i", "--input"):
            input_file = arg
            base = os.path.basename(input_file)
            if not result_dir:
                result_dir = os.path.splitext(base)[0]
        if opt in ("-w", "--weight"):
            weight = float(arg)
            strategy = "mondrian"
        if opt in ("-r", "--result_dir"):
            result_dir = arg
        if opt in ("-v", "--verbose"):
            logging.getLogger().setLevel(logging.DEBUG)

    result_path = Path("experiment_results") / result_dir
    result_path.mkdir(parents=True, exist_ok=True)

    # Let's get started
    logger.info("Anonymizing input file %s", input_file)

    # Initialize and read configuration
    configuration_reader = ConfigurationReader()
    config = configuration_reader.read(configuration_file)

    # Read data using data types defined in the configuration
    data_reader = DataReader(config)
    df = data_reader.read(input_file)

    # Initialize the sensitive terms recognizer
    sensitive_terms_recognizer = SensitiveTermsRecognizer(config, use_cache)

    # Initialize the preprocessor (preprocessor is stateful, so pass df at the beginning)
    pp = Preprocessor(sensitive_terms_recognizer, config, df)

    # Run through preprocessing of dataframe: Data cleansing, analysis of textual attributes, resolving of redundant information, and compression
    pp.clean_textual_attributes()
    pp.analyze_textual_attributes()
    pp.find_redundant_information()
    pp.compress()

    # Get sensitive terms dictionary and preprocessed dataframe
    terms = pp.get_sensitive_terms()
    df = pp.get_df()

    # Initialize the anonymization kernel by providing the sensitive terms dictionary, the configuration, the sensitive terms recognizer, and the preprocessor
    kernel = AnonymizationKernel(terms, config, sensitive_terms_recognizer, pp)
    unanonymized = df

    # Determine k values for experiment
    k_values = [2, 3, 4, 5, 10, 20, 50]
    biases = config.get_biases()

    # Set strategy names
    if strategy == "mondrian":
        strategy_name = "mondrian-{}".format(weight)
    elif strategy == "gdf":
        strategy_name = strategy

    # Parameters for calculating metrics
    quasi_identifiers = config.get_quasi_identifiers()
    textual_attribute_mapping = pp.get_textual_attribute_mapping()

    # Prepare dataframes and json to store experiment results
    total_information_loss = pd.DataFrame(index=k_values, columns=[strategy_name])
    total_information_loss.index.name = 'k'

    relational_information_loss = pd.DataFrame(index=k_values, columns=[strategy_name])
    relational_information_loss.index.name = 'k'

    textual_information_loss = pd.DataFrame(index=k_values, columns=[strategy_name])
    textual_information_loss.index.name = 'k'

    detailed_loss_level_0 = [k for k in textual_attribute_mapping]
    detailed_loss_level_1 = set()
    for k in textual_attribute_mapping:
        for e in textual_attribute_mapping[k]:
            detailed_loss_level_1.add(e.replace("{}_".format(k), ''))
    detailed_loss_level_1 = ["total"] + list(detailed_loss_level_1)
    detailed_textual_information_loss = pd.DataFrame(index=k_values, columns=pd.MultiIndex.from_product([detailed_loss_level_0, detailed_loss_level_1]))
    detailed_textual_information_loss.index.name = 'k'

    partition_sizes = {}
    partition_sizes[strategy_name] = {}

    partition_splits = {}
    partition_splits[strategy_name] = {}

    # Let's start the experiments
    for k in k_values:
        logger.info("-------------------------------------------------------------------------------")
        logger.info("Anonymizing dataset with k=%d and strategy %s", k, strategy_name)

        # Anonymize dataset for a specific k
        anonymized_df, partitions, partition_split_statistics = kernel.anonymize_quasi_identifiers(df, k, strategy, biases, weight)

        # Calculating the total, relational, and textual information loss based on the original and anonymized data frame
        total_il, relational_il, textual_il = calculate_normalized_certainty_penalty(unanonymized, anonymized_df, quasi_identifiers, textual_attribute_mapping)

        # Calculating the mean and std for partition size as well as split statistics
        mean_partition_size = calculate_mean_partition_size(partitions)
        std_partition_size = calculate_std_partition_size(partitions)
        if partition_split_statistics:
            number_of_relational_splits, number_of_textual_splits = get_partition_split_share(partition_split_statistics, textual_attribute_mapping)

        # Notify about the results
        logger.info("Information loss for relational attributes is %4.4f", relational_il)
        if textual_il:
            logger.info("Information loss for textual attribute is %4.4f", textual_il["total"])
        logger.info("Total information loss is %4.4f", total_il)
        logger.info("Ended up with %d partitions with a mean size of %.2f and a std of %.2f", len(partitions), mean_partition_size, std_partition_size)
        if partition_split_statistics:
            logger.info("Split %d times on a relational attribute", number_of_relational_splits)
            logger.info("Split %d times on a textual attribute", number_of_textual_splits)

        # Store experiment results
        total_information_loss.at[k, strategy_name] = total_il
        relational_information_loss.at[k, strategy_name] = relational_il
        if textual_il:
            textual_information_loss.at[k, strategy_name] = textual_il["total"]
            for key in textual_il:
                if isinstance(textual_il[key], dict):
                    for subkey in textual_il[key]:
                        if subkey == "total":
                            detailed_textual_information_loss.at[k, (key, "total")] = textual_il[key]["total"]
                        else:
                            entity_type = subkey.replace("{}_".format(key), '')
                            detailed_textual_information_loss.at[k, (key, entity_type)] = textual_il[key][subkey]

        partition_sizes[strategy_name][k] = get_partition_lengths(partitions)
        if partition_split_statistics:
            partition_splits[strategy_name][k] = {
                "relational": number_of_relational_splits,
                "textual": number_of_textual_splits
            }

    # Define file info
    if strategy == "mondrian":
        file_info = str(weight).replace(".", "_")
    elif strategy == "gdf":
        file_info = strategy

    # Save the experiment results
    with open(result_path / 'partition_distribution_{}.json'.format(file_info), 'w') as f:
        json.dump(partition_sizes, f, ensure_ascii=False)

    if partition_split_statistics:
        with open(result_path / 'partition_splits_{}.json'.format(file_info), 'w') as f:
            json.dump(partition_splits, f, ensure_ascii=False)

    total_information_loss.to_csv(result_path / "total_information_loss_{}.csv".format(file_info))
    relational_information_loss.to_csv(result_path / "relational_information_loss_{}.csv".format(file_info))
    if textual_il:
        textual_information_loss.to_csv(result_path / "textual_information_loss_{}.csv".format(file_info))
        detailed_textual_information_loss.to_csv(result_path / "detailed_textual_information_loss_{}.csv".format(file_info))


if __name__ == "__main__":
    main(sys.argv[1:])
