"""Main application for the anon CLI tool"""

import logging
from logger.tqdm_logging_handler import TqdmLoggingHandler

logging.basicConfig(level=logging.INFO, handlers=[TqdmLoggingHandler()])
logger = logging.getLogger(__name__)

import sys
import getopt

from configuration.configuration_reader import ConfigurationReader
from evaluation.information_loss import calculate_normalized_certainty_penalty
from evaluation.partition import calculate_mean_partition_size, calculate_std_partition_size, get_partition_split_share
from kernel.anonymization_kernel import AnonymizationKernel
from nlp.sensitive_terms_recognizer import SensitiveTermsRecognizer
from postprocessing.postprocessor import PostProcessor
from preprocessing.data_reader import DataReader
from preprocessing.preprocessor import Preprocessor


def main(argv):
    """Main entrypoint for the anonymization tool"""

    # Default parameters
    configuration_file = ''
    input_file = ''
    output_file = ''
    use_cache = False

    # Read and set tool parameters
    try:
        opts, _ = getopt.getopt(argv, "c:i:o:vs", ["config=", "input=", "output=", "verbose", "use_chached_docs"])
    except getopt.GetoptError:
        logger.error('main.py -c <config_file> -i <input_file> -o <output_file>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-c", "--config"):
            configuration_file = arg
        if opt in ("-i", "--input"):
            input_file = arg
        if opt in ("-o", "--output"):
            output_file = arg
        if opt in ("-s", "--use_chached_docs"):
            use_cache = True
        if opt in ("-v", "--verbose"):
            logging.getLogger().setLevel(logging.DEBUG)

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

    # Save the unanonymized dataframe for later
    unanonymized_df = df.copy()

    # Parameters for anonymization
    k = config.parameters["k"]
    strategy = config.parameters["strategy"]
    biases = config.get_biases()
    relational_weight = config.get_relational_weight()

    # Anonymize quasi identifier (applying k-anonymity) and recode textual attributes
    anonymized_df, partitions, partition_split_statistics = kernel.anonymize_quasi_identifiers(df, k, strategy, biases, relational_weight)
    anonymized_df = kernel.recode_textual_attributes(anonymized_df)

    # Parameters for calculating metrics
    quasi_identifiers = config.get_quasi_identifiers()
    textual_attribute_mapping = pp.get_textual_attribute_mapping()

    # Calculating the total, relational, and textual information loss based on the original and anonymized data frame
    total_information_loss, relational_information_loss, textual_information_loss = calculate_normalized_certainty_penalty(unanonymized_df, anonymized_df, quasi_identifiers, textual_attribute_mapping)

    # Calculating the mean and std for partition size as well as split statistics
    mean_partition_size = calculate_mean_partition_size(partitions)
    std_partition_size = calculate_std_partition_size(partitions)
    if partition_split_statistics:
        number_of_relational_splits, number_of_textual_splits = get_partition_split_share(partition_split_statistics, textual_attribute_mapping)

    # Notify about the results
    logger.info("Information loss for relational attributes is %4.4f", relational_information_loss)
    if textual_information_loss:
        logger.info("Information loss for textual attribute is %4.4f", textual_information_loss["total"])
    logger.info("Total information loss is %4.4f", total_information_loss)
    logger.info("Ended up with %d partitions with a mean size of %.2f and a std of %.2f", len(partitions), mean_partition_size, std_partition_size)
    if partition_split_statistics:
        logger.info("Split %d times on a relational attribute", number_of_relational_splits)
        logger.info("Split %d times on a textual attribute", number_of_textual_splits)

    # Initialize the postprocessor with the config and the preprocessor
    post_processor = PostProcessor(config, pp)

    # Perform post processing actions on the anonymized data frame
    anonymized_df = post_processor.clean(anonymized_df)
    anonymized_df = post_processor.uncompress(anonymized_df)
    anonymized_df = post_processor.pretty(anonymized_df)

    # Don't forget to drop the direct identifiers since they are now not needed anymore
    anonymized_df = kernel.remove_direct_identifier(anonymized_df)

    # Notify and save
    logger.info("Saving anonymized file to %s", output_file)
    anonymized_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
