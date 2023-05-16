from generator import generate_missing_datasets, _input
from knnimputer import impute_missing_data
from nrmse_calculator import evaluate_imputations
from tqdm import tqdm
from iviaclr import iviaclr
from multiprocessing import Pool

import csv


# Constants
IMPUTATION_FUNCTION = impute_missing_data
INPUT_FILE = '../dataset/cleaned_csv_file_500.csv'
OUTPUT_FILE = 'output.csv'
HAS_HEADER = True
FRACTIONS = [0.05, 0.1, 0.2, 0.3]
MECHANISMS = ['MCAR', 'MAR', 'MNAR']
NUM_TRIES = 2000
P_OBS = 0.3
SUBSAMPLE = 0.3
PROPORTION = 0.5


def evaluation_pipeline(imputation_function, input_file, output_file, has_header, fractions, mechanisms, num_tries, p_obs, proportion, subsample, processes=12):
    # Load the input CSV file
    input_data, header = _input(input_file, has_header)

    combinations = [(imputation_function, input_data, fraction, mechanism, num_tries, p_obs, proportion, subsample)
                    for fraction in fractions for mechanism in mechanisms]
    pool = Pool(processes=processes)

    # Evaluate each combination of percentage and mechanism\
    results = list(tqdm(pool.imap(single_evaluation, combinations), total=len(combinations), desc='Progress'))

    # Write results to CSV file
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['mechanism', 'percentage_missing', 'avg_nrmse', 'min_nrmse', 'max_nrmse', 'used_nrmse_values', 'nrmse_values'])
        for result in results:
            writer.writerow(result)


def single_evaluation(combination):
    imputation_function, input_data, fraction, mechanism, num_tries, p_obs, proportion, subsample = combination

    # Generate missing datasets using `generate_missing_datasets()`
    created_datas = generate_missing_datasets(input_data, fraction, mechanism, num_tries, p_obs, subsample)

    # Impute missing data using `impute_missing_data()`
    imputed_datas = [(imputation_function(created_data), mask_data) for created_data, mask_data in created_datas]

    # Evaluate imputed datasets using `evaluate_imputations()`
    evaluation_results = evaluate_imputations(imputed_datas, proportion)

    # Store evaluation results in a list
    result = [mechanism, fraction] + list(evaluation_results)

    return result


def _main():
    evaluation_pipeline(IMPUTATION_FUNCTION, INPUT_FILE, OUTPUT_FILE, HAS_HEADER, FRACTIONS, MECHANISMS, NUM_TRIES, P_OBS, PROPORTION, SUBSAMPLE)


if __name__ == "__main__":
    _main()
