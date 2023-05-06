from generator import generate_missing_datasets, _input
from knnimputer import impute_missing_data
from nrmse_calculator import evaluate_imputations
from tqdm import tqdm
import pandas as pd
import csv

# Constants
IMPUTATION_FUNCTION = impute_missing_data
INPUT_FILE = 'data\\cars_small.csv'
OUTPUT_FILE = 'output.csv'
HAS_HEADER = True
PERCENTAGES = [10, 20, 30, 40]
MECHANISMS = ['MCAR', 'MNAR']
NUM_TRIES = 10
OPT = 'logistic'
P_OBS = 0.05
Q = 0.3


def evaluation_pipeline(imputation_function, input_file, output_file, has_header, percentages, mechanisms, num_tries, opt, p_obs, q):
    # Load the input CSV file
    input_data, header = _input(input_file, has_header)

    # Evaluate each combination of percentage and mechanism
    results = []
    for mechanism in tqdm(mechanisms, desc='Mechanism'):
        for percentage in tqdm(percentages, desc='Percentage', leave=False):
            # Generate missing datasets using `generate_missing_datasets()`
            created_datas = generate_missing_datasets(input_data, percentage, mechanism, num_tries, opt, p_obs, q)

            # Impute missing data using `impute_missing_data()`
            imputed_datas = [(imputation_function(created_data), mask_data) for created_data, mask_data in created_datas]

            # Evaluate imputed datasets using `evaluate_imputations()`
            evaluation_results = evaluate_imputations(pd.DataFrame(data = input_data,  columns = header), imputed_datas)

            # Store evaluation results in a list
            result = [mechanism, percentage] + list(evaluation_results)
            results.append(result)

    # Write results to CSV file
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['mechanism', 'percentage_missing', 'min_nrmse', 'avg_nrmse', 'max_nrmse', 'nrmse_values'])
        for result in results:
            writer.writerow(result)

def _main():
    evaluation_pipeline(IMPUTATION_FUNCTION, INPUT_FILE, OUTPUT_FILE, HAS_HEADER, PERCENTAGES, MECHANISMS, NUM_TRIES, OPT, P_OBS, Q)

if __name__ == "__main__":
    _main()
