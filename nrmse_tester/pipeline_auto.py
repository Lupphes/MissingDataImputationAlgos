from generator import generate_missing_datasets, _input
from knnimputer import impute_missing_data
from nrmse_calculator import evaluate_imputations
from tqdm import tqdm
import csv


# Constants
IMPUTATION_FUNCTION = impute_missing_data
INPUT_FILE = 'data\\cars.csv'
OUTPUT_FILE = 'output.csv'
HAS_HEADER = True
FRACTIONS = [0.05, 0.1, 0.2, 0.3, 0.4]
MECHANISMS = ['MCAR', 'MAR', 'MNAR']
NUM_TRIES = 100
P_OBS = 0.3
SUBSAMPLE = 0.05



def evaluation_pipeline(imputation_function, input_file, output_file, has_header, fractions, mechanisms, num_tries, p_obs, subsample):
    # Load the input CSV file
    input_data, header = _input(input_file, has_header)

    # Evaluate each combination of percentage and mechanism
    results = []
    for mechanism in tqdm(mechanisms, desc='Mechanism'):
        for percentage in tqdm(fractions, desc='Percentage', leave=False):
            # Generate missing datasets using `generate_missing_datasets()`
            created_datas = generate_missing_datasets(input_data, percentage, mechanism, num_tries, p_obs, subsample)

            # Impute missing data using `impute_missing_data()`
            imputed_datas = [(imputation_function(created_data), mask_data) for created_data, mask_data in created_datas]

            # Evaluate imputed datasets using `evaluate_imputations()`
            evaluation_results = evaluate_imputations(imputed_datas)

            # Store evaluation results in a list
            result = [mechanism, percentage] + list(evaluation_results)
            results.append(result)

    # Write results to CSV file
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['mechanism', 'percentage_missing', 'avg_nrmse', 'min_nrmse', 'max_nrmse', 'nrmse_values'])
        for result in results:
            writer.writerow(result)

def _main():
    evaluation_pipeline(IMPUTATION_FUNCTION, INPUT_FILE, OUTPUT_FILE, HAS_HEADER, FRACTIONS, MECHANISMS, NUM_TRIES, P_OBS, SUBSAMPLE)

if __name__ == "__main__":
    _main()
