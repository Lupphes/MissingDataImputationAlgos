import argparse
import pandas as pd
from sklearn.impute import KNNImputer


def impute_missing_data(data):
    """
    Imputes missing data using the KNNImputer algorithm.

    Args:
        data (pandas.DataFrame): The input dataset containing missing values.

    Returns:
        pandas.DataFrame: The dataset with missing values replaced with imputed values.
    
    Author: Adam Michalik
    """
    # Create a KNNImputer object
    imputer = KNNImputer()

    # Impute missing values
    imputed_data = imputer.fit_transform(data)
    if imputed_data.shape[1] == 11:
        print(imputed_data)
        print(data)
    # Create a DataFrame from the imputed data
    return pd.DataFrame(imputed_data, columns=data.columns)


def _main(input_file, output_file):
    """
    Reads in a CSV file, imputes any missing values using the KNNImputer algorithm,
    and saves the imputed dataset to a new CSV file.

    Args:
        input_file (str): The path to the input CSV file.
        output_file (str): The path to the output CSV file.

    Returns:
        None

    Author: Adam Michalik
    """
    # Load the CSV file
    data = pd.read_csv(input_file)

    # Impute missing data into original data
    imputed_data = impute_missing_data(data)

    # Save the imputed data to CSV
    imputed_data.to_csv(output_file, index=False)
    print(f"Imputed data saved to {output_file}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Impute missing data using KNNImputer")
    parser.add_argument("input_file", help="Input CSV file")
    parser.add_argument("output_file", help="Output CSV file for imputed data")
    args = parser.parse_args()

    _main(args.input_file, args.output_file)