import argparse
import os
import pandas as pd
import numpy as np


def _calculate_nrmse(original_values, imputed_values):
    """
    Calculate the Normalized Root Mean Squared Error (NRMSE) between the original and imputed datasets.

    Parameters
    ----------
    original_values : numpy.ndarray
        Original dataset values as a NumPy array.
    imputed_values : numpy.ndarray
        Imputed dataset values as a NumPy array.

    Returns
    -------
    nrmse : float
        The calculated NRMSE value.

    Author: Adam Michalik
    """
    # Calculate the difference between original and imputed datasets
    diff = original_values - imputed_values

    # Calculate the root mean squared error (RMSE) between original and imputed datasets
    rmse = np.sqrt(np.mean(diff ** 2))

    # Calculate the normalized RMSE (NRMSE) by dividing RMSE by avg value
    nrmse = rmse / np.mean(original_values)

    return nrmse


def evaluate_imputations(imputed_datas, proportion):
    """
    Evaluate the imputations by calculating the NRMSE for each imputed file.

    Parameters:
    -----------
    imputed_datas : list of tuples
        A list of tuples containing imputed data and mask data.
    proportion : float
        Proportion of data to be used in statistics

    Returns:
    --------
    avg_nrmse : float or None
        The average NRMSE value.
    min_nrmse : float or None
        The minimum NRMSE value.
    max_nrmse : float or None
        The maximum NRMSE value.
    data_statistics:list
        A list of NRMSE values for top n imputed files.
    nrmse_values : list
        A list of NRMSE values for each imputed file.

    Author: Adam Michalik
    """

    nrmse_values = []

    # Calculate NRMSE for each imputed file
    for imputed_data, mask_data in imputed_datas:
        original_values = mask_data.values[~np.isnan(mask_data.values)]
        imputed_values = imputed_data.values[~np.isnan(mask_data.values)]
        nrmse = _calculate_nrmse(original_values, imputed_values)
        nrmse_values.append(nrmse)
        nrmse_values.sort()

    data_statistics = nrmse_values[0:int(len(nrmse_values)*proportion)]
    # Calculate NRMSE statistics
    if nrmse_values:
        avg_nrmse = np.mean(data_statistics)
        min_nrmse = np.min(data_statistics)
        max_nrmse = np.max(data_statistics)
        return avg_nrmse, min_nrmse, max_nrmse, data_statistics, nrmse_values

    return None, None, None, [], []


def _input(imputed_files, has_header):
    """
    Load the original dataset and imputed files and their corresponding mask files. 

    Parameters
    ----------
    imputed_files : List
        of file paths to the imputed datasets.
    has_header : bool
        Specifies if the input files have headers.

    Returns
    -------
    tuple
        A tuple of two elements containing the original data as a pandas dataframe and a list of tuples,
        where each tuple contains the mask data and imputed data as pandas dataframes.

    Author: Adam Michalik
    """

    # Create an empty list to hold the imputed data and their corresponding masks
    imputed_datas = []

    # Evaluate each imputed file
    for imputed_file in imputed_files:
        # Get the corresponding mask file
        mask_file = imputed_file.replace(".csv", ".mask")

        # Check if both imputed and mask files exist
        if not os.path.isfile(imputed_file) or not os.path.isfile(mask_file):
            print(f"Missing imputed file or mask for {imputed_file}. Skipping...")
            continue

        # Load the mask data
        mask_data = pd.read_csv(mask_file, header=0 if has_header else None)

        # Load the imputed dataset
        imputed_data = pd.read_csv(imputed_file, header=0 if has_header else None)

        # Add the imputed data and mask to the list
        imputed_datas.append((imputed_data, mask_data))

    # Return the original data and the list of imputed data and their corresponding masks
    return imputed_datas


def _output(output_file, avg_nrmse, min_nrmse, max_nrmse, nrmse_values):
    """
    Generates the output string with NRMSE statistics and values and either prints it to the console or saves it to a file.

    Args:
        output_file (str or None): Path to the output file to save the results to. If None, the results will be printed to the console.
        avg_nrmse (float or None): Average NRMSE of the evaluated imputed files. If no valid files are found, None will be returned.
        min_nrmse (float or None): Minimum NRMSE of the evaluated imputed files. If no valid files are found, None will be returned.
        max_nrmse (float or None): Maximum NRMSE of the evaluated imputed files. If no valid files are found, None will be returned.
        nrmse_values (list): List of all NRMSE values of the evaluated imputed files. If no valid files are found, an empty list will be returned.

    Returns:
        None

    Author: Adam Michalik
    """

    if nrmse_values:
        # Format the NRMSE values
        formatted_values = ', '.join([f"{val:.6f}" for val in nrmse_values])

        # Generate the output string
        output_string = f"NRMSE\n"
        output_string += f"Min: {min_nrmse:.6f}\n"
        output_string += f"Max: {max_nrmse:.6f}\n"
        output_string += f"Avg: {avg_nrmse:.6f}\n\n"
        output_string += f"Values: {formatted_values}"

        # Print or save the output
        if output_file is None:
            print(output_string)
        else:
            with open(output_file, 'w') as file:
                file.write(output_string)
    else:
        print("No valid imputed files found for evaluation.")


def _main(imputed_files, has_header=False, output_file=None):
    """
    Main function that runs the imputation evaluation on the given imputed files.

    Args:
        imputed_files (list): List of paths to the imputed datasets.
        has_header (bool, optional): Indicates whether the input files have headers. Default is False.
        output_file (str, optional): Path to the output file for NRMSE summary. Default is None.

    Returns:
        None
    
    Author: Adam Michalik
    """

    imputed_datas = _input(imputed_files, has_header)

    avg_nrmse, min_nrmse, max_nrmse, nrmse_values = evaluate_imputations(imputed_datas)

    _output(output_file, avg_nrmse, min_nrmse, max_nrmse, nrmse_values)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate multiple imputed datasets.")
    parser.add_argument("imputed_files", nargs="+", help="Imputed datasets")
    parser.add_argument("-o", "--output", help="Output file for NRMSE summary")
    parser.add_argument("-e", "--has_header", action="store_true", help="Input files have headers")
    args = parser.parse_args()

    # Evaluate imputations
    _main(args.imputed_files, args.has_header, args.output)
