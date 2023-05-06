import argparse
import pandas as pd
import numpy as np
import torch
import seaborn as sns

from utils import *


def _produce_NA(X, p_miss, mecha="MCAR", opt=None, p_obs=None, q=None):
    """
    Generates missing values for a specific missing-data mechanism and proportion of missing values.
    
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a PyTorch tensor.
    p_miss : float
        Proportion of missing values to generate for variables that will have missing values.
    mecha : str, optional
        Indicates the missing-data mechanism to be used. Options: "MCAR" (default), "MAR", "MNAR".
    opt: str, optional
        For mecha = "MNAR", it indicates how the missing-data mechanism is generated:
        - "logistic" for logistic regression
        - "quantile" for quantile censorship
        - "selfmasked" for logistic regression for generating a self-masked MNAR mechanism
    p_obs : float, optional
        If mecha = "MAR" or mecha = "MNAR" with opt = "logistic" or "quanti", proportion of variables with *no* missing values
        that will be used for the logistic masking model.
    q : float, optional
        If mecha = "MNAR" and opt = "quanti", quantile level at which the cuts should occur.
    
    Returns
    -------
    X_nas : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data with the generated missing values.
    mask : torch.DoubleTensor or np.ndarray, shape (n, d)
        Binary mask indicating the missing values.
    """
    # Check if X is a numpy array or a PyTorch tensor, convert to tensor if necessary
    to_torch = torch.is_tensor(X)  
    if not to_torch:
        X = X.astype(np.float32)
        X = torch.from_numpy(X)
    
    # Generate missing values based on the specified missing-data mechanism
    if mecha == "MAR":
        # generate missing values using a masking mechanism that is dependent on non-missing variables
        mask = MAR_mask(X, p_miss, p_obs).double()
    elif mecha == "MNAR" and opt == "logistic":
        # generate missing values using a masking mechanism that is dependent on non-missing variables
        # the masking mechanism is a logistic regression model
        mask = MNAR_mask_logistic(X, p_miss, p_obs).double()
    elif mecha == "MNAR" and opt == "quantile":
        # generate missing values using a masking mechanism that is dependent on non-missing variables
        # the masking mechanism is based on quantiles of the non-missing variables
        mask = MNAR_mask_quantiles(X, p_miss, q, 1 - p_obs).double()
    elif mecha == "MNAR" and opt == "selfmasked":
        # generate missing values using a masking mechanism that is dependent on missing and non-missing variables
        # the masking mechanism is a logistic regression model
        mask = MNAR_self_mask_logistic(X, p_miss).double()
    else:
        # generate missing values using a masking mechanism that is independent of the data
        mask = (torch.rand(X.shape) < p_miss).double()
    
    # Create a copy of X with the generated missing values
    X_nas = X.clone()
    X_nas[mask.bool()] = np.nan
    
    return X_nas, mask


def generate_missing_datasets(input_data, percentage, mechanism, num_files, opt=None, p_obs=None, q=None):
    """
    Generate multiple datasets with missing values based on the input data.

    Parameters
    ----------
    input_data : pandas.DataFrame
        Input data as a pandas DataFrame.
    percentage : float
        Percentage of missing values to generate.
    mechanism : str
        Missing data mechanism to be used. Options: "MCAR" (default), "MAR", "MNAR", or "MNARsmask".
    num_files : int
        Number of files to generate.
    opt: str, optional
        For mecha = "MNAR", it indicates how the missing-data mechanism is generated:
        - "logistic" for logistic regression
        - "quantile" for quantile censorship
        - "selfmasked" for logistic regression for generating a self-masked MNAR mechanism
    p_obs : float, optional
        If mecha = "MAR" or mecha = "MNAR" with opt = "logistic" or "quanti", proportion of variables with *no* missing values
        that will be used for the logistic masking model.
    q : float, optional
        If mecha = "MNAR" and opt = "quanti", quantile level at which the cuts should occur.
        
    Returns
    -------
    created_data : list of tuples
        List containing the generated incomplete datasets and their corresponding missing value masks as pandas DataFrames.

    Notes
    -----
    This function generates multiple datasets with missing values based on the input data. For each dataset, the missing values are generated
    using the specified missing data mechanism and percentage. The generated incomplete datasets and their corresponding missing value masks
    are returned as a list of tuples containing pandas DataFrames.
    """

    created_data = []

    # Create multiple datasets with missing values
    for i in range(num_files):
        # Generate missing values using produce_NA() function
        data_incomp, mask = _produce_NA(input_data, percentage / 100, mecha=mechanism, opt=opt, p_obs=p_obs, q=q)
        
        # Save the incomplete data to a pandas DataFrame
        output_data = pd.DataFrame(data_incomp)
        
        # Save the mask to a pandas DataFrame
        mask_data = pd.DataFrame(mask)

        created_data.append((output_data, mask_data))

    return created_data
        

def _input(input_file, has_header):
    """
    Reads input data from a file and returns it as a NumPy array.
    
    Parameters
    ----------
    input_file : str
        Input file name.
    has_header : bool
        Specifies if the input file has a header.
        
    Returns
    -------
    data : numpy.ndarray
        The input data as a NumPy array.
    header : list or None
        If the input file has a header, returns a list of column names.
        Otherwise, returns None.
    """
    if has_header:
        # If file has header, use pandas to read it
        data = pd.read_csv(input_file)
        header = list(data.columns)
        return data.values, header
    else:
        # If file doesn't have header, use numpy to read it
        header = None
        return np.loadtxt(input_file, delimiter=','), header
    

def _output(created_data, output_name, has_header, header):
    """
    Save the generated data and mask files to CSV.

    Parameters
    ----------
    created_data : list
        List of tuples containing the generated incomplete data and the corresponding mask.
    output_name : str
        Prefix for the output file name.
    has_header : bool
        Specifies if the input file has a header.
    header : list or None
        Header of the input file, if it exists.

    Returns
    -------
    None
    """

    # Loop over the generated data and mask tuples
    for i, (output_data, mask_data) in enumerate(created_data):
        # Generate a new file name
        if len(created_data) > 1:
            output_file = f"{output_name}_{i+1}.csv"
            mask_file = f"{output_name}_{i+1}.mask"
        else:
            output_file = f"{output_name}.csv"
            mask_file = f"{output_name}.mask"

        # Set the header of the mask data, if it exists
        if has_header:
            mask_data.columns = header

        # Save the mask data to CSV
        mask_data.to_csv(mask_file, index=False, header=has_header)

        # Set the header of the output data, if it exists
        if has_header:
            output_data.columns = header

        # Save the output data to CSV
        output_data.to_csv(output_file, index=False, header=has_header)

        # Print out the generated file and mask file names
        print(f"Generated file: {output_file}")
        print(f"Generated mask: {mask_file}")


def _main(input_file, percentage, mechanism, num_files, output_name, has_header, opt=None, p_obs=0.5, q=0.5):
    """
    Run the data generation pipeline for creating multiple datasets with missing values.
    
    Parameters
    ----------
    input_file : str
        Input CSV file.
    percentage : float
        Percentage of missing values to generate.
    mechanism : str
        Missing data mechanism to be used. Options: "MCAR" (default), "MAR", "MNAR", or "MNARsmask".
    num_files : int
        Number of files to generate.
    output_name : str
        Output file name prefix.
    has_header : bool
        Specifies if the input file has a header.
    opt: str, optional
        For mecha = "MNAR", it indicates how the missing-data mechanism is generated:
        - "logistic" for logistic regression
        - "quantile" for quantile censorship
        - "selfmasked" for logistic regression for generating a self-masked MNAR mechanism
    p_obs : float, optional
        If mecha = "MAR" or mecha = "MNAR" with opt = "logistic" or "quanti", proportion of variables with *no* missing values
        that will be used for the logistic masking model.
    q : float, optional
        If mecha = "MNAR" and opt = "quanti", quantile level at which the cuts should occur.
    
    Returns
    -------
    None
    
    Calls `generate_missing_datasets()` to create multiple datasets with missing values based on the input file, and
    then calls `_output()` to save the generated datasets to CSV files.
    """
    
    # Load the input CSV file
    input_data, header = _input(input_file, has_header)

    # Generate missing datasets using `generate_missing_datasets()`
    created_data = generate_missing_datasets(input_data, percentage, mechanism, num_files, opt, p_obs, q)

    # Save the generated datasets using `_output()`
    _output(created_data, output_name, has_header, header)


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate datasets with missing values.')
    parser.add_argument('input_file', help='Input CSV file')
    parser.add_argument('-p', '--percentage', type=float, default=10, help='Percentage of missing values')
    parser.add_argument('-t', '--mechanism', choices=['MCAR', 'MAR', 'MNAR'], default='MCAR', help='Missing data mechanism')
    parser.add_argument('-f', '--num_files', type=int, default=1, help='Number of files to generate')
    parser.add_argument('-o', '--output_name', default='output', help='Output name')
    parser.add_argument('-e', '--has_header', action='store_true', help='Input file has a header')
    parser.add_argument('-x', '--opt', default=None, choices=['logistic', 'quantile', 'selfmasked'], help='Missing-data mechanism option (required for MNAR)')
    parser.add_argument('-b', '--p_obs', type=float, default=None, help='Proportion of variables with no missing values (required for MAR and MNAR with logistic or quantile option)')
    parser.add_argument('-q', '--quantile', type=float, default=None, help='Quantile level for MNAR with quantile option')
    args = parser.parse_args()

    _main(args.input_file, args.percentage, args.mechanism, args.num_files, args.output_name, args.has_header, args.opt, args.p_obs, args.quantile)