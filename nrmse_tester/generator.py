import argparse
import pandas as pd
import numpy as np
import torch

from mask_creator import MAR_mask, MNAR_mask_logistic


def _produce_NA(X, p_miss, mecha, p_obs):
    """
    Generates missing values for a specific missing-data mechanism and proportion of missing values.
    
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a PyTorch tensor.
    p_miss : float
        Proportion of missing values to generate for variables that will have missing values.
    mecha : str
        Indicates the missing-data mechanism to be used. Options: "MCAR", "MAR", "MNAR".
    p_obs : float
        If mecha = "MAR" or mecha = "MNAR", proportion of variables with *no* missing values
        that will be used for the logistic masking model.
    
    Returns
    -------
    X_nas : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data with the generated missing values.
    mask : torch.DoubleTensor or np.ndarray, shape (n, d)
        Binary mask indicating the missing values.

    Authors: Aude Sportisse with the help of Marine Le Morvan and Boris Muzellec
    Online https://rmisstastic.netlify.app/how-to/python/generate_html/how%20to%20generate%20missing%20values

    Edited by Adam Michalik
    """
    # Check if X is a numpy array or a PyTorch tensor, convert to tensor if necessary
    to_torch = torch.is_tensor(X)  
    if not to_torch:
        X = X.astype(np.float32)
        X = torch.from_numpy(X)
    
    # Generate missing values based on the specified missing-data mechanism
    if mecha == "MAR":
        # generate missing values using a masking mechanism that is dependent on non-missing variables
        mask = MAR_mask(X, p_miss, p_obs).float()
    elif mecha == "MNAR":
        # generate missing values using a masking mechanism that is dependent on non-missing variables
        # the masking mechanism is a logistic regression model
        mask = MNAR_mask_logistic(X, p_miss, p_obs).float()
    else:
        # generate missing values using a masking mechanism that is independent of the data
        mask = (torch.rand(X.shape) < p_miss).float()
    
    # Create a copy of X with the generated missing values
    X_nas = X.clone()
    X_nas[mask.bool()] = np.nan
    
    return X_nas, mask


def generate_missing_datasets(input_data, percentage, mechanism, num_files, p_obs, subsample, num_retries=100):
    """
    Generate multiple datasets with missing values based on the input data.

    Parameters
    ----------
    input_data : pandas.DataFrame
        Input data as a pandas DataFrame.
    percentage : float
        Percentage of missing values to generate.
    mechanism : str
        Missing data mechanism to be used. Options: "MCAR", "MAR", "MNAR".
    num_files : int
        Number of files to generate.
    p_obs : float
        If mecha = "MAR" or mecha = "MNAR", proportion of variables with *no* missing values
        that will be used for the logistic masking model.
    subsample : float or None, optional
        Fraction of rows to randomly subsample from the input data. If None, no subsampling is performed. Default is None.
    num_retries : int, optional
        Number of retries to attempt if producing missing data fails. Default is 100.
        
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
        if subsample is not None:
            # Randomly subsample rows
            sample_data = input_data[np.random.choice(input_data.shape[0], int(subsample*input_data.shape[0]), replace=False), :]
        else:
            sample_data = input_data

        # Generate missing values using produce_NA() function
        for i in range(num_retries):
                try:
                    data_incomp, mask = _produce_NA(sample_data, percentage / 100, mecha=mechanism, p_obs=p_obs)
                except ValueError as e:
                    if i == num_retries - 1:
                        raise
                    print(f"WARNING: {e}. Retrying to create missing data")
        
        # Save the incomplete data to a pandas DataFrame
        output_data = pd.DataFrame(data_incomp)
        
        # Save the mask to a pandas DataFrame
        mask_data = pd.DataFrame(mask)
        mask_value_data= pd.DataFrame(sample_data).mask(mask_data != 1)

        created_data.append((output_data, mask_value_data))
        print(mask_value_data)

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


def _main(input_file, percentage, mechanism, num_files, output_name, has_header, p_obs, subsample):
    """
    Run the data generation pipeline for creating multiple datasets with missing values.
    
    Parameters
    ----------
    input_file : str
        Input CSV file.
    percentage : float
        Percentage of missing values to generate.
    mechanism : str
        Missing data mechanism to be used. Options: "MCAR", "MAR", "MNAR".
    num_files : int
        Number of files to generate.
    output_name : str
        Output file name prefix.
    has_header : bool
        Specifies if the input file has a header.
    p_obs : float
        If mecha = "MAR" or mecha = "MNAR", proportion of variables with *no* missing values
        that will be used for the logistic masking model.
    
    Returns
    -------
    None
    
    Calls `generate_missing_datasets()` to create multiple datasets with missing values based on the input file, and
    then calls `_output()` to save the generated datasets to CSV files.
    """
    
    # Load the input CSV file
    input_data, header = _input(input_file, has_header)

    # Generate missing datasets using `generate_missing_datasets()`
    created_data = generate_missing_datasets(input_data, percentage, mechanism, num_files, p_obs, subsample)

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
    parser.add_argument('-b', '--p_obs', type=float, default=0.1, help='Proportion of variables with no missing values (required for MAR and MNAR)')
    parser.add_argument('-s', '--subsample', type=float, default=None, help='Ratio of dataset subsampling for each run')
    args = parser.parse_args()

    _main(args.input_file, args.percentage, args.mechanism, args.num_files, args.output_name, args.has_header, args.p_obs, args.subsample)