import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


def _generate_missing_mask(data, fraction, mechanism, p_obs=0.5):
    """
    Generate a missing value mask for a pandas DataFrame based on the selected missing data mechanism (MCAR, MAR, or MNAR)
    and the percentage of missing values to generate.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data as a pandas DataFrame.
    fraction : float
        Percentage of missing values to generate.
    mechanism : str
        Missing data mechanism to be used. Options: "MCAR", "MAR", "MNAR".
    p_obs : float, optional
        If mechanism = "MAR" or mechanism = "MNAR", proportion of variables with *no* missing values
        that will be used for the logistic masking model. Default is 0.5.

    Returns
    -------
    mask : pandas.DataFrame
        Missing value mask as a pandas DataFrame with the same shape as the input data.

    Raises
    ------
    ValueError
        If the sum of fraction and p_obs is greater than 1.

    Notes
    -----
    This function generates a missing value mask for a pandas DataFrame based on the selected missing data mechanism (MCAR, MAR, or MNAR)
    and the percentage of missing values to generate.

    Author: Adam Michalik
    """

    if fraction + p_obs > 1:
        raise ValueError("The sum of fraction and p_obs cannot be greater than 1.")

    # Compute the number of missing values to generate
    n_missing = int(round(data.size * fraction))
    
    # Generate missing values using the specified mechanism
    if mechanism == 'MCAR':
        # Initialize the missing value mask
        mask = pd.DataFrame(np.zeros_like(data.values), index=data.index, columns=data.columns)
        # Generate random indices for missing values
        missing_indices = np.random.choice(data.size, n_missing, replace=False)
        # Set the corresponding entries in the missing value mask to 1
        mask.values.flat[missing_indices] = 1

    elif mechanism == 'MAR':
        mask = _generate_mar_mask(data, n_missing, p_obs)

    elif mechanism == 'MNAR':
        mask = _generate_mnar_mask(data, n_missing, p_obs)

    
    data_nas = data.mask(mask==1)
    print(data_nas)
    

    return data_nas, mask


def _generate_mar_mask(data, n_missing , p_obs):
    """
    Generate a missing value mask for a pandas DataFrame based on the MAR mechanism and the percentage of missing values
    to generate.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data as a pandas DataFrame.
    percentage : float
        Percentage of missing values to generate.
    p_obs : float, optional
        Proportion of variables with *no* missing values that will be used for the logistic masking model. Default is 0.5.

    Returns
    -------
    mask : pandas.DataFrame
        Missing value mask as a pandas DataFrame with the same shape as the input data.

    Notes
    -----
    This function generates a missing value mask for a pandas DataFrame based on the MAR mechanism and the percentage of
    missing values to generate.

    Author: Adam Michalik
    """

    # Initialize the missing value mask
    mask = pd.DataFrame(np.zeros_like(data.values), index=data.index, columns=data.columns)

    # Compute the number of variables with complete data
    n_observed = int(np.ceil(data.shape[1] * p_obs))
    full_cols = np.random.choice(np.array(data.columns), size=n_observed, replace=False)
    missing_cols = np.array(data.columns)[~np.isin(np.array(data.columns), full_cols)]

    # Fit a logistic regression model to predict missingness in the remaining variables
    probs = pd.DataFrame()
    for col in missing_cols:
        lr = LogisticRegression(solver='lbfgs')
        lr.fit(data[full_cols].sample(n=2), [0, 1])
        # Generate missing values using the logistic masking model
        missing_probs = lr.predict_proba(data[full_cols])
        probs[col] = missing_probs[:, 0]

    # Set the corresponding entries in the missing value mask to 1
    lowest = probs.stack().nsmallest(n_missing).index
    for x, y in lowest:
        mask.loc[x, y] = 1

    return mask


def _generate_mnar_mask(data, n_missing, p_obs):
    """
    Generate a missing value mask for a pandas DataFrame based on the MNAR missing data mechanism and the
    percentage of missing values to generate.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data as a pandas DataFrame.
    fraction : float
        Percentage of missing values to generate.
    p_obs : float, optional
        Proportion of variables with *no* missing values that will be used for the logistic masking model.
        Default is 0.5.

    Returns
    -------
    mask : pandas.DataFrame
        Missing value mask as a pandas DataFrame with the same shape as the input data.

    Notes
    -----
    This function generates a missing value mask for a pandas DataFrame based on the MNAR missing data mechanism
    and the percentage of missing values to generate. It assumes that missingness is dependent on the values of
    other variables in the dataset.

    Author: Adam Michalik
    """

    # Initialize the missing value mask
    mask = pd.DataFrame(np.zeros_like(data.values), index=data.index, columns=data.columns)

    # Compute the number of variables with complete data
    n_observed = int(np.ceil(data.shape[1] * p_obs))
    full_cols = np.random.choice(np.array(data.columns), size=n_observed, replace=False)
    missing_cols = np.array(data.columns)[~np.isin(np.array(data.columns), full_cols)]

    # Fit a logistic regression model to predict missingness in the remaining variables
    probs = pd.DataFrame()
    for col in missing_cols:
        lr = LogisticRegression(solver='lbfgs')
        lr.fit(data[np.append(full_cols, col)].sample(n=2), [0,1])
        # Generate missing values using the logistic masking model
        missing_probs = lr.predict_proba(data[np.append(full_cols, col)])
        probs[col] = missing_probs[:, 0]
        
    # Set the corresponding entries in the missing value mask to 1
    lowest = probs.stack().nsmallest(n_missing).index
    mask = pd.DataFrame(np.zeros_like(data.values), index=data.index, columns=data.columns)
    for x, y in lowest:
        mask.loc[x, y] = 1

    return mask


def generate_missing_datasets(input_data, fraction, mechanism, num_files, p_obs, subsample, num_retries=500):
    """
    Generate multiple datasets with missing values based on the input data.

    Parameters
    ----------
    input_data : pandas.DataFrame
        Input data as a pandas DataFrame.
    fraction : float
        Fraction of missing values to generate.
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

    Author: Adam Michalik
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
        data_incomp, mask = _generate_missing_mask(pd.DataFrame(sample_data), fraction, mechanism, p_obs)
        
        # Save the incomplete data to a pandas DataFrame
        output_data = pd.DataFrame(data_incomp)
        
        # Save the mask to a pandas DataFrame
        mask_data = pd.DataFrame(mask)
        mask_value_data= pd.DataFrame(sample_data).mask(mask_data != 1)

        created_data.append((output_data, mask_value_data))

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

    Author: Adam Michalik
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

    Author: Adam Michalik
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


def _main(input_file, fraction, mechanism, num_files, output_name, has_header, p_obs, subsample):
    """
    Run the data generation pipeline for creating multiple datasets with missing values.
    
    Parameters
    ----------
    input_file : str
        Input CSV file.
    fraction : float
        Fraction of missing values to generate.
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

    Author: Adam Michalik
    """
    
    # Load the input CSV file
    input_data, header = _input(input_file, has_header)

    # Generate missing datasets using `generate_missing_datasets()`
    created_data = generate_missing_datasets(input_data, fraction, mechanism, num_files, p_obs, subsample)

    # Save the generated datasets using `_output()`
    _output(created_data, output_name, has_header, header)


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate datasets with missing values.')
    parser.add_argument('input_file', help='Input CSV file')
    parser.add_argument('-m', '--fraction', type=float, default=10, help='Fraction of missing values')
    parser.add_argument('-t', '--mechanism', choices=['MCAR', 'MAR', 'MNAR'], default='MCAR', help='Missing data mechanism')
    parser.add_argument('-n', '--num_files', type=int, default=1, help='Number of files to generate')
    parser.add_argument('-o', '--output_name', default='output', help='Output name')
    parser.add_argument('-e', '--has_header', action='store_true', help='Input file has a header')
    parser.add_argument('-b', '--p_obs', type=float, default=0.1, help='Proportion of variables with no missing values (required for MAR and MNAR)')
    parser.add_argument('-s', '--subsample', type=float, default=None, help='Ratio of dataset subsampling for each run')
    args = parser.parse_args()

    _main(args.input_file, args.fraction, args.mechanism, args.num_files, args.output_name, args.has_header, args.p_obs, args.subsample)