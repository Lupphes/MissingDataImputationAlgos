import subprocess
import pandas as pd
import tempfile
import os
import uuid


def run_fortran_app(path_to_fortran_executable, params, cwd):
    command = [path_to_fortran_executable] + params
    result = subprocess.run(command, stdout=subprocess.PIPE, cwd=cwd)
    #print('Return code:', result.returncode)
    #print('Output:', result.stdout.decode('utf-8'))


def read_fortran_output(output_file, nft):
    with open(output_file, 'r') as f:
        lines = f.read().splitlines()

    # Join all lines together
    all_data = ' '.join(lines)

    # Split the joined data into segments
    segments = all_data.split()

    # Ensure the number of segments is divisible by nft
    assert len(segments) % nft == 0, "Number of segments is not divisible by nft"

    # Split segments into records
    records = [segments[i:i + nft] for i in range(0, len(segments), nft)]

    # Convert records to a DataFrame
    df = pd.DataFrame(records, dtype=float)

    return df


def iviaclr(
        data_df: pd.DataFrame,
        path_to_fortran: str = "../../../clustering/IviaCLR/iviaclr",
        output: str = None,
        output_csv: str = None,
        index_column_name: str = None,
        maxclust: int = 3,
        init_imp: int = 0,
        predi: int = 1,
        misvalue: int = -9999
):
    if index_column_name:
        # Remove ID column
        data_df = data_df.drop(columns=[index_column_name])

    nrecord = data_df.shape[0]  # number of records in the data set.
    nft = data_df.shape[1]  # number of features in the data set.
    maxclust = maxclust  # maximum number of linear regression functions.
    init_imp = init_imp  # optional, initial imputation method, mean by default.
    # Possible values:
    # 0 - mean imputation (default),
    # 1 - regression on complete objects,
    # 2 - recursive regression.
    predi = predi  # optional, prediction method.
    # Possible values:
    # 0 - k-NN with popularity based weights on clusters.
    # 1 - k-NN with rmsq based weights on clusters (default).
    # 2 - the weight based on popularity of clusters (don't use).
    misvalue = misvalue  # optional, value for missing value, must be a number, -9999 by default.

    # Replace NaNs with misvalue
    data_df.fillna(misvalue, inplace=True)

    # Create a temporary file
    tmpfile = tempfile.NamedTemporaryFile(delete=False)

    # Write the DataFrame to the temporary file
    data_df.to_csv(tmpfile.name, sep=' ', header=False, index=False)

    cwd = "iviaCLR/" + uuid.uuid4().hex + "/"

    if not os.path.exists("iviaCLR/"):
        os.makedirs("iviaCLR/")

    os.mkdir(cwd)

    if not output:
        output = "result.txt"
    parameters = [tmpfile.name, output, str(nrecord), str(nft), str(maxclust), str(init_imp), str(predi), str(misvalue)]
    run_fortran_app(path_to_fortran, parameters, cwd)

    # Delete the temporary file
    os.unlink(tmpfile.name)

    # Read output file into a DataFrame
    output_df = read_fortran_output(cwd + output, nft)

    if output_csv:
        # Save DataFrame to CSV
        output_df.to_csv(output_csv, index=False)

    return output_df


if __name__ == "__main__":
    PATH_TO_FORTRAN = "../../../clustering/IviaCLR/iviaclr"
    CARS_DATASET_25 = "dataset/cleaned_csv_file_25_missing.csv"

    sample_data_df = pd.read_csv(CARS_DATASET_25)

    sample_output_df = iviaclr(
        sample_data_df,
        path_to_fortran=PATH_TO_FORTRAN,
    )

    print(f"Final result:\n{sample_output_df}")
