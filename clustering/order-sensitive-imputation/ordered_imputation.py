import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MinMaxScaler
from operator import itemgetter
from typing import Dict, List, Tuple, Union

def convert_similarity_matrix_to_csv(
    similarity_matrix: pd.DataFrame,
    dataset_similarity_file_path: str,
    dataset_with_data: pd.DataFrame,
    index_column_name: str,
) -> pd.DataFrame:
    """
    This function converts a similarity matrix to a DataFrame and saves it as a CSV file.
    The first column is set to zero.

    Parameters:
    similarity_matrix (DataFrame): The similarity matrix to be converted.
    dataset_similarity_file_path (str): The path where the CSV file should be saved.
    dataset_with_data (DataFrame): Data with missing values
    index_column_name (str): Columns of ID which will be removed

    Returns:
    A pandas DataFrame.
    """

    dataset_without_ID = dataset_with_data.drop(columns=[index_column_name])

    nan_rows = dataset_without_ID[dataset_without_ID.isna().any(axis=1)].index

    for column in similarity_matrix.columns:
        if column not in nan_rows:
            similarity_matrix[column] = 0

    # Rename the columns of the new dataframe
    similarity_matrix.rename(
        columns={i: f"x{i+1}" for i in range(similarity_matrix.shape[1])}, inplace=True
    )

    # Reindex to start from 1
    similarity_matrix.index = similarity_matrix.index + 1

    # Insert ID column
    similarity_matrix.insert(
        0, index_column_name, similarity_matrix.index.map(lambda x: f"x{x}")
    )
    if dataset_similarity_file_path:
        # Save the similarity DataFrame as a CSV file
        similarity_matrix.to_csv(dataset_similarity_file_path, index=False)

    return similarity_matrix




def apply_binary_threshold_to_similarity_matrix(
    similarity_matrix: np.ndarray, threshold: float
) -> List[List[int]]:
    """
    This function applies a binary threshold to a similarity matrix.
    Values equal to 1 are replaced with 0, values above the threshold are replaced with 1,
    and all other values are replaced with 0.

    Parameters:
    similarity_matrix (numpy array): The similarity matrix to apply the threshold to.
    threshold (float): The threshold value.

    Returns:
    thresholded_similarity_matrix (list): The thresholded similarity matrix.
    """

    thresholded_similarity_matrix = [
        [0 if value == 1 else (1 if value >= threshold else 0) for value in row]
        for row in similarity_matrix
    ]

    return thresholded_similarity_matrix


def apply_threshold_to_similarity_matrix(
    similarity_matrix: np.ndarray, threshold: float
) -> List[List[Union[int, float]]]:
    """
    This function applies a threshold to a similarity matrix. Values equal to 1 are replaced with 0,
    values above the threshold are kept as they are, and all other values are replaced with 0.

    Parameters:
    similarity_matrix (numpy array): The similarity matrix to apply the threshold to.
    threshold (float): The threshold value.

    Returns:
    thresholded_similarity_matrix (list): The thresholded similarity matrix.
    """
    thresholded_similarity_matrix = [
        [k if k >= threshold else 0 for k in k] for k in similarity_matrix
    ]

    # Replace 1s with 0s in the thresholded similarity matrix.
    thresholded_similarity_matrix = [
        [0 if value == 1 else value for value in row]
        for row in thresholded_similarity_matrix
    ]

    return thresholded_similarity_matrix


def create_similarity_score_dataframe(
    thresholded_similarity_matrix: List[List[Union[int, float]]]
) -> pd.DataFrame:
    """
    Creates a new dataframe with similarity scores, renames the columns,
    and adds a new column 'ID2' with index values.

    Args:
    thresholded_similarity_matrix: A 2D array-like object with similarity scores.

    Returns:
    A pandas DataFrame.
    """
    # Create a new dataframe with the similarity scores
    similarity_score_df = pd.DataFrame(thresholded_similarity_matrix)

    # Rename the columns
    column_mapping = {i: f"X{i+1}" for i in range(similarity_score_df.shape[1])}
    similarity_score_df.rename(columns=column_mapping, inplace=True)

    # Adjust the index to start from 1 instead of 0
    similarity_score_df.index += 1

    # Create a new column 'ID2' with index values
    similarity_score_df["ID2"] = similarity_score_df.index.to_series().apply(
        lambda x: f"X{x}"
    )

    return similarity_score_df

def create_incomplete_connection_dicts(complete_score_list: pd.DataFrame, incomplete_pairs_dict: Dict[str, List[str]]) -> Tuple[Dict[str, List[str]], Dict[str, List[float]]]:
    """
    This function creates two dictionaries.
    The first one, incomplete_node_connections_dict, maps each incomplete node to a list of nodes that it is connected to.
    The second one, incomplete_node_weights_dict, maps each incomplete node to a list of its connection weights.
    """
    # Initialize dictionaries to hold connection nodes and weights
    incomplete_node_connections_dict = {}
    incomplete_node_weights_dict = {}

    # Iterate over each column in the complete list and the incomplete tuples dictionary
    for column in complete_score_list.columns:
        if column in incomplete_pairs_dict:
            # Filter the complete_score_list for non-zero values in the current column
            non_zero_mask = complete_score_list[column] != 0.0
            connected_nodes = complete_score_list.index[non_zero_mask].tolist()
            connection_weights = complete_score_list.loc[non_zero_mask, column].tolist()

            # Map the incomplete node to its connection nodes and weights
            incomplete_node_connections_dict[column] = connected_nodes
            incomplete_node_weights_dict[column] = connection_weights

    return incomplete_node_connections_dict, incomplete_node_weights_dict

def calculate_p_value(
    incomplete_pairs: List[Tuple[str, str]],
    node_data_dict: Dict[str, List[Union[int, float]]],
) -> int:
    """
    This function calculates the p value for the final calculation of node weights.

    Parameters:
    incomplete_pairs (list): List of tuples which are considered 'incomplete'
    node_data_dict (dict): Dictionary where keys are elements from 'incomplete_pairs' and values are lists of associated values

    Returns:
    p_value (int): Calculated p value
    """

    term = 1 / len(incomplete_pairs) if incomplete_pairs else 1
    total_values = sum(len(values) for values in node_data_dict.values())

    p_value = math.ceil(term * total_values)

    return p_value


def get_associated_values_length(
    incomplete_tuples: List[Tuple], data_dict: Dict
) -> Dict:
    """
    Gets the length of associated values from a data dictionary for each key in a list of incomplete tuples.

    Args:
            incomplete_tuples: A list of tuples considered 'incomplete'.
            data_dict: A dictionary with keys as elements from 'incomplete_tuples' and values as lists of associated values.

    Returns:
            A dictionary where keys are elements from 'incomplete_tuples' and values are the length of associated values in 'data_dict'.
    """

    return {key: len(data_dict.get(key, [])) for key in incomplete_tuples}


def calculate_node_weights(
    similarity_dict: Dict,
    length_dict: Dict,
    tuple_list: List[Tuple],
    p_value: np.float64,
) -> Dict:
    """
    Calculates node weights based on the provided similarity and length dictionaries.

    Args:
            similarity_dict: A dictionary containing similarity scores.
            length_dict: A dictionary containing lengths of each item.
            tuple_list: A list of tuples to be processed.
            p_value: A constant value used in the calculation.

    Returns:
            A dictionary with calculated node weights.
    """

    node_weights = {}

    for tuple in tuple_list:
        length = length_dict[tuple]

        if length > 0:
            initial_calculations = [
                (1 - similarity_dict[tuple][i]) ** (1 / p_value) for i in range(length)
            ]
            weight = 1 - np.prod(initial_calculations)
        else:
            weight = 0.0

        node_weights[tuple] = np.round(weight, 3)

    return node_weights


def process_node_weights(
    incomplete_tuples: List[Tuple], incomplete_values_dict: Dict, scores_dict: Dict
) -> Tuple[Dict, Dict, List[str], List[str]]:
    """
    Processes node weights based on the provided incomplete tuples, incomplete data dictionary, and scores dictionary.

    Args:
            incomplete_tuples: A list of tuples considered 'incomplete'.
            incomplete_values_dict: A dictionary with incomplete data.
            scores_dict: A dictionary with scores.

    Returns:
            A tuple containing dictionaries with final scores and lengths, and lists with score names and removed items.
    """

    # print("Incoming incomplete tuples:\n", incomplete_tuples)
    # print("Incoming incomplete data dict:\n", incomplete_values_dict)
    # print("Incoming scores dict:\n", scores_dict)
    data_dict_2 = get_associated_values_length(
        incomplete_tuples, incomplete_values_dict
    )
    # print("Length of incomplete neighbours:\n", data_dict_2)

    # print("Node Weights:", scores_dict)
    score_list = []
    score_list_name = []
    removed_item = []
    for element in incomplete_tuples:
        # print("Processing element:", element)
        # print("data_dict_2 value:", data_dict_2[element])
        # print("dict_scores value:", scores_dict[element])
        if data_dict_2[element] > 0:
            score_list_name.append(element)
            score_list.append(scores_dict[element])
            # print("score_list_name after append:", score_list_name)
            # print("score_list after append:", score_list)
        else:
            # print("Removed")
            removed_item.append(element)
    zip_score_final = zip(score_list_name, score_list)
    data_final_scores = dict(zip_score_final)
    # print("removed item", removed_item)
    # print("Node Weights input to Greedy Algorithm:\n", data_final_scores)

    return data_final_scores, data_dict_2, score_list_name, removed_item


def create_incomplete_dictionaries(
    columns_dict: Dict[str, List[Union[int, float]]], data_dict: Dict
) -> Tuple[Dict, Dict]:
    """
    Creates two dictionaries based on the input dictionary of column names and data dictionary.

    Args:
            columns_dict: Dictionary with column names as keys and lists of column values as values.
            data_dict: Dictionary with column names as keys.

    Returns:
            Two dictionaries based on the input dictionary and data dictionary.
    """

    listed = []
    listed1 = []

    # Iterating over each column in the dictionary
    for column, values in columns_dict.items():

        # Creating lists for non-zero indexed values and column values
        indexed_values = []
        column_values = []

        # Going through each value in the column
        for value in values:

            # If the value is non-zero, add it to both lists
            if value != 0.0:
                indexed_values.append(value)
                column_values.append(value[column])

        # Appending the indexed_values and column_values lists to the main lists
        listed.append(indexed_values)
        listed1.append(column_values)

    # Creating dictionaries from the incomplete_tuples and the main lists
    dict_da = dict(zip(data_dict, listed))
    dict_db = dict(zip(data_dict, listed1))

    return dict_da, dict_db


def get_highest_score_node(
    incomplete_tuples_score: Dict, complete_tuples: List[Tuple]
) -> Tuple:
    """
    Gets the node with the highest score that is not already in the list of complete tuples.

    Args:
            incomplete_tuples_score: A dictionary with scores for incomplete tuples.
            complete_tuples: A list of tuples considered 'complete'.

    Returns:
            The node (tuple) with the highest score not already in the list of complete tuples.
    """

    # print(f"Incomplete Tuples Score: {incomplete_tuples_score}")
    # print(f"Complete Tuples: {complete_tuples}")

    # Exclude the nodes that are already in the complete_tuples list
    incomplete_tuples_score = {
        k: v for k, v in incomplete_tuples_score.items() if k not in complete_tuples
    }
    return max(incomplete_tuples_score, key=incomplete_tuples_score.get)


def calculate_gain(
    node: Tuple,
    lengths_dict: Dict,
    incomplete_data_dict: Dict,
    incomplete_tuples_score: Dict,
    p_value: np.float64,
    data_dict_inc_1: Dict,
) -> np.float64:
    """
    Calculates the gain for a given node.

    Args:
            node: A tuple representing a node.
            lengths_dict: A dictionary with lengths for each node.
            incomplete_data_dict: A dictionary with incomplete data for each node.
            incomplete_tuples_score: A dictionary with scores for incomplete tuples.
            power_value: A constant power value used in the calculation.
            incomplete_values_dict: A dictionary with incomplete values for each node.

    Returns:
            The calculated gain for the given node.
    """

    final_gain_score_holder = []
    for i in range(lengths_dict[node]):
        w_label = incomplete_data_dict[node][i]
        w_score = incomplete_tuples_score[w_label]
        w_difference = np.round((incomplete_tuples_score[node] - w_score), 3)
        w_second_part = np.round((1 - data_dict_inc_1[node][i]) ** (1 / p_value), 3)
        gain = np.round((w_difference * (1 - w_second_part)), 3)
        final_gain_score_holder.append(gain)
    return sum(final_gain_score_holder)


def update_tuple_lists(
    node: Tuple, incomplete_tuples: List[Tuple], complete_tuples: List[Tuple]
) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Updates the lists of incomplete and complete tuples by moving the given node from the incomplete list to the complete list

    Args:
            node: The tuple (node) to be moved.
            incomplete_tuples: The list of tuples that are considered 'incomplete'.
            complete_tuples: The list of tuples that are considered 'complete'.

    Returns:
            A tuple of the updated lists of incomplete and complete tuples.
    """

    incomplete_tuples.remove(node)
    complete_tuples.append(node)
    return incomplete_tuples, complete_tuples


def update_incomplete_scores(
    node: Tuple,
    updated_incomplete_values: Dict,
    updated_incomplete_lengths: Dict,
    updated_lengths_dict: Dict,
    p_value: np.float64,
    incomplete_tuples_score: Dict,
) -> Dict:
    """
    Updates the scores of incomplete tuples after a node has been moved from the incomplete list to the complete list.

    Args:
            node: The tuple (node) that has been moved.
            updated_incomplete_values: The updated dictionary of incomplete values.
            updated_incomplete_lengths: The updated dictionary of incomplete lengths.
            updated_lengths_dict: The updated dictionary of lengths.
            p_value: A constant power value used in the calculation.
            incomplete_tuples_score: The dictionary of scores for incomplete tuples.

    Returns:
            The updated dictionary of scores for incomplete tuples.
    """

    new_dict = {k: v for k, v in updated_incomplete_values.items() if node in v}
    new_dict_input = {key: updated_incomplete_lengths[key] for key in new_dict.keys()}
    dict_scores_upd = calculate_node_weights(
        new_dict_input, updated_lengths_dict, new_dict.keys(), p_value
    )
    dict_scores_upd.update({node: 0.0})
    incomplete_tuples_score.update(dict_scores_upd)
    return incomplete_tuples_score


def calculate_order_by_greedy(
    incomplete_tuples_score: Dict,
    incomplete_tuples: List[Tuple],
    complete_tuples: List[Tuple],
    greedy_order: List[Tuple],
    lengths_dict: Dict,
    unstacked_data: np.ndarray,
    incomplete_data_dict: Dict,
    p_value: np.float64,
    incomplete_values_dict: Dict,
) -> Tuple[List[Tuple], Dict]:
    """
    Calculates the order for imputing missing values by greedy algorithm.

    Args:
                    incomplete_tuples_score: The dictionary of scores for incomplete tuples.
                    incomplete_tuples: The list of tuples that are considered 'incomplete'.
                    complete_tuples: The list of tuples that are considered 'complete'.
                    greedy_order: The list of tuples in the greedy order.
                    lengths_dict: The dictionary of lengths.
                    unstacked_data: The unstacked dataframe.
                    incomplete_data_dict: The dictionary of incomplete data.
                    p_value: A constant power value used in the calculation.
                    incomplete_values_dict: The dictionary of incomplete values.

    Returns:
                    A tuple of the final greedy order and the updated dictionary of scores for incomplete tuples.
    """

    for _ in range(len(incomplete_tuples)):
        # print(f"Iteration: {_}")
        # print("Before:")
        # print(f"Incomplete tuples: {incomplete_tuples}")
        # print(f"Complete tuples: {complete_tuples}")
        # print(f"Greedy order: {greedy_order}")

        max_value = get_highest_score_node(incomplete_tuples_score, complete_tuples)
        # print(f"Max value: {max_value}")

        if max_value in incomplete_tuples:
            gain_for_element = calculate_gain(
                max_value,
                lengths_dict,
                incomplete_data_dict,
                incomplete_tuples_score,
                p_value,
                incomplete_values_dict,
            )
            incomplete_tuples, complete_tuples = update_tuple_lists(
                max_value, incomplete_tuples, complete_tuples
            )
            inc_complete_updated = unstacked_data[
                ~unstacked_data["indexed"].isin(incomplete_tuples)
            ]
            data_dict_upd_1, data_dict_upd_2 = create_incomplete_connection_dicts(
                inc_complete_updated, incomplete_tuples
            )
            data_dict_upd = get_associated_values_length(
                incomplete_tuples, data_dict_upd_2
            )
            incomplete_tuples_score = update_incomplete_scores(
                max_value,
                data_dict_upd_1,
                data_dict_upd_2,
                data_dict_upd,
                p_value,
                incomplete_tuples_score,
            )
            greedy_order.append(max_value)

        # print("After:")
        # print(f"Incomplete tuples: {incomplete_tuples}")
        # print(f"Complete tuples: {complete_tuples}")
        # print(f"Greedy order: {greedy_order}")
        # print("-" * 50)

    return greedy_order, incomplete_tuples_score


def create_final_order(removed_item: List[str], greedy_order: List[str]) -> List[int]:
    """
    Creates the final order for imputing missing values.

    Args:
            removed_items: The list of tuples (items) that have been removed.
            greedy_order: The list of tuples in the greedy order.

    Returns:
            The final order for imputing missing values.
    """
    final_order = removed_item + greedy_order
    return [int(str(a)[1:]) for a in final_order]


def impute_missing_values(
    dataset: np.ndarray, greedy_order: List[int], K: int = 5
) -> np.ndarray:
    """
    Imputes missing values in a dataframe based on a specified order  computed by greedy algorithm and a specified number of nearest neighbors.

    Args:
            dataset: The dataframe with missing values.
            greedy_order: The list of indices in the greedy order.
            K: The number of nearest neighbors to consider for imputation.

    Returns:
            The dataframe with missing values imputed.
    """

    # Convert the dataset into a Numpy array and create an empty array for the imputed data
    dataset_array = dataset.to_numpy()
    imputed_data = np.zeros_like(dataset_array)

    # Initialize lists to hold complete and incomplete rows, along with their indices
    complete, incompleted, com_ind, incomp_ind = [], [], [], []

    # Iterate through the rows of the dataset
    for i, row in enumerate(dataset_array):
        if math.isnan(np.sum(row)):
            # If the row contains NaN values, add it to the list of incomplete rows
            incompleted.append(row)
            incomp_ind.append(i + 1)
        else:
            # If the row doesn't contain NaN values, add it to the list of complete rows
            complete.append(row)
            com_ind.append(i + 1)

    # print(dataset_array)

    # Create dictionaries mapping indices to complete and incomplete rows
    dict_complete = dict(zip(com_ind, complete))
    dict_incompleted = dict(zip(incomp_ind, incompleted))

    # Order the incomplete rows based on the final greedy impute order
    sorted_incomplete = []
    for v in greedy_order:
        if v in dict_incompleted:
            sorted_incomplete.append((v, dict_incompleted[v]))
        elif v in dict_complete:
            sorted_incomplete.append((v, dict_complete[v]))

    # Split the sorted list into separate lists of indices and rows
    incom_ind, incomplete = zip(*sorted_incomplete)

    # Fill the imputed data with the complete rows
    for i in com_ind:
        imputed_data[i - 1] = dict_complete[i]

    # For each incomplete row, calculate the distance to all complete rows
    for k in incomplete:
        dist = [np.nansum(np.power([k] - row, 2)) for row in complete]

        # Sort the distances and get the indices of the K smallest ones
        dict_pos_sorted = sorted(enumerate(dist), key=itemgetter(1))
        cluster_dictionary = dict_pos_sorted[:K]

        # Calculate the weights for the K nearest neighbours
        k_dist = [s for d, s in cluster_dictionary]
        k_weight = [1 / (d + 1e-10) for d in k_dist]
        k_final_weight = [w / sum(k_weight) for w in k_weight]

        # For each NaN value in the row, calculate the imputed value
        for a in range(np.size(dataset_array, 1)):
            if math.isnan(k[a]):
                complete_values_list = [complete[o][a] for o, p in cluster_dictionary]
                k[a] = round(
                    sum(f * e for f, e in zip(complete_values_list, k_final_weight)), 3
                )

        # Add the imputed row to the list of complete rows
        complete.append(k)

    # Create a DataFrame from the imputed data
    incomplete_dataframe = pd.DataFrame(incomplete, index=incom_ind)
    imputed_data_frame = pd.DataFrame(imputed_data)
    imputed_data_frame.index = imputed_data_frame.index + 1
    final_imputed_df = incomplete_dataframe.combine_first(imputed_data_frame)
    final_imputed_df.columns = dataset.columns

    return final_imputed_df


def draw_graph(thresholded_similarity_matrix_df: pd.DataFrame) -> None:
    """
    Draw a graph based on a boolean similarity matrix.

    Args:
            sim_bool_matrix_df (pd.DataFrame): DataFrame of boolean similarity matrix.

    Returns:
    None
    """
    # Plotting the network
    stacked = thresholded_similarity_matrix_df.set_index(["ID2"]).stack()
    stacked = stacked[stacked == 1]
    edges = stacked.index.tolist()

    # Edges
    G = nx.Graph(edges)
    # Add nodes from the index and columns of the DataFrame
    G.add_nodes_from(thresholded_similarity_matrix_df.set_index("ID2").index)
    G.add_nodes_from(thresholded_similarity_matrix_df.set_index("ID2").columns)

    # Add edges
    G.add_edges_from(edges)

    top_nodes = set(thresholded_similarity_matrix_df.set_index("ID2").columns)

    # Project the bipartite graph
    Gp = nx.bipartite.projected_graph(G, top_nodes)
    Gp.edges()

    nx.draw(G, with_labels=True)
    plt.show()


def main(
    data_df: pd.DataFrame,
    dataset_sim_path: str = "matrix_sim.csv",
    calculate_csv: bool = True,
    calculate_median: bool = True,
    draw: bool = False,
    index_column_name: str = None,
) -> pd.DataFrame:
    """
    Main function to handle the overall process.

    Args:
            data_df (DataFrame): Dataset with missing values.
            dataset_sim_path (str): The path to the dataset similarity file.
            dataset_full_data (str): The path to the full dataset.
            calculate_csv (bool): A flag to determine whether to calculate csv.
            calculate_median (bool): A flag to determine whether to calculate median.
            index_column_name (str): The name of the index column.
    """
    if not index_column_name:
        index_column_name = "ID"
        data_df[index_column_name] = data_df.index + 1
        data_df.columns = data_df.columns.astype(str)


    if calculate_csv:
        scaler = MinMaxScaler()

        # Scale the DataFrame
        scaled_data = scaler.fit_transform(data_df)

        def nan_euclidean(u, v):
            mask = ~np.isnan(u) & ~np.isnan(v)
            dist = np.sqrt(np.sum((u[mask] - v[mask]) ** 2))
            return dist

        # Normalize Euclidean distance inversely related to the similarity is adopted to measure the similarity between two tuples
        distances = pdist(scaled_data, metric=nan_euclidean)
        dist_matrix = squareform(distances)
        distance_df = pd.DataFrame(
            dist_matrix, index=data_df.index, columns=data_df.index
        )

        # Round to make dataset more readable
        distance_df = distance_df.round(3)

        df = convert_similarity_matrix_to_csv(
            distance_df, None, data_df, index_column_name
        )

    else:
        # Load the similarity matrix from file
        df = pd.read_csv(dataset_sim_path)

    if calculate_median:
        # Convert the DataFrame into a single pandas Series
        df_copy = df.drop(columns=[index_column_name])

        df_nonzero = df_copy.loc[:, (df != 0).any(axis=0)]

        melted_df = df_nonzero.melt()

        # Calculate the median of all values
        threshold = melted_df["value"].median(skipna=True)
    else:
        # Used in PDF
        threshold = 0.785

    # print("Median of all values in the DataFrame:", threshold)

    df[index_column_name] = df.index + 1
    df.set_index(index_column_name, inplace=True)

    # Convert into a numpy array and fixing the threshold
    similarity_matrix = np.array(df)

    thresholded_similarity_matrix = apply_binary_threshold_to_similarity_matrix(
        similarity_matrix, threshold
    )

    # Create a new dataframe using the boolean matrix
    thresholded_similarity_matrix_df = pd.DataFrame(thresholded_similarity_matrix)

    # Rename the columns of the new dataframe.
    column_mapping = {
        i: f"X{i+1}" for i in range(thresholded_similarity_matrix_df.shape[1])
    }
    thresholded_similarity_matrix_df.rename(columns=column_mapping, inplace=True)

    # Reindex to start from 1
    thresholded_similarity_matrix_df.index = thresholded_similarity_matrix_df.index + 1

    # Create an index column to get the label in the desired format
    thresholded_similarity_matrix_df["ID2"] = thresholded_similarity_matrix_df.index
    thresholded_similarity_matrix_df["ID2"] = thresholded_similarity_matrix_df[
        "ID2"
    ].apply(lambda x: f"X{x}")

    # Draw the graph
    if draw:
        draw_graph(thresholded_similarity_matrix_df)

    # Reindex to start from 1
    data_df.index = data_df.index + 1
    data_df = data_df.drop(columns=[index_column_name])

    # Create an index column to get the label in the desired format
    data_df["ID2"] = data_df.index
    data_df["ID2"] = data_df["ID2"].apply(lambda x: f"X{x}")

    # Split the null values and create a new dataframe using the similarity boolean matrix.
    # null_data = dataset_with_data[dataset_with_data.isnull().any(axis=1)]
    similarity_bool = thresholded_similarity_matrix_df
    similarity_bool.set_index("ID2", inplace=True)

    # Reset the index
    similarity_bool.reset_index(inplace=True)

    similarity_thres1 = apply_threshold_to_similarity_matrix(
        similarity_matrix, threshold
    )

    # Stacked view to only take the values not equal to 0.0 and then unstacking it.
    score_df = create_similarity_score_dataframe(similarity_thres1)

    stacked = score_df.set_index(["ID2"]).stack()
    stacked = stacked[stacked != 0.0]
    unstacked_data = stacked.unstack()

    total_tuples = unstacked_data.index.to_list()

    # To determine the P value
    incomplete_tuples = unstacked_data.columns.to_list()

    complete_tuples = list(set(total_tuples) - set(incomplete_tuples))

    # Create a new column in unstacked to house the index values.
    unstacked_data["indexed"] = unstacked_data.index

    # Fill the unstacked df for NaN
    unstacked_data.fillna(0, inplace=True)

    # Keep only complete neighbours for the incomplete tuples in this "inc_complete" dataframe.
    inc_complete = unstacked_data[~unstacked_data["indexed"].isin(incomplete_tuples)]

    # Keep only incomplete neighbours for the incomplete tuples in this "inc_complete" dataframe.
    inc_incomplete = unstacked_data[unstacked_data["indexed"].isin(incomplete_tuples)]

    # Get the complete neighbours and their similarity scores
    node_data_dict, similarity_dict = create_incomplete_connection_dicts(
        inc_complete, incomplete_tuples
    )

    # P-value:
    p_value = calculate_p_value(incomplete_tuples, node_data_dict)
    # print(f"P values is: {p_value}")

    length_dict = get_associated_values_length(incomplete_tuples, similarity_dict)

    # Initial Node weight calculator
    scores_dict = calculate_node_weights(
        similarity_dict, length_dict, incomplete_tuples, p_value
    )
    # print("Initial Node weight before Imputation Order calc:\n", scores_dict)

    # Incomplete connections and similarities
    data_dict_inc, incomplete_values_dict = create_incomplete_connection_dicts(
        inc_incomplete, incomplete_tuples
    )
    # print("Incomplete neighbours to incomplete tuples:\n", data_dict_inc)
    # print("Incomplete neighbours similarity Score:\n", incomplete_values_dict)

    (
        data_final_scores,
        lengths_dict,
        score_list_name,
        removed_item,
    ) = process_node_weights(incomplete_tuples, incomplete_values_dict, scores_dict)

    # Initialization of the data
    incomplete_tuples_score = data_final_scores
    updated_incomplete_tuples = score_list_name.copy()
    updated_complete_tuples = complete_tuples.copy()
    greedy_order = []

    # Running the Greedy Algorithm
    greedy_order, incomplete_tuples_score = calculate_order_by_greedy(
        incomplete_tuples_score,
        updated_incomplete_tuples,
        updated_complete_tuples,
        greedy_order,
        lengths_dict,
        unstacked_data,
        data_dict_inc,
        p_value,
        incomplete_values_dict,
    )

    # Creating the final order for feeding to the KNN
    greedy_order = create_final_order(removed_item, greedy_order)

    # KNN Modelling Approach
    dataset = data_df

    dataset.set_index("ID2", inplace=True)

    imputed_dataset = impute_missing_values(dataset, greedy_order)

    return imputed_dataset


if __name__ == "__main__":
    cars_dataset_25 = "dataset/cleaned_csv_file_25_missing.csv"
    cars_dataset_500 = "dataset/cleaned_csv_file_500_missing.csv"
    example_dataset = "clustering/order-sensitive-imputation/SDD.csv"

    # Custom dataset
    data_df = pd.read_csv(cars_dataset_25)
    imputed_dataset = main(
        data_df=data_df,
        dataset_sim_path="clustering/order-sensitive-imputation/sim_cars.csv",
        draw=False,
        calculate_csv=True,
        calculate_median=True,
    )

    # Example
    # data_df = pd.read_csv(example_dataset)
    # imputed_dataset = main(
    #     data_df=data_df,
    #     dataset_sim_path="clustering/order-sensitive-imputation/sim.csv",
    #     draw=False,
    #     calculate_csv=False,
    #     calculate_median=False,
    # )

    print(f"Final result:\n{imputed_dataset}")
