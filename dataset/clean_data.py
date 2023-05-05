import pandas as pd

# Load the CSV file into a DataFrame
file_path = './dataset/all_cars_1941-2022.csv'
df = pd.read_csv(file_path)

# Create a new column 'ID' with a unique identifier for each row
df['ID'] = df.index + 1

# Keep only the desired columns
columns_to_keep = [
    'ID', 'year', 'engine_cc', 'cylinder', 'valves_per_cylinder', 'power_ps', 'torque_nm',
    'engine_compression', 'doors', 'hwy', 'mixed', 'city', 'fuel_cap_l'
]
df = df[columns_to_keep]

# Split the 'engine_compression' column using ':' as the separator, and keep only the first part (index 0)
df['engine_compression'] = df['engine_compression'].str.split(':').str.get(0)

# Print the number of items before cleaning
print("Number of items before cleaning:", df.shape[0])

# Remove rows with missing data
df_clean = df.dropna()

# Print the number of items after cleaning
print("Number of items after cleaning:", df_clean.shape[0])

# Save the cleaned DataFrame to a new CSV file
clean_csv_path = './dataset/cleaned_csv_file.csv'
df_clean.to_csv(clean_csv_path, index=False)
