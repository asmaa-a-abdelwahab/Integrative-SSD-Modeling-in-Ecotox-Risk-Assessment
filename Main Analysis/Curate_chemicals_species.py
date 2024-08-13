import pandas as pd

# Paths to your files
file_a_path = 'noecall.xlsx' # File with all chemicals'  # Replace with the file path
file_b_path = 'noecafterknime.xlsx'   # File Path of Curated Chemical  # Replace with the file path
# Load the data into DataFrames
df_a = pd.read_excel (file_a_path)
df_b = pd.read_excel (file_b_path)
df_a
# Ensure that both DataFrames have a column named 'Chemical Name'
chemical_name_column = 'Chemical Name'
# Filter df_a to only include rows where the chemical_name exists in df_b
filtered_df_a = df_a[df_a[chemical_name_column].isin(df_b[chemical_name_column])]
# Save the filtered DataFrame to a new file
filtered_file_a_path = "NOECremovedthechemicalwasnotcurated.csv"
filtered_df_a.to_csv(filtered_file_a_path, index=False)
filtered_df_a
species_to_remove = ['Fish', 'Flowers, Trees, Shrubs, Ferns', 'Fungi', 'Echinodermata', 'Mammals', 'Mussels', 'Microorganisms', 'Rotifers', 'Birds', 'Moss, Hornworts', 'Cyanophyceae']

# Filter the DataFrame to remove specified species
df_filtered2 = filtered_df_a[~filtered_df_a['Species Group'].isin(species_to_remove)]
df_filtered2

# Define the species to replace
species_to_replace = ['Insects/Spiders', 'Molluscs', 'Worms', 'Reptiles']
new_df = df_filtered2.copy()
new_df['Species Group'] = new_df['Species Group'].replace(species_to_replace, 'Other Invertebrates')
new_df
new_df.to_csv('NOECremovedunspeciesandmergedspecies.csv', index=False)
