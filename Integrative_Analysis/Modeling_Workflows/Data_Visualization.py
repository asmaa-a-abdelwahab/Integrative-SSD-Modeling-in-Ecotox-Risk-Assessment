import pandas as pd
import matplotlib.pyplot as plt


def analyze_species_lifestage_distribution(
    df, species_column, endpoint_column, lifestage_column, figure_path, csv_path
):
    # Get unique species from the specified species column
    species_groups = df[species_column].unique()

    # Dictionary to store the pivoted DataFrame for each species
    species_pivot_tables = {}

    for species in species_groups:
        # Filter DataFrame for the current species
        df_species = df[df[species_column] == species]

        # Group by endpoint and lifestage, and count occurrences
        endpoint_lifestage_counts = df_species.groupby(
            [endpoint_column, lifestage_column]
        ).size()

        # Convert the counts to a DataFrame
        endpoint_lifestage_counts_df = endpoint_lifestage_counts.reset_index(
            name="Count"
        )

        # Pivot the DataFrame
        pivoted_df = endpoint_lifestage_counts_df.pivot(
            index=lifestage_column, columns=endpoint_column, values="Count"
        ).fillna(0)

        if len(pivoted_df) > 0:
            # Store the pivoted DataFrame in the dictionary
            species_pivot_tables[species] = pivoted_df

            # Plotting the data
            pivoted_df.plot(
                kind="bar", stacked=False, figsize=(12, 8), colormap="viridis"
            )

            # Customize the plot
            plt.title(
                f"Distribution of {lifestage_column} per {endpoint_column} for {species}"
            )
            plt.xlabel(lifestage_column)
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.legend(
                title=endpoint_column, bbox_to_anchor=(1.05, 1), loc="upper left"
            )

            # Save and show the plot
            plt.tight_layout()
            plt.savefig(
                f"{figure_path}/LifeStageData_curated_distribution_{species}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.show()

            # Optionally print and save the pivoted DataFrame
            print(f"Data Distribution for {species}:")
            print(pivoted_df, "\n")
            pivoted_df.to_csv(
                f"{csv_path}/LifeStageData_curated_distribution_{species}.csv"
            )

    return species_pivot_tables


species_pivot_tables = analyze_species_lifestage_distribution(
    df=pd.read_csv(
        "../Modeling_Workflows/Processed_Combinations/TEST/Curated_Data_TEST-Descriptors_processed.csv"
    ),
    species_column="Species Group",
    endpoint_column="Endpoint",
    lifestage_column="Broad Lifestage Group",
    figure_path="../Modeling_Workflows/Processed_Combinations/TEST/Figures",
    csv_path="../Modeling_Workflows/Processed_Combinations/TEST/Stats",
)


# import pandas as pd

# # Define the data
# data = {
#     "Compound": [
#         "Acetyl salicylic acid",
#         "Amikacin",
#         "Amiodarone",
#         "Amitriptyline",
#         "Amphotericin B",
#         "Azathioprine",
#         "Captopril",
#         "Chlorpromazine",
#         "Citrate",
#         "Clofibrate",
#         "Cumene hydroperoxide",
#         "Cyclosporine A",
#         "Deoxycholate",
#         "Doxycycline",
#         "Etoposide",
#         "Fenofibrate",
#         "Flutamide",
#         "Ketotifen",
#         "Maprotiline",
#         "Perhexiline",
#         "Phenobarbital",
#         "Rotenone",
#         "Tamoxifen",
#         "Tetracycline",
#         "Tianeptine",
#         "Ticlopidine",
#         "Valproate",
#         "Zidovudine",
#     ],
#     "Abbreviation": [
#         "ASA",
#         "AMK",
#         "AMD",
#         "AMT",
#         "APB",
#         "AZA",
#         "CAP",
#         "CHL",
#         "CIT",
#         "CLO",
#         "CHP",
#         "CYA",
#         "DEO",
#         "DOX",
#         "ETO",
#         "FEN",
#         "FLU",
#         "KET",
#         "MAP",
#         "PER",
#         "PHE",
#         "ROT",
#         "TAM",
#         "TET",
#         "TIA",
#         "TIC",
#         "VAL",
#         "ZID",
#     ],
#     "Classification": [
#         "S",
#         "NS-NH",
#         "S",
#         "NS-H",
#         "S",
#         "S",
#         "NS-H",
#         "NS-H",
#         "NS-NH",
#         "S",
#         "NS-H",
#         "S",
#         "S",
#         "S",
#         "NS-H",
#         "S",
#         "NS-H",
#         "NS-NH",
#         "NS-H",
#         "S",
#         "NS-H",
#         "NS-H",
#         "S",
#         "S",
#         "S",
#         "S",
#         "S",
#         "S",
#     ],
#     "DILI category": [
#         "Not assigned",
#         "Not assigned",
#         "Severe",
#         "Not assigned",
#         "Not assigned",
#         "Mild",
#         "Severe",
#         "Mild",
#         "Not applicable",
#         "Mild",
#         "Not applicable",
#         "Mild",
#         "Not applicable",
#         "Mild",
#         "Not applicable",
#         "Not assigned",
#         "Most concern",
#         "Not assigned",
#         "Moderate",
#         "Not assigned",
#         "Mild",
#         "Not applicable",
#         "Moderate",
#         "Mild",
#         "Not assigned",
#         "Moderate",
#         "Severe",
#         "Severe",
#     ],
#     "Label": [
#         "",
#         "",
#         "BW",
#         "",
#         "",
#         "BW",
#         "AR",
#         "AR",
#         "",
#         "WP",
#         "",
#         "WP",
#         "",
#         "WP",
#         "",
#         "",
#         "BW",
#         "",
#         "AR",
#         "WD",
#         "WP",
#         "",
#         "WP",
#         "WP",
#         "",
#         "WP",
#         "BW",
#         "BW",
#     ],
#     "Toxicity": [
#         "MI, OS",
#         "MI, OS",
#         "MI",
#         "MI, OS",
#         "MI, OS",
#         "MI, OS",
#         "AP",
#         "MI",
#         "",
#         "MI",
#         "OS",
#         "MI",
#         "MI",
#         "MI, AP",
#         "AP",
#         "MI, AP",
#         "MI, BA",
#         "",
#         "MI, CA",
#         "MI",
#         "MI",
#         "MI, AP",
#         "MI, OS, CA",
#         "MI",
#         "MI",
#         "OS, BA",
#         "MI, OS",
#         "MI, OS",
#     ],
#     "Concentrations (µM)": [
#         "62.5, 125, 250, 500",
#         "2500, 5000, 10,000, 20,000",
#         "6.25, 12.5, 25, 50",
#         "12.5, 25, 50, 100",
#         "5, 10, 20, 30",
#         "100, 200, 300, 400",
#         "25, 50, 100, 200",
#         "5, 10, 20, 40",
#         "1, 10, 100, 1000",
#         "31.5, 62.5, 125, 250",
#         "50, 100, 250, 500",
#         "12.5, 25, 50, 100",
#         "5, 10, 20, 40",
#         "62.5, 125, 250, 500",
#         "12.5, 25, 50, 100",
#         "62.5, 125, 250, 500",
#         "50, 100, 200, 400",
#         "50, 75, 100, 150",
#         "12.5, 25, 50, 100",
#         "10, 15, 17.5, 20",
#         "500, 750, 1000, 1500",
#         "0.5, 1, 5, 10",
#         "5, 10, 20, 30",
#         "50, 100, 200, 400",
#         "125, 250, 500, 1000",
#         "100, 200, 500, 1000",
#         "1000, 2000, 4000, 8000",
#         "100, 200, 400, 800",
#     ],
#     "Cmax (µM)": [
#         "7",
#         "34.30",
#         "1.94",
#         "0.145",
#         "1.84",
#         "0.03",
#         "4",
#         "0.5",
#         "NA",
#         "439",
#         "NA",
#         "0.61",
#         "NA",
#         "10.01",
#         "17",
#         "14.96",
#         "6",
#         "0.00015",
#         "0.16",
#         "1.4",
#         "174",
#         "NA",
#         "0.25",
#         "14.19",
#         "0.77",
#         "7.58",
#         "481",
#         "5.43",
#     ],
# }

# # Create the DataFrame
# df = pd.DataFrame(data)

# # Save to an Excel file
# file_path = "./converted_table.xlsx"
# df.to_excel(file_path, index=False)

# file_path
