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
