import os
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.feature_selection import VarianceThreshold
from rdkit import Chem
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    filename="data_processing.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class DataPipeline:
    def __init__(self, data_files, target_column, id_column="SMILES"):
        self.data_files = data_files
        self.target_column = target_column
        self.id_column = id_column
        self.df = None
        self.name = None
        logging.info("Initialized DataPipeline class.")

    def load_main_data(self):
        try:
            main_file = next(iter(self.data_files.items()))
            name, path = main_file
            logging.info(f"Loading main dataset: {name}")
            self.df = pd.read_csv(path)
            # Define the columns you want to keep
            columns_to_select = [
                "Species Group",
                "Broad Lifestage Group",
                "Exposure Type",
                "Media Type",
                "Endpoint",
                "Observed Duration Mean (Days)",
                "Conc 1 Mean (Standardized)",
                "SMILES",
                "Cleaned_SMILES",
                "Activity_Cluster",
            ]

            # Filter the DataFrame to keep only the specified columns
            self.df = self.df[columns_to_select]
            logging.info(f"Loaded main dataset with shape: {self.df.shape}")

            if "Cleaned_SMILES" in self.df.columns:
                if "SMILES" in self.df.columns:
                    self.df.drop(columns=["SMILES"], inplace=True)
                self.df.rename(columns={"Cleaned_SMILES": "SMILES"}, inplace=True)

        except Exception as e:
            logging.error(f"Error loading main dataset: {e}")

    def preprocess_and_merge_file(self, file_name, file_path, chunksize=100):
        """
        Load, preprocess, and merge one dataset file at a time with the main dataset.
        Handles specific cases for 'CDK', 'OPERA', 'Padel' and processes large files using chunks if they exceed 1000 columns.
        """
        try:
            logging.info(f"Processing file: {file_name}")

            # Determine the number of columns in the file
            num_columns = len(pd.read_csv(file_path, nrows=1).columns)

            # If the file has 1000 or fewer columns, process the entire file at once
            if num_columns <= 1000:
                logging.info(
                    f"Processing the entire file at once for file: {file_name}"
                )
                df = pd.read_csv(file_path)

                # Special handling for "CDK", "OPERA", "Padel"
                if any(
                    substring in file_name for substring in ["CDK", "OPERA", "Padel"]
                ):
                    smiles_list = self.read_smiles_from_sdf(
                        "../DataSet/5.LifeStageData-Cleaned_SMILES.sdf"
                    )
                    if len(smiles_list) < len(df):
                        raise ValueError(
                            f"SMILES list is shorter than the data rows in '{file_name}'."
                        )

                    df["SMILES"] = smiles_list[: len(df)]

                # Handle SMILES column renaming
                if "Cleaned_SMILES" in df.columns:
                    if "SMILES" in df.columns:
                        df.drop(columns=["SMILES"], inplace=True)
                    df.rename(columns={"Cleaned_SMILES": "SMILES"}, inplace=True)

                elif "Query" in df.columns:
                    if "SmilesRan" in df.columns:
                        df.drop(columns=["SmilesRan"], inplace=True)
                    df.rename(columns={"Query": "SMILES"}, inplace=True)

                # Process and merge the entire DataFrame
                self.df[self.id_column] = self.df[self.id_column].astype(str)
                df[self.id_column] = df[self.id_column].astype(str)
                df.set_index(self.id_column, inplace=True)
                self.df = self.df.join(df, on=self.id_column, how="left")

                # for col in df.columns:
                #     if col != self.id_column:
                #         self.df[col] = self.df[self.id_column].map(
                #             df.set_index(self.id_column)[col]
                #         )
                # self.df.drop_duplicates(inplace=True)

                # self._process_and_merge_chunk(file_name, df)
            else:  # Process the file in chunks if it has more than 1000 columns
                logging.info(f"Processing file in chunks for file: {file_name}")

                if any(
                    substring in file_name
                    for substring in ["TEST", "CDK", "OPERA", "Padel"]
                ):
                    smiles_list = self.read_smiles_from_sdf(
                        "../DataSet/5.LifeStageData-Cleaned_SMILES.sdf"
                    )
                    smiles_pointer = 0

                    for i, chunk in enumerate(
                        pd.read_csv(file_path, chunksize=chunksize)
                    ):
                        logging.info(f"Processing chunk {i + 1} for file: {file_name}")

                        # Determine the SMILES subset for this chunk
                        chunk_length = len(chunk)
                        if smiles_pointer + chunk_length > len(smiles_list):
                            raise ValueError(
                                f"SMILES list exhausted at chunk {i + 1} of '{file_name}'."
                            )

                        chunk["SMILES"] = smiles_list[
                            smiles_pointer : smiles_pointer + chunk_length
                        ]
                        smiles_pointer += chunk_length

                        # Process and merge the chunk
                        self._process_and_merge_chunk(file_name, chunk)

                else:
                    for i, chunk in enumerate(
                        pd.read_csv(file_path, chunksize=chunksize)
                    ):
                        logging.info(f"Processing chunk {i + 1} for file: {file_name}")

                        # Handle SMILES column renaming
                        if "Cleaned_SMILES" in chunk.columns:
                            if "SMILES" in chunk.columns:
                                chunk.drop(columns=["SMILES"], inplace=True)
                            chunk.rename(
                                columns={"Cleaned_SMILES": "SMILES"}, inplace=True
                            )

                        # Process and merge the chunk
                        self._process_and_merge_chunk(file_name, chunk)

            logging.info(f"Successfully processed file: {file_name}")

        except Exception as e:
            logging.error(f"Error processing file {file_name}: {e}")

    def _process_and_merge_chunk(self, file_name, chunk):
        """
        Process individual chunk: drop low-informative columns and merge with main DataFrame.
        """
        # Drop columns with all NaN, all 0, or all 1 values
        threshold = 0.5 * len(chunk)
        chunk = chunk.loc[
            :,
            ~chunk.apply(
                lambda col: col.isna().all()
                or (col == 0).all()
                or (col == 1).all()
                or col.isna().sum() > threshold
            ),
        ]

        # Ensure id_column exists in both DataFrames
        if self.id_column not in self.df.columns:
            logging.error(f"{self.id_column} not found in main DataFrame")
            return
        if self.id_column not in chunk.columns:
            logging.error(f"{self.id_column} not found in chunk DataFrame")
            return

        # Convert id_column to string to avoid data type mismatches
        self.df[self.id_column] = self.df[self.id_column].astype(str)
        chunk[self.id_column] = chunk[self.id_column].astype(str)

        # Check for duplicate SMILES in the chunk
        if chunk[self.id_column].duplicated().any():
            logging.warning(
                f"Duplicate {self.id_column} values found in chunk for file: {file_name}"
            )

        # Merge chunk into main DataFrame
        try:
            merged_df = pd.merge(
                self.df, chunk, on=self.id_column, how="left", suffixes=("", "_dup")
            )

            # Handle duplicate columns after merge
            for col in merged_df.columns:
                if col.endswith("_dup"):
                    original_col = col[:-4]
                    if merged_df[original_col].isna().all():
                        merged_df[original_col] = merged_df[col]
                    merged_df.drop(columns=[col], inplace=True)

            self.df = merged_df
            self.df.drop_duplicates(inplace=True)
            logging.info(f"Merged chunk, resulting shape: {self.df.shape}")
        except Exception as e:
            logging.error(f"Error merging chunk: {e}")

    def read_smiles_from_sdf(self, sdf_file):
        """
        Reads an SDF file and extracts SMILES strings.

        Parameters:
        - sdf_file (str): Path to the SDF file.

        Returns:
        - smiles_list (list): List of SMILES strings.
        """
        # Use SDMolSupplier to load molecules
        supplier = Chem.SDMolSupplier(sdf_file)

        # Extract SMILES for valid molecules
        smiles_list = [Chem.MolToSmiles(mol) for mol in supplier if mol is not None]

        return smiles_list

    def drop_constant_columns(self):
        try:
            constant_columns = [
                col
                for col in self.df.columns
                if self.df[col].nunique() == 1 and self.df[col].iloc[0] in [0, 1]
            ]
            self.df.drop(columns=constant_columns, inplace=True)
            logging.info(f"Dropped columns with all values 0 or 1: {constant_columns}")
        except Exception as e:
            logging.error(f"Error dropping constant columns: {e}")

    def drop_low_variance_features(self, threshold=0.01):
        """
        Drop features with low variance (useful for binary descriptors/fingerprints).
        """
        try:
            numeric_df = self.df.select_dtypes(include=["number"])
            selector = VarianceThreshold(threshold=threshold)
            reduced_data = selector.fit_transform(numeric_df)

            # Align columns using get_support
            selected_columns = numeric_df.columns[selector.get_support()]
            self.df = pd.concat(
                [
                    self.df.drop(columns=numeric_df.columns),
                    pd.DataFrame(reduced_data, columns=selected_columns),
                ],
                axis=1,
            )

            logging.info(
                f"Dropped low variance features. Remaining shape: {self.df.shape}"
            )
        except Exception as e:
            logging.error(f"Error dropping low variance features: {e}")

    def drop_low_correlation_features(self, low_threshold=0.01):
        """
        Drop features with low correlation to the target column.
        Only numerical features are considered.
        """
        try:
            if self.target_column not in self.df.columns:
                logging.warning(
                    f"Target column '{self.target_column}' not found. Skipping correlation filter."
                )
                return

            # Filter numeric columns only
            numeric_df = self.df.select_dtypes(include=["number"])

            if self.target_column not in numeric_df.columns:
                logging.warning(
                    "Target column is not numerical. Skipping correlation filter."
                )
                return

            # Calculate correlations
            low_corr_columns = []
            columns = [col for col in numeric_df.columns if col != self.target_column]

            def calculate_correlation(feature):
                """Safely calculate Spearman correlation."""
                return numeric_df[feature].corr(
                    numeric_df[self.target_column], method="spearman"
                )

            # Use ThreadPoolExecutor for parallel correlation calculation
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {
                    executor.submit(calculate_correlation, col): col for col in columns
                }
                for future in as_completed(futures):
                    col = futures[future]
                    try:
                        corr_value = abs(future.result())
                        if (
                            corr_value < low_threshold
                        ):  # Identify columns with low correlation
                            low_corr_columns.append(col)
                    except Exception as e:
                        logging.error(
                            f"Error calculating correlation for column {col}: {e}"
                        )

            # Drop columns with low correlation
            if low_corr_columns:
                logging.info(
                    f"Dropping columns with low correlation: {low_corr_columns}"
                )
                self.df = self.df.drop(columns=low_corr_columns, errors="ignore")

            logging.info(
                f"Dropped columns with correlation < {low_threshold}. Remaining shape: {self.df.shape}"
            )
        except Exception as e:
            logging.error(f"Error dropping low correlation features: {e}")

    def show_missing_value_stats(self):
        try:
            missing_values = self.df.isnull().sum()
            missing_percentage = (missing_values / len(self.df)) * 100
            missing_stats = pd.DataFrame(
                {"Missing Values": missing_values, "Percentage": missing_percentage}
            ).sort_values(by="Missing Values", ascending=False)
            logging.info(f"Missing value statistics:\n{missing_stats}")
            missing_stats.to_csv("missing_value_stats.csv")
            print("Missing Value Statistics:")
            print(missing_stats)
        except Exception as e:
            logging.error(f"Error generating missing value stats: {e}")

    def exploratory_analysis(self):
        """
        Perform basic exploratory data analysis for both numerical and categorical features,
        and calculate correlations for numerical features if a target column exists.
        """
        try:
            logging.info("Performing exploratory analysis.")
            print("=== Basic Data Information ===")
            print(self.df.info())
            print("\n=== Missing Values ===")
            print(self.df.isnull().sum())

            # Separate numerical and categorical columns
            numerical_features = self.df.select_dtypes(include=["number"]).columns
            categorical_features = self.df.select_dtypes(exclude=["number"]).columns

            # Numerical Features Analysis
            print("\n=== Numerical Features Analysis ===")
            if not numerical_features.empty:
                print("Descriptive Statistics for Numerical Features:")
                print(self.df[numerical_features].describe())
            else:
                print("No numerical features found.")

            # Categorical Features Analysis
            print("\n=== Categorical Features Analysis ===")
            if not categorical_features.empty:
                for col in categorical_features:
                    print(f"\nFeature: {col}")
                    print("Unique Values:", self.df[col].nunique())
                    print("Value Counts:")
                    print(self.df[col].value_counts())
            else:
                print("No categorical features found.")

            # Correlation Analysis with Target Column (Numerical only)
            if (
                self.target_column in self.df.columns
                and self.target_column in numerical_features
            ):
                print("\n=== Correlation with Target Column ===")
                correlation = (
                    self.df[numerical_features]
                    .corr()[self.target_column]
                    .dropna()
                    .sort_values(ascending=False)
                )
                print("Top Correlations:")
                print(correlation)
                logging.info(f"Top correlations with target column:\n{correlation}")
            else:
                print(
                    "\nTarget column not found or is not numerical. Skipping correlation analysis."
                )
                logging.warning("Target column not found for correlation analysis.")
        except Exception as e:
            logging.error(f"Error during exploratory analysis: {e}")
            print(f"An error occurred during exploratory analysis: {e}")

    def parallel_merge(self):
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.preprocess_and_merge_file, name, path): name
                for name, path in list(self.data_files.items())[1:]
            }

            for future in as_completed(futures):
                file_name = futures[future]
                try:
                    future.result()
                    logging.info(f"Successfully processed {file_name}")
                except Exception as e:
                    logging.error(f"Error processing {file_name}: {e}")

    def visualize_data(self):
        try:
            self.name = "Data&" + list(self.data_files.keys())[1] + "_Combined"
            logging.info("Generating visualizations.")
            plt.figure(figsize=(10, 6))

            # Check if target column exists and has data
            if self.target_column in self.df.columns:
                if not self.df[self.target_column].dropna().empty:
                    if pd.api.types.is_numeric_dtype(self.df[self.target_column]):
                        # Line plot for trends
                        plt.figure(figsize=(10, 6))
                        plt.plot(
                            self.df[self.target_column].dropna(),
                            marker="o",
                            linestyle="-",
                        )
                        plt.title(f"Trend of {self.target_column}")
                        plt.xlabel("Index")
                        plt.ylabel(self.target_column)
                        plt.savefig(
                            f"{self.name}_{self.target_column}_trend.png",
                            dpi=300,
                            bbox_inches="tight",
                        )
                        plt.close()

                        logging.info("Visualizations generated successfully.")

            # Correlation Heatmap
            numeric_df = self.df.select_dtypes(include=["number"])
            if not numeric_df.empty:
                plt.figure(figsize=(12, 10))
                sns.heatmap(numeric_df.corr(), annot=False, cmap="coolwarm")
                plt.title("Correlation Heatmap")
                plt.savefig(
                    f"{self.name}_correlation_heatmap.png",
                    dpi=300,
                    bbox_inches="tight",
                    format="png",
                    pad_inches=0.1,
                )

            # Pairplot for top correlated features
            if self.target_column in numeric_df.columns:
                top_corr_features = (
                    numeric_df.corr()[self.target_column]
                    .abs()
                    .sort_values(ascending=False)
                    .head(5)
                    .index
                )
                sns.pairplot(self.df[top_corr_features])
                plt.savefig(
                    f"{self.name}_pairplot.png",
                    dpi=300,
                    bbox_inches="tight",
                    format="png",
                    pad_inches=0.1,
                )

            logging.info("Visualizations generated successfully.")
        except Exception as e:
            logging.error(f"Error generating visualizations: {e}")

    def preprocess_and_save(self, threshold_ratio=0.7):
        """
        Preprocess data files by dropping columns with all NaN, all 0, all 1 values,
        or columns exceeding a missing data threshold, and save them back.

        Parameters:
        - threshold_ratio (float): Ratio of missing data to consider a column for removal.
        """
        for name, file_path in self.data_files.items():
            try:
                logging.info(f"Processing file: {name}")

                # Read the dataset
                data = pd.read_csv(file_path)

                # Calculate the threshold for missing values
                threshold = threshold_ratio * len(data)

                # Drop columns with all NaN, all 0, all 1, or exceeding the threshold of missing data
                data = data.loc[
                    :,
                    ~data.apply(
                        lambda col: col.isna().all()
                        or (col == 0).all()
                        or (col == 1).all()
                        or col.isna().sum() > threshold,
                    ),
                ]

                # Save the processed dataset
                data.to_csv(f"{file_path}", index=False)
                logging.info(
                    f"File processed and saved: {file_path} with shape {data.shape}"
                )

            except Exception as e:
                logging.error(f"Error processing file {name}: {e}")

    def run_pipeline(self):
        try:
            # Load the main data
            self.load_main_data()

            # Check if self.df is properly loaded
            if self.df is None or self.df.empty:
                logging.error("Main dataset is empty or not loaded properly.")
                return

            # Preprocess and save data files
            self.preprocess_and_save()

            # Perform other processing steps
            self.parallel_merge()
            self.drop_low_variance_features()
            self.drop_constant_columns()
            self.drop_low_correlation_features()
            self.show_missing_value_stats()
            self.exploratory_analysis()
            logging.info("Pipeline execution completed.")

        except Exception as e:
            logging.error(f"Error during pipeline execution: {e}")


if __name__ == "__main__":
    # Define data files
    data_files = {
        "Curated_Data": "../DataSet/5.LifeStageData-CompoundsCurated.csv",
        "TEST-Descriptors": "../DataSet/6.LifeStageData-EPA-TEST_Descriptors.csv",
        # "Physicochemical_PubChem": "../DataSet/6.LifeStageData-PhysicoChemicalProperties.csv",
        "MACCS": "../DataSet/6.MACCS_fingerprints.csv",
        "Morgan": "../DataSet/6.Morgan_fingerprints.csv",
        "AtomPair": "../DataSet/6.AtomPair_fingerprints.csv",
        "Descriptors": "../DataSet/6.LifeStageData-Descriptors.csv",
        "CDK": "../DataSet/6.LifeStageData-CDK_Descriptors.csv",
        "RDKit": "../DataSet/6.RDKitFP_fingerprints.csv",
        "Torsion": "../DataSet/6.Torsion_fingerprints.csv",
    }

    # Create output directory
    output_dir = "Processed_Combinations"
    os.makedirs(output_dir, exist_ok=True)

    # Step 1:  the first file
    first_file_name, first_file_path = list(data_files.items())[0]

    # Step 2: Use the pipeline for further combinations
    for other_file_name, other_file_path in list(data_files.items())[1:]:
        try:
            print(f"Processing combination: {first_file_name} + {other_file_name}")

            # Define the combination of files for the pipeline
            combination_files = {
                f"{first_file_name}": first_file_path,
                other_file_name: other_file_path,
            }

            # Initialize the DataPipeline
            pipeline = DataPipeline(
                data_files=combination_files, target_column="Conc 1 Mean (Standardized)"
            )

            # Run the pipeline
            pipeline.run_pipeline()

            # Save the processed data
            output_file = os.path.join(
                output_dir,
                f"{first_file_name}_{other_file_name}_processed.csv",
            )
            pipeline.df.to_csv(output_file, index=False)

            # Generate visualizations
            pipeline.visualize_data()

            print(f"Saved processed combination to: {output_file}")

        except Exception as e:
            print(
                f"Error processing combination: {first_file_name} + {other_file_name}: {e}"
            )


# Example Usage
# if __name__ == "__main__":
#     data_files = {
#         "Curated_Data": "../DataSet/5.LifeStageData-CompoundsCurated.csv",
#         "MACCS": "../DataSet/6.MACCS_fingerprints.csv",
#         "Morgan": "../DataSet/6.Morgan_fingerprints.csv",
#         "AtomPair": "../DataSet/6.AtomPair_fingerprints.csv",
#         "TEST-Descriptors": "../DataSet/6.LifeStageData-EPA-TEST_Descriptors.csv",
#         "Physicochemical_PubChem": "../DataSet/5.LifeStageData-PhysicoChemicalProperties.csv",
#         "Descriptors": "../DataSet/6.LifeStageData-Descriptors.csv",
#         "CDK": "../DataSet/6.LifeStageData-CDK_Descriptors.csv",
#         "RDKit": "../DataSet/6.RDKitFP_fingerprints.csv",
#         "Torsion": "../DataSet/6.Torsion_fingerprints.csv",
#     }

#     pipeline = DataPipeline(
#         data_files=data_files, target_column="Conc 1 Mean (Standardized)"
#     )

#     pipeline.run_pipeline(data_files)
#     pipeline.df.to_csv("processed_data.csv", index=False)
#     pipeline.visualize_data()
#     print("Pipeline execution finished. Processed data saved to 'processed_data.csv'.")
