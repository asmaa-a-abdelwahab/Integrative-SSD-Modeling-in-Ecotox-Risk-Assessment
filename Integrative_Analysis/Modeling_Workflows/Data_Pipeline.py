import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    filename="data_processing.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class DataPipeline:
    def __init__(self, data_files, target_column, id_column="SMILES"):
        """
        Initialize the DataPipeline class.

        Parameters:
        - data_files (dict): Dictionary of dataset file paths.
        - target_column (str): The name of the target variable column.
        - id_column (str): Column to use for merging datasets, default is 'SMILES'.
        """
        self.data_files = data_files
        self.target_column = target_column
        self.id_column = id_column
        self.df = None
        logging.info("Initialized DataPipeline class.")

    def load_main_data(self):
        """
        Load the main dataset, which is the first file in the data_files dictionary.
        """
        try:
            main_file = next(iter(self.data_files.items()))
            name, path = main_file
            logging.info(f"Loading main dataset: {name}")
            self.df = pd.read_csv(path)
            logging.info(f"Loaded main dataset with shape: {self.df.shape}")

            # Standardize SMILES column
            if "Cleaned_SMILES" in self.df.columns:
                if "SMILES" in self.df.columns:
                    self.df.drop(columns=["SMILES"], inplace=True)
                self.df.rename(columns={"Cleaned_SMILES": "SMILES"}, inplace=True)

        except Exception as e:
            logging.error(f"Error loading main dataset: {e}")

    def preprocess_and_merge_file(self, file_name, file_path):
        """
        Load, preprocess, and merge one dataset file at a time with the main dataset.
        """
        try:
            logging.info(f"Processing file: {file_name}")
            new_data = pd.read_csv(file_path, low_memory=False)

            # Standardize SMILES column
            if "Cleaned_SMILES" in new_data.columns:
                if "SMILES" in new_data.columns:
                    new_data.drop(columns=["SMILES"], inplace=True)
                new_data.rename(columns={"Cleaned_SMILES": "SMILES"}, inplace=True)

            # Merge with the main dataset
            new_data = new_data.loc[
                :,
                ~new_data.apply(
                    lambda col: col.isna().all() or (col == 0).all() or (col == 1).all()
                ),
            ]
            self.df = pd.merge(self.df, new_data, on=self.id_column, how="left")
            self.df.drop_duplicates(inplace=True)
            logging.info(f"Merged {file_name}, resulting shape: {self.df.shape}")

            # Drop irrelevant columns based on correlation
            self.drop_low_correlation_features()

        except Exception as e:
            logging.error(f"Error processing file {file_name}: {e}")

    def drop_low_correlation_features(self, low_threshold=0.1, high_threshold=0.9):
        """
        Drop numerical columns that have low correlation with the target column or high correlation with other features.
        """
        try:
            if self.target_column not in self.df.columns:
                logging.warning(
                    f"Target column '{self.target_column}' not found. Skipping correlation filter."
                )
                return

            # Filter numerical columns only
            numeric_df = self.df.select_dtypes(include=["number"])
            if self.target_column not in numeric_df.columns:
                logging.warning(
                    "Target column is not numerical. Skipping correlation filter."
                )
                return

            # Drop features with low correlation to the target
            correlation = numeric_df.corr()[self.target_column].abs()
            low_corr_columns = correlation[correlation < low_threshold].index.tolist()

            # Ensure target column is not in the list
            if self.target_column in low_corr_columns:
                low_corr_columns.remove(self.target_column)

            # Drop features with high multicollinearity
            high_corr_columns = set()
            for col in numeric_df.columns:
                if col != self.target_column:
                    other_corr = numeric_df.corr()[col].abs()
                    high_corr_features = other_corr[other_corr > high_threshold].index
                    for feature in high_corr_features:
                        if feature != col and feature != self.target_column:
                            high_corr_columns.add(feature)

            # Combine columns to drop
            columns_to_drop = set(low_corr_columns) | high_corr_columns

            # Drop the identified columns
            self.df.drop(columns=columns_to_drop, inplace=True)
            logging.info(
                f"Dropped columns with low correlation < {low_threshold} and high correlation > {high_threshold}, resulting shape: {self.df.shape}"
            )
        except Exception as e:
            logging.error(f"Error dropping low correlation features: {e}")

    def show_missing_value_stats(self):
        """
        Display and log missing value statistics.
        """
        try:
            missing_values = self.df.isnull().sum()
            missing_percentage = (missing_values / len(self.df)) * 100
            missing_stats = pd.DataFrame(
                {"Missing Values": missing_values, "Percentage": missing_percentage}
            ).sort_values(by="Missing Values", ascending=False)
            logging.info(f"Missing value statistics:\n{missing_stats}")
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

    def run_pipeline(self):
        """
        Run the full pipeline: load main data, sequentially merge files, and analyze.
        """
        self.load_main_data()

        # Process each additional file step-by-step
        for name, path in list(self.data_files.items())[1:]:
            self.preprocess_and_merge_file(name, path)

        logging.info("Pipeline execution completed.")
        self.show_missing_value_stats()
        self.exploratory_analysis()


# Example Usage
if __name__ == "__main__":
    data_files = {
        "Curated_Data": "../DataSet/5.LifeStageData-CompoundsCurated.csv",
        "MACCS": "../DataSet/6.MACCS_fingerprints.csv",
        "Morgan": "../DataSet/6.Morgan_fingerprints.csv",
        "AtomPair": "../DataSet/6.AtomPair_fingerprints.csv",
        "Physicochemical_PubChem": "../DataSet/5.LifeStageData-PhysicoChemicalProperties.csv",
        "Descriptors": "../DataSet/6.LifeStageData-Descriptors.csv",
        "CDK": "../DataSet/6.LifeStageData-CDK_Descriptors.csv",
        "RDKit": "../DataSet/6.RDKitFP_fingerprints.csv",
        "Torsion": "../DataSet/6.Torsion_fingerprints.csv",
    }

    pipeline = DataPipeline(
        data_files=data_files, target_column="Conc 1 Mean (Standardized)"
    )
    pipeline.run_pipeline()

    # Save the final processed dataset
    pipeline.df.to_csv("processed_data.csv", index=False)
    print("Pipeline execution finished. Processed data saved to 'processed_data.csv'.")
