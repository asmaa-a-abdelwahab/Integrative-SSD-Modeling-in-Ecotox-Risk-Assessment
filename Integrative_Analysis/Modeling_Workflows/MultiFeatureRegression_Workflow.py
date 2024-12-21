import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from joblib import dump, load
import matplotlib.pyplot as plt
from rdkit import Chem


# Configure logging
logging.basicConfig(
    filename="model_training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class MultiFeatureRegression:
    def __init__(self, data_files, target_column, save_path, id_column="SMILES"):
        self.data_files = data_files
        self.target_column = target_column
        self.save_path = save_path
        self.id_column = id_column
        self.df = None
        self.best_model = None
        self.best_features = None
        logging.info("Initialized MultiFeatureRegression class.")

    def load_and_merge_data(self):
        """
        Load and merge each dataset with the curated data. Handle specific cases for SMILES columns.
        """
        try:
            # Load curated data
            curated_data_path = self.data_files.pop("Curated_Data", None)
            if not curated_data_path:
                raise ValueError("Curated data file path is missing in data_files.")

            curated_df = pd.read_csv(curated_data_path)
            logging.info(f"Loaded curated data with shape {curated_df.shape}.")

            # Ensure id_column exists and drop duplicates
            if "Cleaned_SMILES" in curated_df.columns:
                if "SMILES" in curated_df.columns:
                    curated_df.drop(columns=["SMILES"], inplace=True)
                curated_df.rename(columns={"Cleaned_SMILES": "SMILES"}, inplace=True)
            logging.info(
                f"Curated data after renaming and deduplication: {curated_df.shape}"
            )

            merged_dataframes = []
            skipped_files = []

            # Process each file
            for name, path in self.data_files.items():
                try:
                    # Load dataset
                    logging.info(f"Processing file '{name}'...")
                    df = pd.read_csv(path)

                    # Handle specific case for SMILES
                    if any(substring in name for substring in ["OPERA", "Padel"]):
                        # Read the SMILES from a text file
                        with open(
                            "../DataSet/5.LifeStageData-Cleaned_SMILES.smiles",
                            "r",
                        ) as f:
                            smiles_list = [line.strip() for line in f.readlines()]

                        # Check if the number of SMILES matches the DataFrame rows
                        if len(smiles_list) != len(df):
                            raise ValueError(
                                f"Mismatch between SMILES list and rows in '{name}'."
                            )

                        # Append SMILES column to the DataFrame
                        df["SMILES"] = smiles_list
                        logging.info(f"Appended SMILES column to '{name}'.")
                    else:
                        # Handle Cleaned_SMILES if present
                        if "Cleaned_SMILES" in df.columns:
                            if "SMILES" in df.columns:
                                df.drop(columns=["SMILES"], inplace=True)
                            df.rename(
                                columns={"Cleaned_SMILES": "SMILES"}, inplace=True
                            )

                    # Merge with curated data
                    merged_df = pd.merge(curated_df, df, on=self.id_column, how="left")
                    merged_df.drop_duplicates(inplace=True)
                    logging.info(f"Merged dataset '{name}' shape: {merged_df.shape}")

                    # Drop columns with all missing values or constant values
                    merged_df.dropna(axis=1, how="all", inplace=True)
                    merged_df = merged_df.loc[:, merged_df.nunique() > 1]
                    logging.info(
                        f"After cleaning, dataset '{name}' shape: {merged_df.shape}"
                    )
                    # Calculate the threshold for missing values (70% of rows)
                    threshold = int(0.9 * len(merged_df))

                    # Drop columns with more than 70% missing values
                    merged_df = merged_df.loc[:, merged_df.isnull().sum() <= threshold]

                    logging.info(
                        f"Shape after dropping columns with >70% missing values: {merged_df.shape}"
                    )

                    # Filter rows based on the minimum non-NaN threshold
                    min_non_nan = int(0.9 * len(merged_df.columns))
                    merged_df = merged_df.dropna(thresh=min_non_nan)

                    # Save the preprocessed file for review
                    merged_df.to_csv(f"../DataSet/{name}_merged.csv", index=False)
                    merged_dataframes.append(merged_df)

                except Exception as e:
                    logging.error(f"Error processing file '{name}': {e}")
                    skipped_files.append(name)

            # Ensure SMILES column exists in all DataFrames and then join them
            if not merged_dataframes:
                raise ValueError("No valid datasets were processed successfully.")

            # Start with the first DataFrame as the base
            base_df = merged_dataframes[0]

            # Iteratively join the remaining DataFrames on the SMILES column
            for df in merged_dataframes[1:]:
                base_df = pd.merge(
                    base_df, df, on="SMILES", how="left"
                )  # Use 'outer' to include all SMILES

            logging.info(f"Final joined dataset shape: {base_df.shape}")

            # Save the final filtered DataFrame
            base_df.to_csv("../DataSet/Final_Joined.csv", index=False)
            logging.info(
                f"Filtered joined dataset saved to '../DataSet/Final_Joined.csv'"
            )

            # Log skipped files
            if skipped_files:
                logging.warning(f"Skipped files: {skipped_files}")

        except Exception as e:
            logging.error(f"Error in load_and_merge_data: {e}")
            raise

    def show_missing_value_stats(self, df):
        # Calculate total missing values per column
        total_missing = df.isnull().sum()

        # Calculate percentage of missing values per column
        missing_percentage = (df.isnull().mean() * 100).round(2)

        # Create a DataFrame to display statistics
        missing_stats = pd.DataFrame(
            {
                "Total Missing": total_missing,
                "Missing Percentage (%)": missing_percentage,
            }
        ).sort_values(by="Missing Percentage (%)", ascending=False)

        # Display columns with missing values
        missing_columns = missing_stats[missing_stats["Total Missing"] > 0]
        print("Missing Value Statistics:")
        print(missing_columns)

        return missing_columns

    def preprocess_data(self):
        try:
            self.df.dropna(inplace=True)
            X = self.df.drop(columns=[self.target_column, self.id_column])
            y = self.df[self.target_column]
            logging.info("Data preprocessing completed successfully.")
            return X, y
        except Exception as e:
            logging.error(f"Error in preprocess_data: {e}")
            raise

    def reduce_features(self, X, y, num_features=100):
        try:
            logging.info("Starting feature reduction...")
            selector = SelectKBest(
                score_func=f_regression, k=min(num_features, X.shape[1])
            )
            X_new = selector.fit_transform(X, y)
            self.best_features = X.columns[selector.get_support()]
            logging.info(f"Selected top {len(self.best_features)} features.")
            return X_new, self.best_features
        except Exception as e:
            logging.error(f"Error in reduce_features: {e}")
            raise

    def apply_pca(self, X, n_components=100):
        try:
            logging.info("Starting PCA...")
            pca = PCA(n_components=min(n_components, X.shape[1]))
            X_pca = pca.fit_transform(X)
            explained_variance = np.sum(pca.explained_variance_ratio_)
            logging.info(
                f"PCA reduced features to {X_pca.shape[1]} dimensions with {explained_variance:.2%} variance explained."
            )
            return X_pca
        except Exception as e:
            logging.error(f"Error in apply_pca: {e}")
            raise

    def train_models(self, X, y):
        try:
            logging.info("Starting model training...")
            models = {
                "LinearRegression": LinearRegression(),
                "RandomForest": RandomForestRegressor(),
                "GradientBoosting": GradientBoostingRegressor(),
                "SupportVectorRegression": SVR(),
                "XGBoost": XGBRegressor(),
                "LightGBM": LGBMRegressor(),
                "CatBoost": CatBoostRegressor(verbose=0),
            }
            param_grids = {
                "RandomForest": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20],
                },
                "GradientBoosting": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                },
                "SupportVectorRegression": {
                    "C": [0.1, 1, 10],
                    "kernel": ["linear", "rbf"],
                },
                "XGBoost": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                },
                "LightGBM": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                },
                "CatBoost": {
                    "iterations": [100, 200, 300],
                    "learning_rate": [0.01, 0.1, 0.2],
                },
            }

            best_score = float("inf")
            model_scores = {}
            for model_name, model in models.items():
                if model_name in param_grids:
                    grid = GridSearchCV(
                        model,
                        param_grids[model_name],
                        cv=5,
                        scoring="neg_mean_squared_error",
                    )
                    grid.fit(X, y)
                    score = -grid.best_score_
                    model_scores[model_name] = score
                    logging.info(f"{model_name} Best Score: {score}")
                    if score < best_score:
                        best_score = score
                        self.best_model = grid.best_estimator_
                else:
                    scores = cross_val_score(
                        model, X, y, cv=5, scoring="neg_mean_squared_error"
                    )
                    score = -np.mean(scores)
                    model_scores[model_name] = score
                    logging.info(f"{model_name} Cross-validated Score: {score}")
                    if score < best_score:
                        best_score = score
                        self.best_model = model
            logging.info(f"Training completed. Best model: {self.best_model}")
            self.plot_model_comparison(model_scores)
        except Exception as e:
            logging.error(f"Error in train_models: {e}")
            raise

    def evaluate_model(self, X_test, y_test):
        try:
            logging.info("Evaluating the best model...")
            y_pred = self.best_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            logging.info(f"Evaluation results - MSE: {mse}, R²: {r2}")
            self.plot_predictions(y_test, y_pred)
            self.plot_residuals(y_test, y_pred)
            return mse, r2
        except Exception as e:
            logging.error(f"Error in evaluate_model: {e}")
            raise

    def save_model(self):
        try:
            if self.best_model:
                dump(self.best_model, self.save_path)
                logging.info(f"Model saved to {self.save_path}")
        except Exception as e:
            logging.error(f"Error in save_model: {e}")
            raise

    def plot_predictions(self, y_test, y_pred):
        try:
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test, y_pred, alpha=0.7, edgecolor="k")
            plt.plot(
                [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2
            )
            plt.title("Actual vs Predicted Values")
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.grid(True)
            plt.show()
            logging.info("Generated predictions plot.")
        except Exception as e:
            logging.error(f"Error in plot_predictions: {e}")
            raise

    def plot_residuals(self, y_test, y_pred):
        try:
            residuals = y_test - y_pred
            plt.figure(figsize=(8, 6))
            plt.scatter(y_pred, residuals, alpha=0.7, edgecolor="k")
            plt.axhline(y=0, color="r", linestyle="--", lw=2)
            plt.title("Residual Plot")
            plt.xlabel("Predicted Values")
            plt.ylabel("Residuals")
            plt.grid(True)
            plt.show()
            logging.info("Generated residuals plot.")
        except Exception as e:
            logging.error(f"Error in plot_residuals: {e}")
            raise

    def plot_model_comparison(self, model_scores):
        try:
            model_names = list(model_scores.keys())
            scores = list(model_scores.values())
            plt.figure(figsize=(10, 6))
            plt.barh(model_names, scores, color="skyblue", edgecolor="k")
            plt.title("Model Performance Comparison")
            plt.xlabel("Mean Squared Error")
            plt.gca().invert_yaxis()
            plt.grid(axis="x", linestyle="--", alpha=0.7)
            plt.show()
            logging.info("Generated model comparison plot.")
        except Exception as e:
            logging.error(f"Error in plot_model_comparison: {e}")
            raise

    def run(self, num_features=100, use_pca=False):
        """
        Run the complete workflow.
        """
        try:
            logging.info("Workflow started.")
            self.load_and_merge_data()

            # Split the data for training and testing
            X = self.df.drop(columns=[self.target_column, self.id_column])
            y = self.df[self.target_column]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            self.train_models(X_train, y_train)
            self.evaluate_model(X_test, y_test)
            self.save_model()
            logging.info("Workflow completed successfully.")
        except Exception as e:
            logging.error(f"Error in run: {e}")
            raise

    def predict(self, new_data_path, num_features=100, use_pca=False):
        """
        Predict target values for new data using the best-performing model.

        Parameters:
        - new_data_path: Path to the CSV file containing new data.
        - num_features: Number of features to use for prediction (if feature selection was applied).
        - use_pca: Whether PCA was applied during training.

        Returns:
        - predictions: Predicted values for the new data.
        """
        if not self.best_model:
            # Load the saved model
            self.best_model = load(self.save_path)
            print("Loaded the best model from disk.")

        # Load new data
        new_data = pd.read_csv(new_data_path)
        new_data.dropna(inplace=True)

        # Preprocess new data
        X_new = new_data.drop(columns=[self.id_column], errors="ignore")

        if use_pca:
            # Apply PCA to match the training setup
            pca = PCA(n_components=min(num_features, X_new.shape[1]))
            X_new = pca.fit_transform(X_new)
            print("Applied PCA to new data.")
        elif self.best_features is not None:
            # Select the same features as during training
            X_new = X_new[self.best_features]
            print("Selected the same features as during training.")

        # Make predictions
        predictions = self.best_model.predict(X_new)
        print("Predictions completed.")

        return predictions


if __name__ == "__main__":

    def main():
        data_files = {
            "Curated_Data": "../DataSet/5.LifeStageData-CompoundsCurated.csv",
            "Physicochemical_OPERA": "../DataSet/6.OPERA_predictions.csv",
            "Padel_Fingerprints": "../DataSet/6.LifeStageData-CompoundsCurated_Unique_PadelFP.csv",
            "Padel_Descriptors": "../DataSet/6.LifeStageData-CompoundsCurated_Unique_PadelDesc.csv",
            "Physicochemical_PubChem": "../DataSet/5.LifeStageData-PhysicoChemicalProperties.csv",
            "Descriptors": "../DataSet/6.LifeStageData-Descriptors.csv",
            "MACCS": "../DataSet/6.MACCS_fingerprints.csv",
            "Morgan": "../DataSet/6.Morgan_fingerprints.csv",
            "RDKit": "../DataSet/6.RDKitFP_fingerprints.csv",
            "AtomPair": "../DataSet/6.AtomPair_fingerprints.csv",
            "Torsion": "../DataSet/6.Torsion_fingerprints.csv",
        }
        target_column = "Conc 1 Mean (Standardized)"
        save_path = "best_model.joblib"

        # Create the model instance
        model = MultiFeatureRegression(data_files, target_column, save_path)

        # Run the workflow
        model.run(num_features=100)

        # Make predictions on new data
        predictions = model.predict(new_data_path="new_data.csv", num_features=100)
        print(predictions)

    main()


"""
data_files = {
    "Physicochemical": "path_to_physicochemical_properties.csv",
    "MACCS": "path_to_MACCS_fingerprints.csv",
    "Morgan": "path_to_Morgan_fingerprints.csv",

}
target_column = "Conc_Mean"
save_path = "../Models/best_model.joblib"

model = MultiFeatureRegression(data_files, target_column, save_path)
model.run(num_features=100)

predictions = model.predict(new_data_path="new_data.csv", num_features=100)
print(predictions)

"""
