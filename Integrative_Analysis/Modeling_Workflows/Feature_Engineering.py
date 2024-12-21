import os
import logging
import dask.dataframe as dd
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import SGDRegressor
from sklearn.decomposition import IncrementalPCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import umap.umap_ as umap
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

matplotlib.use("Agg")  # Use a non-interactive backend


class FeatureEngineering:
    def __init__(self, file_path, target_column):
        """
        Initialize the FeatureEngineering class.
        """
        self.file_path = file_path
        self.output_dir = os.path.dirname(file_path)
        self.target_column = target_column

        # Setup logging
        log_file = os.path.join(self.output_dir, "feature_engineering.log")
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filemode="w",
        )
        logging.info("FeatureEngineering initialized.")

        # Load data using Dask
        self.data = dd.read_csv(file_path)
        columns_to_exclude = ["SMILES", "ID", "Activity_Cluster", "Index"]
        self.data = self.data.drop(columns=columns_to_exclude, errors="ignore")
        self.data = self.data.reset_index(drop=True)
        self.data = self.data.dropna(how="any")
        self.data = self.data.reset_index(drop=True)
        self.features = self.data.drop(columns=[target_column]).columns

    def summarize_data(self):
        """Summarize the dataset and display missing value counts."""
        logging.info("Summarizing dataset.")
        dataset_summary = self.data.describe()
        missing_values = self.data.isnull().sum()

        summary_path = os.path.join(self.output_dir, "data_summary.txt")
        with open(summary_path, "w") as f:
            f.write("Dataset Summary:\n")
            f.write(dataset_summary.to_string())
            f.write("\n\nMissing Value Counts:\n")
            f.write(missing_values.to_string())
        logging.info(f"Dataset summary saved at {summary_path}")

        # Target Distribution Plot with Outlier Filtering
        filtered_data = self.data[self.target_column].clip(
            upper=self.data[self.target_column].quantile(0.99)
        )  # Limit to 99th percentile

        target_dist_plot_path = os.path.join(self.output_dir, "target_distribution.png")
        plt.figure(figsize=(12, 8))
        sns.histplot(
            filtered_data,
            bins=100,
            kde=True,
            color="blue",
            edgecolor="black",
        )
        plt.xscale("log")  # Use log scale for x-axis
        plt.grid(axis="both", linestyle="--", alpha=0.6)
        plt.title(
            f"Distribution of {self.target_column} (Filtered, Log-Scale)", fontsize=16
        )
        plt.xlabel(self.target_column, fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.tight_layout()
        plt.savefig(target_dist_plot_path, dpi=300)
        plt.close()
        logging.info(f"Target distribution plot saved at {target_dist_plot_path}")

    def encode_categorical_features(self, encoding_method="one-hot"):
        """Encode categorical features."""
        logging.info(f"Encoding categorical features using {encoding_method} method.")
        categorical_features = self.data.select_dtypes(include=["object"]).columns

        if categorical_features.empty:
            logging.info("No categorical features found.")
            return

        self.data = self.data.categorize(columns=categorical_features)

        if encoding_method == "one-hot":
            encoder = OneHotEncoder(sparse_output=False, drop="first")
            data_pd = self.data.compute()
            encoded_features = encoder.fit_transform(data_pd[categorical_features])
            encoded_df = pd.DataFrame(
                encoded_features,
                columns=encoder.get_feature_names_out(categorical_features),
                index=data_pd.index,
            )
            self.data = dd.from_pandas(
                pd.concat(
                    [data_pd.drop(columns=categorical_features), encoded_df], axis=1
                ),
                npartitions=1,
            )
        elif encoding_method == "label":
            encoder = LabelEncoder()
            data_pd = self.data.compute()
            for feature in categorical_features:
                data_pd[feature] = encoder.fit_transform(data_pd[feature])
            self.data = dd.from_pandas(data_pd, npartitions=1)

    def feature_importance(self, n_features=10):
        """Compute and visualize feature importance."""
        logging.info("Calculating feature importance.")
        data_pd = self.data.compute()

        numeric_data = data_pd.select_dtypes(include=["number"]).dropna()
        X = numeric_data.drop(columns=[self.target_column])
        y = numeric_data[self.target_column]

        model = SGDRegressor(max_iter=1000, tol=1e-3)
        model.fit(X, y)

        importance = np.abs(model.coef_)
        feature_importance_df = pd.DataFrame(
            {"Feature": X.columns, "Importance": importance}
        ).sort_values(by="Importance", ascending=False)

        output_path = os.path.join(self.output_dir, "feature_importance.png")
        plt.figure(figsize=(14, 8))
        sns.barplot(
            x="Importance",
            y="Feature",
            data=feature_importance_df.head(n_features),
            palette="viridis",
            hue="Feature",
            legend=False,
        )
        plt.title("Top Features by Importance", fontsize=16)
        plt.xlabel("Feature Importance", fontsize=14)
        plt.ylabel("Features", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        logging.info(f"Feature importance plot saved at {output_path}")

        return feature_importance_df.head(n_features)["Feature"].tolist()

    def dimensionality_reduction(self, features, method="PCA", n_components=10):
        """Perform dimensionality reduction using PCA or UMAP."""
        logging.info(f"Performing dimensionality reduction using {method}.")
        data_pd = self.data.compute()
        X = data_pd[features]

        # Handle missing values by imputing or dropping
        if X.isnull().values.any():
            logging.warning("Missing values detected. Imputing with mean.")
            X = X.fillna(X.mean())  # Replace NaN with column mean

        if method == "PCA":
            reducer = IncrementalPCA(n_components=n_components)
        elif method == "UMAP":
            reducer = umap.UMAP(n_components=n_components, init="random")
        else:
            raise ValueError("Unsupported dimensionality reduction method.")

        reduced_features = reducer.fit_transform(X)
        reduced_data = pd.DataFrame(
            reduced_features,
            columns=[f"Component_{i + 1}" for i in range(n_components)],
        )

        # Save variance plot for PCA
        if method == "PCA" and hasattr(reducer, "explained_variance_ratio_"):
            explained_variance_ratio = reducer.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)

            variance_output = os.path.join(
                self.output_dir, "pca_cumulative_variance.png"
            )
            plt.figure(figsize=(10, 6))
            plt.plot(
                range(1, len(cumulative_variance) + 1),
                cumulative_variance,
                marker="o",
                linestyle="--",
                color="b",
            )
            plt.axhline(
                y=0.90, color="r", linestyle="--", label="90% Explained Variance"
            )
            plt.axhline(
                y=0.95, color="g", linestyle="--", label="95% Explained Variance"
            )
            plt.xlabel("Number of Components", fontsize=14)
            plt.ylabel("Cumulative Explained Variance", fontsize=14)
            plt.title("PCA Cumulative Explained Variance", fontsize=16)
            plt.legend()
            plt.tight_layout()
            plt.savefig(variance_output, dpi=300)
            plt.close()
            logging.info(f"PCA cumulative variance plot saved at {variance_output}")

        return reduced_data

    def shap_analysis(self, top_features):
        """Perform SHAP analysis."""
        logging.info("Performing SHAP analysis.")
        data_pd = self.data.compute()

        # Ensure features match the training data
        X = data_pd[top_features]
        y = data_pd[self.target_column]

        # Replace NaNs with column means to ensure no missing values
        if X.isnull().values.any():
            logging.warning("Handling missing values for SHAP analysis.")
            X = X.fillna(X.mean())

        # Train the model
        model = RandomForestRegressor(random_state=42)
        model.fit(X, y)

        # Initialize SHAP explainer
        explainer = shap.Explainer(model, X)
        try:
            shap_values = explainer(X)
        except shap.utils._exceptions.ExplainerError as e:
            logging.warning(
                f"Additivity check failed: {e}. Retrying with check_additivity=False."
            )
            shap_values = explainer(X, check_additivity=False)

        # Save SHAP summary plot
        shap_summary_path = os.path.join(self.output_dir, "shap_summary.png")
        shap.summary_plot(shap_values, X, show=False)
        plt.tight_layout()
        plt.savefig(shap_summary_path, dpi=300)
        plt.close()
        logging.info(f"SHAP summary plot saved at {shap_summary_path}")

    def process_pipeline(self, n_features=10):
        """Run the feature engineering pipeline."""
        logging.info("Starting feature engineering pipeline.")
        self.summarize_data()
        self.encode_categorical_features()
        top_features = self.feature_importance(n_features)
        self.dimensionality_reduction(top_features, method="PCA")
        self.dimensionality_reduction(top_features, method="UMAP")
        self.shap_analysis(top_features)
        logging.info("Feature engineering pipeline completed.")


# Example Usage
if __name__ == "__main__":
    dataset_path = (
        "./Processed_Combinations/TEST/Curated_Data_TEST-Descriptors_processed.csv"
    )
    target_column = "Conc 1 Mean (Standardized)"
    feature_engineering = FeatureEngineering(
        file_path=dataset_path, target_column=target_column
    )
    feature_engineering.process_pipeline(n_features=10)


# import os
# import logging
# import pandas as pd
# import numpy as np
# from sklearn.feature_selection import RFE
# from sklearn.linear_model import SGDRegressor
# from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# from sklearn.decomposition import IncrementalPCA
# import matplotlib.pyplot as plt
# import seaborn as sns
# from concurrent.futures import ThreadPoolExecutor
# import matplotlib

# matplotlib.use("Agg")  # Use a non-interactive backend


# class FeatureEngineering:
#     def __init__(self, file_path, target_column):
#         """
#         Initialize the FeatureEngineering class.

#         Parameters:
#         - file_path (str): Path to the dataset file.
#         - target_column (str): The name of the target variable column.
#         """
#         self.file_path = file_path
#         self.output_dir = os.path.dirname(file_path)
#         self.target_column = target_column

#         # Setup logging
#         log_file = os.path.join(self.output_dir, "feature_engineering.log")
#         logging.basicConfig(
#             filename=log_file,
#             level=logging.INFO,
#             format="%(asctime)s - %(levelname)s - %(message)s",
#             filemode="w",
#         )
#         logging.info("FeatureEngineering initialized.")

#         # Load data
#         self.data = pd.read_csv(file_path)
#         columns_to_exclude = ["SMILES", "ID", "Activity_Cluster", "Index"]
#         self.data = self.data.drop(columns=columns_to_exclude, errors="ignore")
#         self.data = self.data.dropna()
#         self.features = self.data.drop(columns=[target_column]).columns
#         self.target = self.data[target_column]

#     def encode_categorical_features(self, encoding_method="one-hot"):
#         """Encode categorical features."""
#         logging.info(f"Encoding categorical features using {encoding_method} method.")
#         categorical_features = self.data.select_dtypes(include=["object"]).columns

#         try:
#             if encoding_method == "one-hot":
#                 encoder = OneHotEncoder(sparse_output=False, drop="first")
#                 encoded_features = encoder.fit_transform(
#                     self.data[categorical_features]
#                 )
#                 encoded_df = pd.DataFrame(
#                     encoded_features,
#                     columns=encoder.get_feature_names_out(categorical_features),
#                     index=self.data.index,
#                 )
#                 # Concatenate encoded features back to the original data
#                 self.data = pd.concat(
#                     [self.data.drop(columns=categorical_features), encoded_df],
#                     axis=1,
#                 )
#             elif encoding_method == "label":
#                 encoder = LabelEncoder()
#                 for feature in categorical_features:
#                     self.data[feature] = encoder.fit_transform(self.data[feature])
#             else:
#                 raise ValueError(
#                     "Unsupported encoding method. Use 'one-hot' or 'label'."
#                 )
#             logging.info("Categorical features encoded successfully.")
#         except Exception as e:
#             logging.error(f"Error encoding categorical features: {e}")

#     def summarize_data(self):
#         """Summarize the dataset and display missing value counts."""
#         logging.info("Summarizing dataset.")
#         dataset_summary = self.data.describe()
#         missing_values = self.data.isnull().sum()

#         summary_path = os.path.join(self.output_dir, "data_summary.txt")
#         with open(summary_path, "w") as f:
#             f.write("Dataset Summary:\n")
#             f.write(dataset_summary.to_string())
#             f.write("\n\nMissing Value Counts:\n")
#             f.write(missing_values.to_string())
#         logging.info(f"Dataset summary saved at {summary_path}")

#         # Target Distribution Plot
#         sampled_target = self.target.sample(frac=0.1, random_state=42)
#         target_dist_plot_path = os.path.join(self.output_dir, "target_distribution.png")
#         plt.figure(figsize=(10, 6))
#         sns.histplot(sampled_target, bins=50, kde=True, color="blue", edgecolor="black")
#         plt.title(f"Distribution of {self.target_column}", fontsize=16)
#         plt.xlabel(self.target_column, fontsize=14)
#         plt.ylabel("Frequency", fontsize=14)
#         plt.tight_layout()
#         plt.savefig(target_dist_plot_path, dpi=300)
#         plt.close()
#         logging.info(f"Target distribution plot saved at {target_dist_plot_path}")

#     def feature_importance(self, n_features=10):
#         """Compute and visualize feature importance using SGDRegressor."""
#         logging.info("Calculating feature importance.")

#         # Ensure all features are numeric
#         numeric_data = self.data.select_dtypes(include=["number"])
#         numeric_data = numeric_data.dropna()

#         # Separate features and target
#         X = numeric_data.drop(columns=[self.target_column])
#         y = numeric_data[self.target_column]

#         # Fit model
#         model = SGDRegressor(max_iter=1000, tol=1e-3)
#         model.partial_fit(X, y)

#         # Calculate feature importance
#         importance = np.abs(model.coef_)
#         feature_importance_df = pd.DataFrame(
#             {"Feature": X.columns, "Importance": importance}
#         ).sort_values(by="Importance", ascending=False)

#         # Feature Importance Plot
#         output_path = os.path.join(self.output_dir, "feature_importance.png")
#         plt.figure(figsize=(14, 8))
#         sns.barplot(
#             x="Importance",
#             y="Feature",
#             data=feature_importance_df.head(n_features),
#             palette="viridis",
#             hue="Feature",
#             dodge=False,
#             legend=False,  # Suppress warnings about `hue`
#         )
#         plt.title("Top Features by Importance", fontsize=16)
#         plt.xlabel("Feature Importance", fontsize=14)
#         plt.ylabel("Features", fontsize=14)
#         plt.tight_layout()
#         plt.savefig(output_path, dpi=300)
#         plt.close()
#         logging.info(f"Feature importance plot saved at {output_path}")

#         top_features = feature_importance_df.head(n_features)["Feature"].tolist()
#         logging.info(f"Top {n_features} features: {top_features}")
#         return top_features

#     def dimensionality_reduction(self, features=None, n_components=10):
#         """Perform dimensionality reduction with incremental PCA."""
#         logging.info("Performing dimensionality reduction.")
#         if features is None:
#             features = self.features

#         sampled_data = self.data[features].sample(frac=0.1, random_state=42)
#         reducer = IncrementalPCA(n_components=n_components)
#         reduced_features = reducer.fit_transform(sampled_data)
#         reduced_data = pd.DataFrame(
#             reduced_features,
#             columns=[f"Component_{i + 1}" for i in range(n_components)],
#         )

#         explained_variance_ratio = reducer.explained_variance_ratio_
#         cumulative_variance = np.cumsum(explained_variance_ratio)

#         variance_output = os.path.join(self.output_dir, "pca_cumulative_variance.png")
#         plt.figure(figsize=(10, 6))
#         plt.plot(
#             range(1, len(cumulative_variance) + 1),
#             cumulative_variance,
#             marker="o",
#             linestyle="--",
#             color="b",
#         )
#         plt.axhline(y=0.90, color="r", linestyle="--", label="90% Explained Variance")
#         plt.axhline(y=0.95, color="g", linestyle="--", label="95% Explained Variance")
#         plt.xlabel("Number of Components", fontsize=14)
#         plt.ylabel("Cumulative Explained Variance", fontsize=14)
#         plt.title("PCA Cumulative Explained Variance", fontsize=16)
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig(variance_output, dpi=300)
#         plt.close()
#         logging.info(f"PCA cumulative variance plot saved at {variance_output}")

#         return reduced_data

#     def process_pipeline(self, n_features=10):
#         """Automate the feature engineering pipeline."""
#         logging.info("Starting feature engineering pipeline.")
#         self.summarize_data()
#         self.encode_categorical_features()
#         top_features = self.feature_importance(n_features=n_features)
#         self.dimensionality_reduction(features=top_features)
#         logging.info("Feature engineering pipeline completed.")


# # Example Usage
# if __name__ == "__main__":
#     dataset_path = (
#         "./Processed_Combinations/TEST/Curated_Data_TEST-Descriptors_processed.csv"
#     )
#     target_column = "Conc 1 Mean (Standardized)"
#     feature_engineering = FeatureEngineering(
#         file_path=dataset_path, target_column=target_column
#     )
#     feature_engineering.process_pipeline(n_features=10)


# import os
# import logging
# import pandas as pd
# import numpy as np
# from sklearn.feature_selection import RFE
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import (
#     StandardScaler,
#     MinMaxScaler,
#     OneHotEncoder,
#     LabelEncoder,
# )
# from sklearn.decomposition import PCA
# import shap
# import umap
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import seaborn as sns


# class FeatureEngineering:
#     def __init__(self, file_path, target_column):
#         """
#         Initialize the FeatureEngineering class.

#         Parameters:
#         - file_path (str): Path to the dataset file.
#         - target_column (str): The name of the target variable column.
#         """
#         self.file_path = file_path
#         self.data = pd.read_csv(file_path)
#         self.output_dir = os.path.dirname(
#             file_path
#         )  # Save outputs in the dataset's directory
#         self.target_column = target_column

#         # Setup logging
#         log_file = os.path.join(self.output_dir, "feature_engineering.log")
#         logging.basicConfig(
#             filename=log_file,
#             level=logging.INFO,
#             format="%(asctime)s - %(levelname)s - %(message)s",
#             filemode="w",
#         )
#         logging.info("FeatureEngineering initialized.")

#         columns_to_exclude = ["SMILES", "ID", "Activity_Cluster", "Index"]
#         self.data = self.data.drop(columns=columns_to_exclude, errors="ignore")
#         self.data.dropna(how="any", inplace=True)
#         self.data.reset_index(drop=True, inplace=True)
#         self.features = self.data.drop(columns=[target_column]).columns
#         self.target = self.data[target_column]

#     def encode_categorical_features(self, encoding_method="one-hot"):
#         """Encode categorical features."""
#         logging.info(
#             "Encoding categorical features using {} method.".format(encoding_method)
#         )
#         categorical_features = self.data.select_dtypes(include=["object"]).columns

#         try:
#             if encoding_method == "one-hot":
#                 encoder = OneHotEncoder(sparse_output=False, drop="first")
#                 encoded_features = encoder.fit_transform(
#                     self.data[categorical_features]
#                 )
#                 encoded_df = pd.DataFrame(
#                     encoded_features,
#                     columns=encoder.get_feature_names_out(categorical_features),
#                 )
#                 self.data = pd.concat(
#                     [self.data.drop(columns=categorical_features), encoded_df], axis=1
#                 )
#             elif encoding_method == "label":
#                 encoder = LabelEncoder()
#                 for feature in categorical_features:
#                     self.data[feature] = encoder.fit_transform(self.data[feature])
#             else:
#                 raise ValueError(
#                     "Unsupported encoding method. Use 'one-hot' or 'label'."
#                 )
#             logging.info("Categorical features encoded successfully.")
#         except Exception as e:
#             logging.error("Error encoding categorical features: {}".format(e))

#         self.features = self.data.drop(columns=[self.target_column]).columns
#         return self.data

#     def summarize_data(self):
#         """Summarize the dataset and display missing value counts."""
#         logging.info("Summarizing dataset.")
#         dataset_summary = self.data.describe()
#         missing_values = self.data.isnull().sum()

#         # Save summary to a file
#         summary_path = os.path.join(self.output_dir, "data_summary.txt")
#         with open(summary_path, "w") as f:
#             f.write("Dataset Summary:\n")
#             f.write(dataset_summary.to_string())
#             f.write("\n\nMissing Value Counts:\n")
#             f.write(missing_values.to_string())
#         logging.info("Dataset summary saved at {}".format(summary_path))

#         # Distribution Plot
#         target_dist_plot_path = os.path.join(self.output_dir, "target_distribution.png")
#         plt.figure(figsize=(10, 6))
#         sns.histplot(
#             self.target,
#             bins=50,
#             color="blue",
#             kde=True,
#             edgecolor="black",
#             log_scale=(
#                 False,
#                 True,
#             ),  # Log scale y-axis for better distribution visibility
#         )
#         plt.title(f"Distribution of {self.target_column}", fontsize=16)
#         plt.xlabel(self.target_column, fontsize=14)
#         plt.ylabel("Frequency", fontsize=14)
#         plt.tight_layout()
#         plt.savefig(target_dist_plot_path, dpi=300)
#         plt.close()
#         logging.info(
#             "Target distribution plot saved at {}".format(target_dist_plot_path)
#         )

#     def feature_importance(self, n_features=10):
#         """Compute and visualize feature importance using Random Forest."""
#         logging.info("Calculating feature importance.")
#         model = RandomForestRegressor(random_state=42)
#         model.fit(self.data[self.features], self.target)

#         importance = model.feature_importances_
#         feature_importance_df = pd.DataFrame(
#             {"Feature": self.features, "Importance": importance}
#         ).sort_values(by="Importance", ascending=False)

#         # Enhanced Plot for Feature Importance
#         output_path = os.path.join(self.output_dir, "feature_importance.png")
#         plt.figure(figsize=(14, 8))
#         sns.barplot(
#             x="Importance",
#             y="Feature",
#             data=feature_importance_df.head(n_features),
#             palette="viridis",
#         )
#         plt.title("Top Features by Importance", fontsize=16)
#         plt.xlabel("Feature Importance", fontsize=14)
#         plt.ylabel("Features", fontsize=14)
#         plt.tight_layout()
#         plt.savefig(output_path, dpi=300)
#         plt.close()
#         logging.info("Feature importance plot saved at {}".format(output_path))

#         top_features = feature_importance_df.head(n_features)["Feature"].tolist()
#         logging.info("Top {} features: {}".format(n_features, top_features))
#         return top_features

#     def dimensionality_reduction(self, features=None, method="PCA", n_components=10):
#         """Perform dimensionality reduction with SHAP integration and PCA visualization."""
#         logging.info("Performing {} dimensionality reduction.".format(method))
#         if features is None:
#             features = self.features
#         clean_data = self.data[features].dropna()

#         # SHAP Analysis
#         model = RandomForestRegressor(random_state=42)
#         model.fit(clean_data, self.target)
#         explainer = shap.TreeExplainer(model)
#         shap_values = explainer.shap_values(clean_data)

#         shap_output_path = os.path.join(self.output_dir, "shap_summary_plot.png")
#         shap.summary_plot(shap_values, clean_data, show=False, plot_type="bar")
#         plt.tight_layout()
#         plt.savefig(shap_output_path, dpi=300)
#         plt.close()
#         logging.info("SHAP summary plot saved at {}".format(shap_output_path))

#         # PCA Visualization
#         if method == "PCA":
#             reducer = PCA(n_components=n_components, random_state=42)
#         elif method == "UMAP":
#             reducer = umap.UMAP(n_components=n_components, random_state=42)
#         else:
#             raise ValueError("Unsupported dimensionality reduction method.")

#         reduced_features = reducer.fit_transform(clean_data)
#         reduced_data = pd.DataFrame(
#             reduced_features,
#             columns=["Component_{}".format(i + 1) for i in range(n_components)],
#         )

#         if method == "PCA" and hasattr(reducer, "explained_variance_ratio_"):
#             explained_variance_ratio = reducer.explained_variance_ratio_
#             cumulative_variance = np.cumsum(explained_variance_ratio)

#             # Explained Variance Plot
#             variance_output = os.path.join(
#                 self.output_dir, "pca_cumulative_variance.png"
#             )
#             plt.figure(figsize=(10, 6))
#             plt.plot(
#                 range(1, len(cumulative_variance) + 1),
#                 cumulative_variance,
#                 marker="o",
#                 linestyle="--",
#                 color="b",
#             )
#             plt.axhline(
#                 y=0.90, color="r", linestyle="--", label="90% Explained Variance"
#             )
#             plt.axhline(
#                 y=0.95, color="g", linestyle="--", label="95% Explained Variance"
#             )
#             plt.xlabel("Number of Components", fontsize=14)
#             plt.ylabel("Cumulative Explained Variance", fontsize=14)
#             plt.title("PCA Cumulative Explained Variance", fontsize=16)
#             plt.legend()
#             plt.tight_layout()
#             plt.savefig(variance_output, dpi=300)
#             plt.close()
#             logging.info(
#                 "PCA Cumulative Explained Variance plot saved at {}".format(
#                     variance_output
#                 )
#             )

#         return reduced_data

#     def process_pipeline(self, n_features=10, test_size=0.2):
#         """Automate the feature engineering pipeline."""
#         logging.info("Starting feature engineering pipeline.")
#         self.summarize_data()
#         self.encode_categorical_features()
#         top_features = self.feature_importance(n_features=n_features)
#         self.dimensionality_reduction(features=top_features)
#         logging.info("Feature engineering pipeline completed.")


# # Example Usage
# if __name__ == "__main__":
#     dataset_path = (
#         "./Processed_Combinations/TEST/Curated_Data_TEST-Descriptors_processed.csv"
#     )
#     target_column = "Conc 1 Mean (Standardized)"
#     feature_engineering = FeatureEngineering(
#         file_path=dataset_path, target_column=target_column
#     )
#     feature_engineering.process_pipeline(n_features=10)


# import os
# import logging
# import pandas as pd
# import numpy as np
# from sklearn.feature_selection import RFE
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import (
#     StandardScaler,
#     MinMaxScaler,
#     OneHotEncoder,
#     LabelEncoder,
# )
# from sklearn.decomposition import PCA
# import shap
# import umap
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import seaborn as sns


# class FeatureEngineering:
#     def __init__(self, file_path, target_column):
#         """
#         Initialize the FeatureEngineering class.

#         Parameters:
#         - file_path (str): Path to the dataset file.
#         - target_column (str): The name of the target variable column.
#         """
#         self.file_path = file_path
#         self.data = pd.read_csv(file_path)
#         self.output_dir = os.path.dirname(
#             file_path
#         )  # Save figures and logs in the same directory as the dataset
#         self.target_column = target_column

#         # Setup logging
#         log_file = os.path.join(self.output_dir, "feature_engineering.log")
#         logging.basicConfig(
#             filename=log_file,
#             level=logging.INFO,
#             format="%(asctime)s - %(levelname)s - %(message)s",
#             filemode="w",
#         )
#         logging.info("FeatureEngineering initialized.")

#         columns_to_exclude = ["SMILES", "ID", "Activity_Cluster", "Index"]
#         self.data = self.data.drop(columns=columns_to_exclude, errors="ignore")
#         self.data.reset_index(drop=True, inplace=True)
#         self.data.dropna(how="any", inplace=True)
#         self.data.reset_index(drop=True, inplace=True)
#         self.features = self.data.drop(columns=[target_column]).columns
#         self.target = self.data[target_column]

#     def summarize_data(self):
#         """Summarize the dataset and display missing value counts."""
#         logging.info("Summarizing dataset.")
#         dataset_summary = self.data.describe()
#         missing_values = self.data.isnull().sum()
#         logging.info(f"Dataset Summary:\n{dataset_summary}")
#         logging.info(f"Missing Value Counts:\n{missing_values}")
#         print("Dataset Summary and Missing Value Counts logged.")

#     def encode_categorical_features(self, encoding_method="one-hot"):
#         """
#         Encode categorical features using the specified method.

#         Parameters:
#         - encoding_method (str): Encoding method ('one-hot' or 'label').

#         Returns:
#         - self.data (pd.DataFrame): Updated DataFrame with encoded features.
#         """
#         logging.info(f"Encoding categorical features using {encoding_method} method.")
#         categorical_features = self.data.select_dtypes(include=["object"]).columns

#         try:
#             if encoding_method == "one-hot":
#                 encoder = OneHotEncoder(sparse_output=False, drop="first")
#                 encoded_features = encoder.fit_transform(
#                     self.data[categorical_features]
#                 )
#                 encoded_df = pd.DataFrame(
#                     encoded_features,
#                     columns=encoder.get_feature_names_out(categorical_features),
#                 )
#                 self.data = pd.concat(
#                     [self.data.drop(columns=categorical_features), encoded_df], axis=1
#                 )
#             elif encoding_method == "label":
#                 encoder = LabelEncoder()
#                 for feature in categorical_features:
#                     self.data[feature] = encoder.fit_transform(self.data[feature])
#             else:
#                 raise ValueError(
#                     "Unsupported encoding method. Use 'one-hot' or 'label'."
#                 )
#             logging.info("Categorical features encoded successfully.")
#         except Exception as e:
#             logging.error(f"Error encoding categorical features: {e}")

#         self.features = self.data.drop(columns=[self.target_column]).columns
#         return self.data

#     def feature_importance(self, n_features=10):
#         """
#         Compute and visualize feature importance using Random Forest.

#         Parameters:
#         - n_features (int): Number of top features to display.

#         Returns:
#         - top_features (list): List of top feature names.
#         """
#         logging.info("Calculating feature importance.")
#         model = RandomForestRegressor(random_state=42)
#         model.fit(self.data[self.features], self.target)

#         importance = model.feature_importances_
#         feature_importance_df = pd.DataFrame(
#             {"Feature": self.features, "Importance": importance}
#         ).sort_values(by="Importance", ascending=False)

#         # Plot Feature Importance
#         plt.figure(figsize=(12, 8))
#         sns.barplot(
#             x="Importance",
#             y="Feature",
#             data=feature_importance_df.head(n_features),
#             palette="viridis",
#         )
#         plt.title("Top Features by Importance", fontsize=16)
#         plt.xlabel("Feature Importance", fontsize=14)
#         plt.ylabel("Features", fontsize=14)
#         plt.tight_layout()
#         output_path = os.path.join(self.output_dir, "feature_importance.png")
#         plt.savefig(output_path, dpi=300)
#         plt.close()
#         logging.info(f"Feature importance plot saved at {output_path}.")

#         top_features = feature_importance_df.head(n_features)["Feature"].tolist()
#         logging.info(f"Top {n_features} features: {top_features}")
#         return top_features

#     def dimensionality_reduction(self, features=None, method="PCA", n_components=10):
#         """
#         Perform dimensionality reduction with SHAP integration and visualizations.

#         Parameters:
#         - features (list): List of features to reduce. If None, all features are used.
#         - method (str): Dimensionality reduction method ('PCA', 'UMAP').
#         - n_components (int): Number of components to reduce to.

#         Returns:
#         - reduced_data (pd.DataFrame): DataFrame with reduced dimensions.
#         """
#         logging.info(
#             f"Performing {method} dimensionality reduction with {n_components} components."
#         )
#         if features is None:
#             features = self.features
#         clean_data = self.data[features].dropna()

#         model = RandomForestRegressor(random_state=42)
#         model.fit(clean_data, self.target)

#         explainer = shap.TreeExplainer(model)
#         shap_values = explainer.shap_values(clean_data)

#         shap.summary_plot(shap_values, clean_data, show=False, plot_type="bar")
#         output_path = os.path.join(self.output_dir, "shap_summary_plot.png")
#         plt.tight_layout()
#         plt.savefig(output_path, dpi=300)
#         plt.close()
#         logging.info(f"SHAP summary plot saved at {output_path}.")

#         if method == "PCA":
#             reducer = PCA(n_components=n_components, random_state=42)
#         elif method == "UMAP":
#             reducer = umap.UMAP(n_components=n_components, random_state=42)
#         else:
#             raise ValueError("Unsupported dimensionality reduction method.")

#         reduced_features = reducer.fit_transform(clean_data)
#         reduced_data = pd.DataFrame(
#             reduced_features, columns=[f"Component_{i+1}" for i in range(n_components)]
#         )

#         if method == "PCA" and hasattr(reducer, "explained_variance_ratio_"):
#             explained_variance_ratio = reducer.explained_variance_ratio_
#             plt.figure(figsize=(10, 6))
#             plt.bar(
#                 range(1, len(explained_variance_ratio) + 1),
#                 explained_variance_ratio,
#                 color="skyblue",
#             )
#             plt.title("PCA Components Explained Variance", fontsize=16)
#             plt.xlabel("Principal Components", fontsize=14)
#             plt.ylabel("Explained Variance Ratio", fontsize=14)
#             plt.xticks(range(1, len(explained_variance_ratio) + 1))
#             plt.tight_layout()
#             output_path = os.path.join(self.output_dir, "pca_explained_variance.png")
#             plt.savefig(output_path, dpi=300)
#             plt.close()
#             logging.info(f"PCA Explained Variance plot saved at {output_path}.")

#         logging.info("Dimensionality reduction completed.")
#         return reduced_data

#     def process_pipeline(self, n_features=10, n_components=10, test_size=0.2):
#         """
#         Automate the full feature engineering pipeline.

#         Parameters:
#         - n_features (int): Number of top features to select.
#         - n_components (int): Number of PCA components.
#         - test_size (float): Proportion of the dataset to include in the test split.

#         Returns:
#         - X_train, X_test, y_train, y_test: Train and test datasets.
#         """
#         logging.info("Starting feature engineering pipeline.")
#         self.summarize_data()
#         self.encode_categorical_features()
#         top_features = self.feature_importance(n_features=n_features)
#         self.dimensionality_reduction(
#             features=top_features, method="PCA", n_components=n_components
#         )
#         logging.info("Feature engineering pipeline completed.")


# # Example Usage
# if __name__ == "__main__":
#     dataset_path = "./Processed_Combinations/TEST/Curated_Data_TEST-Descriptors_processed.csv"  # Update with your dataset path
#     target_column = "Conc 1 Mean (Standardized)"  # Replace with your target column
#     feature_engineering = FeatureEngineering(
#         file_path=dataset_path, target_column=target_column
#     )
#     feature_engineering.process_pipeline(n_features=10, n_components=10, test_size=0.2)


# import pandas as pd
# import numpy as np
# from sklearn.feature_selection import RFE
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.decomposition import PCA
# import shap
# import umap
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import seaborn as sns


# class FeatureEngineering:
#     def __init__(self, data, target_column):
#         """
#         Initialize the FeatureEngineering class.

#         Parameters:
#         - data (pd.DataFrame): The input dataset.
#         - target_column (str): The name of the target variable column.
#         """
#         self.data = data
#         self.target_column = target_column
#         self.features = data.drop(columns=[target_column]).columns
#         self.target = data[target_column]

#     def visualize_data(self):
#         """
#         Generate visualizations for distributions, correlations, and feature importance.
#         """
#         # Distribution of Target Variable
#         plt.figure(figsize=(10, 6))
#         sns.histplot(self.target, kde=True, bins=30, color="blue")
#         plt.title(f"Distribution of {self.target_column}")
#         plt.xlabel(self.target_column)
#         plt.ylabel("Frequency")
#         plt.show()

#         # Correlation Heatmap
#         plt.figure(figsize=(12, 10))
#         corr_matrix = self.data.corr()
#         sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", cbar=True)
#         plt.title("Correlation Heatmap")
#         plt.show()

#         # Pairplot for Features and Target
#         top_features = self.features[
#             :5
#         ]  # Adjust the number of features to plot if needed
#         sns.pairplot(self.data[top_features.tolist() + [self.target_column]])
#         plt.show()

#     def feature_importance(self, n_features=10):
#         """
#         Compute and visualize feature importance using Random Forest.

#         Parameters:
#         - n_features (int): Number of top features to display.

#         Returns:
#         - top_features (list): List of top feature names.
#         """
#         model = RandomForestRegressor(random_state=42)
#         model.fit(self.data[self.features], self.target)

#         importance = model.feature_importances_
#         feature_importance_df = pd.DataFrame(
#             {"Feature": self.features, "Importance": importance}
#         ).sort_values(by="Importance", ascending=False)

#         # Plot Feature Importance
#         plt.figure(figsize=(10, 6))
#         sns.barplot(
#             x="Importance",
#             y="Feature",
#             data=feature_importance_df.head(n_features),
#             palette="viridis",
#         )
#         plt.title("Top Features by Importance")
#         plt.xlabel("Feature Importance")
#         plt.ylabel("Features")
#         plt.show()

#         top_features = feature_importance_df.head(n_features)["Feature"].tolist()
#         print(f"Top {n_features} features by importance: {top_features}")
#         return top_features

#     def recursive_feature_elimination(self, n_features=10):
#         """
#         Perform Recursive Feature Elimination (RFE) using Random Forest.

#         Parameters:
#         - n_features (int): Number of top features to select.

#         Returns:
#         - selected_features (list): List of selected feature names.
#         """
#         model = RandomForestRegressor(random_state=42)
#         rfe = RFE(model, n_features_to_select=n_features)
#         rfe.fit(self.data[self.features], self.target)

#         selected_features = self.features[rfe.support_].tolist()
#         print(f"Selected features using RFE: {selected_features}")
#         return selected_features

#     def shap_feature_selection(self, n_features=10):
#         """
#         Perform feature selection using SHAP values.

#         Parameters:
#         - n_features (int): Number of top features to select.

#         Returns:
#         - selected_features (list): List of selected feature names.
#         """
#         model = RandomForestRegressor(random_state=42)
#         model.fit(self.data[self.features], self.target)

#         explainer = shap.TreeExplainer(model)
#         shap_values = explainer.shap_values(self.data[self.features])

#         # Summary Plot
#         shap.summary_plot(shap_values, self.data[self.features], plot_type="bar")

#         mean_shap_values = np.abs(shap_values).mean(axis=0)
#         feature_importance = pd.DataFrame(
#             {"Feature": self.features, "Importance": mean_shap_values}
#         ).sort_values(by="Importance", ascending=False)

#         selected_features = feature_importance.head(n_features)["Feature"].tolist()
#         print(f"Selected features using SHAP: {selected_features}")
#         return selected_features

#     def normalize_data(self, features=None, method="z-score"):
#         """
#         Normalize or standardize the selected features.

#         Parameters:
#         - features (list): List of features to normalize. If None, all features are used.
#         - method (str): Normalization method ('z-score' or 'min-max').

#         Returns:
#         - normalized_data (pd.DataFrame): DataFrame with normalized features.
#         """
#         if features is None:
#             features = self.features

#         scaler = StandardScaler() if method == "z-score" else MinMaxScaler()
#         normalized_features = scaler.fit_transform(self.data[features])

#         normalized_data = pd.DataFrame(normalized_features, columns=features)
#         normalized_data[self.target_column] = self.target

#         print(f"Data normalized for features: {features} using {method} method.")
#         return normalized_data

#     def dimensionality_reduction(self, features=None, method="PCA", n_components=2):
#         """
#         Perform dimensionality reduction on selected features.

#         Parameters:
#         - features (list): List of features to reduce. If None, all features are used.
#         - method (str): Dimensionality reduction method ('PCA', 'UMAP').
#         - n_components (int): Number of components to reduce to.

#         Returns:
#         - reduced_data (pd.DataFrame): DataFrame with reduced dimensions.
#         """
#         if features is None:
#             features = self.features

#         if method == "PCA":
#             reducer = PCA(n_components=n_components, random_state=42)
#         elif method == "UMAP":
#             reducer = umap.UMAP(n_components=n_components, random_state=42)
#         else:
#             raise ValueError("Unsupported dimensionality reduction method.")

#         reduced_features = reducer.fit_transform(self.data[features])
#         reduced_data = pd.DataFrame(
#             reduced_features, columns=[f"Component_{i+1}" for i in range(n_components)]
#         )

#         print(
#             f"Data reduced for features: {features} to {n_components} components using {method}."
#         )
#         return reduced_data


# # Example Usage
# if __name__ == "__main__":
#     # Sample dataset
#     df = pd.read_csv(
#         "./Processed_Combinations/Curated_Data_TEST-Descriptors_processed.csv"
#     )

#     feature_engineering = FeatureEngineering(
#         data=df, target_column="Conc 1 Mean (Standardized)"
#     )

#     # Visualization
#     feature_engineering.visualize_data()

#     # Feature Selection
#     important_features = feature_engineering.feature_importance(n_features=50)

#     # Dimensionality Reduction
#     reduced_data = feature_engineering.dimensionality_reduction(
#         features=important_features, method="PCA", n_components=2
#     )


# import pandas as pd
# import numpy as np
# from sklearn.feature_selection import RFE
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.decomposition import PCA
# import shap
# import umap
# from sklearn.model_selection import train_test_split


# class FeatureEngineering:
#     def __init__(self, data, target_column):
#         """
#         Initialize the FeatureEngineering class.

#         Parameters:
#         - data (pd.DataFrame): The input dataset.
#         - target_column (str): The name of the target variable column.
#         """
#         self.data = data
#         self.target_column = target_column
#         self.features = data.drop(columns=[target_column]).columns
#         self.target = data[target_column]

#     def recursive_feature_elimination(self, n_features=10):
#         """
#         Perform Recursive Feature Elimination (RFE) using Random Forest.

#         Parameters:
#         - n_features (int): Number of top features to select.

#         Returns:
#         - selected_features (list): List of selected feature names.
#         """
#         model = RandomForestRegressor(random_state=42)
#         rfe = RFE(model, n_features_to_select=n_features)
#         rfe.fit(self.data[self.features], self.target)

#         selected_features = self.features[rfe.support_].tolist()
#         print(f"Selected features using RFE: {selected_features}")
#         return selected_features

#     def shap_feature_selection(self, n_features=10):
#         """
#         Perform feature selection using SHAP values.

#         Parameters:
#         - n_features (int): Number of top features to select.

#         Returns:
#         - selected_features (list): List of selected feature names.
#         """
#         model = RandomForestRegressor(random_state=42)
#         model.fit(self.data[self.features], self.target)

#         explainer = shap.TreeExplainer(model)
#         shap_values = explainer.shap_values(self.data[self.features])

#         shap.summary_plot(shap_values, self.data[self.features], plot_type="bar")

#         mean_shap_values = np.abs(shap_values).mean(axis=0)
#         feature_importance = pd.DataFrame(
#             {"Feature": self.features, "Importance": mean_shap_values}
#         ).sort_values(by="Importance", ascending=False)

#         selected_features = feature_importance.head(n_features)["Feature"].tolist()
#         print(f"Selected features using SHAP: {selected_features}")
#         return selected_features

#     def normalize_data(self, features=None, method="z-score"):
#         """
#         Normalize or standardize the selected features.

#         Parameters:
#         - features (list): List of features to normalize. If None, all features are used.
#         - method (str): Normalization method ('z-score' or 'min-max').

#         Returns:
#         - normalized_data (pd.DataFrame): DataFrame with normalized features.
#         """
#         if features is None:
#             features = self.features

#         scaler = StandardScaler() if method == "z-score" else MinMaxScaler()
#         normalized_features = scaler.fit_transform(self.data[features])

#         normalized_data = pd.DataFrame(normalized_features, columns=features)
#         normalized_data[self.target_column] = self.target

#         print(f"Data normalized for features: {features} using {method} method.")
#         return normalized_data

#     def dimensionality_reduction(self, features=None, method="PCA", n_components=2):
#         """
#         Perform dimensionality reduction on selected features.

#         Parameters:
#         - features (list): List of features to reduce. If None, all features are used.
#         - method (str): Dimensionality reduction method ('PCA', 'UMAP').
#         - n_components (int): Number of components to reduce to.

#         Returns:
#         - reduced_data (pd.DataFrame): DataFrame with reduced dimensions.
#         """
#         if features is None:
#             features = self.features

#         if method == "PCA":
#             reducer = PCA(n_components=n_components, random_state=42)
#         elif method == "UMAP":
#             reducer = umap.UMAP(n_components=n_components, random_state=42)
#         else:
#             raise ValueError("Unsupported dimensionality reduction method.")

#         reduced_features = reducer.fit_transform(self.data[features])
#         reduced_data = pd.DataFrame(
#             reduced_features, columns=[f"Component_{i+1}" for i in range(n_components)]
#         )

#         print(
#             f"Data reduced for features: {features} to {n_components} components using {method}."
#         )
#         return reduced_data


# # Example Usage
# if __name__ == "__main__":
#     # Step 1: Load the dataset
#     df = pd.DataFrame(
#         {
#             "Feature1": np.random.rand(100),
#             "Feature2": np.random.rand(100),
#             "Feature3": np.random.rand(100),
#             "Feature4": np.random.rand(100),
#             "Target": np.random.rand(100),
#         }
#     )

#     # Step 2: Initialize the FeatureEngineering class
#     feature_engineering = FeatureEngineering(data=df, target_column="Target")

#     # Step 3: Perform feature selection
#     selected_features = feature_engineering.recursive_feature_elimination(n_features=2)

#     # Step 4: Normalize the selected features
#     normalized_data = feature_engineering.normalize_data(
#         features=selected_features, method="z-score"
#     )

#     # Step 5: Perform dimensionality reduction
#     reduced_data = feature_engineering.dimensionality_reduction(
#         features=selected_features, method="PCA", n_components=2
#     )

#     # Step 6: Prepare data for modeling
#     X = normalized_data[selected_features]
#     y = normalized_data["Target"]

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     print("Training Data Shape:", X_train.shape)
#     print("Testing Data Shape:", X_test.shape)

#     # Reduced data can also be used for visualization or as input for simpler models
#     print("Reduced Data Shape:", reduced_data.shape)
