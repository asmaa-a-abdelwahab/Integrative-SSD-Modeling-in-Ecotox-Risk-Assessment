from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem, rdFingerprintGenerator, SDWriter
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
import requests
import logging
import os
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ChemicalCuration:
    """
    A class to perform chemical data curation, including duplicate removal, inorganics, organometallics, and mixtures,
    structure normalization, tautomer and isomer handling, stereochemistry verification, and validation.
    """

    def __init__(self, df):
        self.df = df
        self.cleaned_data = None
        self.df["Status"] = ""  # Initialize a Status column to track the curation steps
        self.inchikey_smiles_map = {}  # Initialize the attribute to hold InChIKey to SMILES mapping
        logging.info(
            "ChemicalCuration initialized with DataFrame of shape %s", df.shape
        )

    def remove_duplicates(self):
        """
        Remove duplicates from the mapping dictionary based on InChIKey and canonicalized Cleaned_SMILES.
        """
        try:
            if "Cleaned_SMILES" not in self.cleaned_data.columns:
                logging.error("'Cleaned_SMILES' column not found in the dataset")
                return

            logging.info(
                "Removing duplicates based on InChIKey and canonical Cleaned_SMILES..."
            )

            def update_status(idx, message):
                self.cleaned_data.at[idx, "Status"] += f"{message}; "

            # Function to canonicalize SMILES strings
            def canonicalize_smiles(smiles):
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        logging.error(
                            f"Invalid SMILES: '{smiles}' - RDKit failed to parse."
                        )
                        return None
                    return Chem.MolToSmiles(mol, canonical=True)
                except Exception as e:
                    logging.error(f"Failed to canonicalize SMILES '{smiles}': {e}")
                    return None

            unique_mapping = {}

            for inchikey, smiles in self.inchikey_smiles_map.items():
                if pd.notna(smiles):
                    canonical_smiles = canonicalize_smiles(smiles)
                    if canonical_smiles is not None:
                        if (inchikey, canonical_smiles) not in unique_mapping:
                            unique_mapping[(inchikey, canonical_smiles)] = (
                                canonical_smiles
                            )

            self.inchikey_smiles_map = {k[0]: v for k, v in unique_mapping.items()}

            self.df["SMILES"] = self.df["InChIKey"].map(self.inchikey_smiles_map)
            self.cleaned_data = self.df.dropna(subset=["SMILES"]).copy()

            # Set the Cleaned_SMILES column to the canonical SMILES values and update status
            for idx, row in self.cleaned_data.iterrows():
                cleaned_smiles = canonicalize_smiles(row["SMILES"])
                self.cleaned_data.at[idx, "Cleaned_SMILES"] = cleaned_smiles
                update_status(idx, "Duplicate check and canonicalization completed")

            logging.info(
                f"Number of unique entries after duplicate removal: {len(self.cleaned_data)}"
            )

        except Exception as e:
            logging.error(f"An error occurred during duplicate removal: {e}")

    def remove_inorganics_and_mixtures(self):
        """
        Remove inorganic compounds, organometallics, counterions, biologics, and mixtures from the DataFrame.
        """

        def update_status(idx, message):
            self.df.at[idx, "Status"] += f"{message}; "

        unique_inchikeys = self.df["InChIKey"].unique()
        logging.info(
            "Processing %d unique InChIKeys to remove inorganics, organometallics, counterions, biologics, and mixtures.",
            len(unique_inchikeys),
        )

        for idx, inchikey in enumerate(unique_inchikeys):
            smiles = self.inchikey_to_smiles(inchikey)
            if smiles:
                if (
                    self.is_organic(smiles)
                    and not self.is_counterion(smiles)
                    and not self.is_biologic(smiles)
                    and not self.is_mixture(smiles)
                ):
                    self.inchikey_smiles_map[inchikey] = smiles
                    update_status(idx, "Passed organic checks")
                else:
                    update_status(idx, "Failed organic checks")
            else:
                update_status(idx, "Failed to retrieve SMILES")

        self.df["SMILES"] = self.df["InChIKey"].map(self.inchikey_smiles_map)
        self.cleaned_data = self.df.dropna(subset=["SMILES"]).copy()
        self.cleaned_data["Cleaned_SMILES"] = None

        logging.info(
            "Removed inorganics, organometallics, counterions, biologics, and mixtures. Cleaned data shape: %s",
            self.cleaned_data.shape,
        )

    def inchikey_to_smiles(self, inchikey):
        """
        Convert an InChIKey to SMILES by querying an external chemical database.

        Args:
            inchikey (str): The InChIKey to convert.

        Returns:
            str: The corresponding SMILES string, or None if retrieval fails.
        """
        try:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchikey}/property/CanonicalSMILES/TXT"
            response = requests.get(url)
            response.raise_for_status()
            smiles = response.text.strip()
            if smiles:
                logging.info(f"Retrieved SMILES for InChIKey {inchikey}")
                return smiles  # Return valid SMILES string
            else:
                logging.warning(
                    f"No SMILES found for InChIKey {inchikey}. Returning empty string."
                )
                return ""  # Return empty string if no SMILES is found
        except Exception as e:
            logging.error(f"Failed to retrieve SMILES for InChIKey {inchikey}: {e}")
            return ""  # Return empty string if there is an error

    def structural_cleaning(self):
        """
        Perform comprehensive structural cleaning on the chemical dataset.
        """

        def update_status(idx, message):
            self.cleaned_data.at[idx, "Status"] += f"{message}; "

        def pre_optimization_check(mol):
            try:
                if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) == 0:
                    return mol, "3D conformation successful"
                else:
                    return None, "3D conformation failed"
            except Exception as e:
                logging.error(f"3D conformation error: {e}")
                return None, "3D conformation error"

        def apply_tautomer_rules(mol):
            """
            Apply RDKit's built-in tautomer rules to standardize the tautomeric form of the molecule.

            Args:
                mol (rdkit.Chem.Mol): The RDKit molecule object.

            Returns:
                rdkit.Chem.Mol: The molecule with standardized tautomers.
            """
            try:
                enumerator = rdMolStandardize.TautomerEnumerator()
                canonical_tautomer = enumerator.Canonicalize(mol)
                return canonical_tautomer
            except Exception as e:
                logging.error(f"Error applying tautomer rules: {e}")
                return mol

        def clean_structure(smiles, idx):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    update_status(idx, "Invalid SMILES")
                    return None

                mol = Chem.AddHs(mol)
                mol, pre_opt_status = pre_optimization_check(mol)
                if not mol:
                    update_status(idx, f"Pre-optimization failed: {pre_opt_status}")
                    return None

                try:
                    Chem.SanitizeMol(mol)
                    update_status(idx, "Sanitization successful")
                except Chem.MolSanitizeException:
                    try:
                        Chem.SanitizeMol(
                            mol,
                            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
                            ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
                        )
                        update_status(idx, "Sanitization corrected")
                    except Exception as e:
                        update_status(idx, f"Sanitization error: {e}")
                        return None

                try:
                    if AllChem.MMFFOptimizeMolecule(mol) != 0:
                        if AllChem.UFFOptimizeMolecule(mol) != 0:
                            update_status(idx, "Bond length/angle optimization failed")
                except Exception as e:
                    update_status(idx, f"Optimization error: {e}")

                try:
                    Chem.Kekulize(mol, clearAromaticFlags=True)
                    Chem.SetAromaticity(mol)
                    update_status(idx, "Aromatization successful")
                except Exception as e:
                    update_status(idx, f"Aromatization error: {e}")

                normalizer = rdMolStandardize.Normalizer()
                try:
                    mol = normalizer.normalize(mol)
                except Exception as e:
                    update_status(idx, f"Normalization error: {e}")
                    return None

                try:
                    mol = apply_tautomer_rules(mol)
                except Exception as e:
                    update_status(idx, f"Tautomer rules error: {e}")
                    return None

                if mol:
                    try:
                        cleaned_smiles = Chem.MolToSmiles(mol, canonical=True)
                        update_status(idx, "Structural cleaning successful")
                        return cleaned_smiles
                    except Exception as e:
                        update_status(idx, f"Failed to convert to SMILES: {e}")
                        return None
                else:
                    update_status(idx, "Molecule is None after processing")
                    return None

            except Exception as e:
                update_status(idx, f"Cleaning failed: {e}")
                return None

        self.cleaned_data["Cleaned_SMILES"] = None

        for idx, row in self.cleaned_data.iterrows():
            cleaned_smiles = clean_structure(row["SMILES"], idx)
            self.cleaned_data.at[idx, "Cleaned_SMILES"] = cleaned_smiles

        if "Cleaned_SMILES" not in self.cleaned_data.columns:
            logging.error("Cleaned_SMILES column missing after structural cleaning.")
            return

        self.cleaned_data.dropna(subset=["Cleaned_SMILES"], inplace=True)
        logging.info(
            "Structural cleaning completed. Cleaned data shape: %s",
            self.cleaned_data.shape,
        )

    def is_organic(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            elements = {atom.GetSymbol() for atom in mol.GetAtoms()}
            metals = {
                "Li",
                "Be",
                "Na",
                "Mg",
                "Al",
                "K",
                "Ca",
                "Sc",
                "Ti",
                "V",
                "Cr",
                "Mn",
                "Fe",
                "Co",
                "Ni",
                "Cu",
                "Zn",
                "Ga",
                "Rb",
                "Sr",
                "Y",
                "Zr",
                "Nb",
                "Mo",
                "Tc",
                "Ru",
                "Rh",
                "Pd",
                "Ag",
                "Cd",
                "In",
                "Sn",
                "Sb",
                "Cs",
                "Ba",
                "La",
                "Hf",
                "Ta",
                "W",
                "Re",
                "Os",
                "Ir",
                "Pt",
                "Au",
                "Hg",
                "Tl",
                "Pb",
                "Bi",
            }
            if elements & metals or "C" not in elements or len(elements) <= 1:
                return False
            return True
        except Exception as e:
            logging.error(f"Error processing SMILES {smiles}: {e}")
            return False

    def is_counterion(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            return Descriptors.MolWt(mol) < 100
        except Exception as e:
            logging.error(f"Error processing SMILES {smiles} for counterion check: {e}")
            return False

    def is_biologic(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            peptide_bond_count = smiles.count("C(=O)N")
            return peptide_bond_count > 5
        except Exception as e:
            logging.error(f"Error processing SMILES {smiles} for biologic check: {e}")
            return False

    def is_mixture(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            return "." in smiles
        except Exception as e:
            logging.error(f"Error processing SMILES {smiles} for mixture check: {e}")
            return False

    def verify_stereochemistry(self):
        """
        Verify the correctness of stereochemistry for bioactive chemicals.
        The method includes:
        - Detection of stereocenters using RDKit.
        - Comparison of stereocenters with entries in PubChem.
        - Indication of potential issues requiring manual curation.

        Returns:
            pd.DataFrame: DataFrame with additional columns indicating stereocenter verification results.
        """

        def update_status(idx, message):
            self.cleaned_data.at[idx, "Status"] += f"{message}; "

        def get_stereocenters(mol):
            """
            Detect stereocenters in a molecule.

            Args:
                mol (rdkit.Chem.Mol): RDKit molecule object.

            Returns:
                list: A list of tuples where each tuple contains atom index and stereochemistry (R/S).
            """
            stereo_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
            return stereo_centers

        def query_pubchem(smiles):
            """
            Query PubChem for similar structures and retrieve stereochemistry information.

            Args:
                smiles (str): SMILES string of the molecule.

            Returns:
                dict: A dictionary with PubChem stereochemistry information.
            """
            try:
                url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/property/CanonicalSMILES,IsomericSMILES/JSON"
                response = requests.get(url)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                logging.error(
                    f"Failed to query PubChem for SMILES {smiles}. Reason: {e}"
                )
                return None

        def manual_curation_needed(stereo_centers, pubchem_info):
            """
            Determine if manual curation is needed based on discrepancies in stereochemistry.

            Args:
                stereo_centers (list): List of detected stereocenters.
                pubchem_info (dict): Stereochemistry info from PubChem.

            Returns:
                bool: True if manual curation is recommended, False otherwise.
            """
            if not pubchem_info:
                return True

            pubchem_stereocenters = []  # Extract this from PubChem's response if available

            # Compare detected stereocenters with PubChem info
            if len(stereo_centers) != len(pubchem_stereocenters):
                return True

            # Further comparison logic can be added here

            return False

        def verify_structure(smiles, idx):
            """
            Verify the stereochemistry of a single structure.

            Args:
                smiles (str): SMILES string of the molecule.

            Returns:
                dict: A dictionary with verification results.
            """
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                update_status(idx, "Invalid SMILES")
                return {"SMILES": smiles, "Valid": False, "Error": "Invalid SMILES"}

            stereo_centers = get_stereocenters(mol)
            pubchem_info = query_pubchem(smiles)

            needs_manual_curation = manual_curation_needed(stereo_centers, pubchem_info)

            if needs_manual_curation:
                update_status(idx, "Stereochemistry verification required")
            else:
                update_status(idx, "Stereochemistry verified")

            return {
                "SMILES": smiles,
                "Stereocenters": stereo_centers,
                "PubChem_Info": pubchem_info,
                "Manual_Curation_Required": needs_manual_curation,
            }

        for idx, row in self.cleaned_data.iterrows():
            verify_structure(row["SMILES"], idx)

        logging.info("Stereochemistry verification completed.")

        return self.cleaned_data

    def curate_data(self):
        """
        Run the full curation workflow on the DataFrame.
        The workflow includes:
        - Removal of inorganics, organometallics, counterions, biologics, and mixtures.
        - Structural cleaning, including valence correction, bond optimization, and chemotype normalization.
        - Duplicate removal based on InChIKey and Cleaned_SMILES.
        - Stereochemistry verification.
        """
        logging.info("Starting data curation process.")

        # Step 1: Initial Data Cleaning
        self.remove_inorganics_and_mixtures()

        # Step 2: Structural Cleaning and Normalization
        self.structural_cleaning()

        # Step 3: Duplicate Removal
        self.remove_duplicates()  # Add the duplicate removal step

        # Step 4: Stereochemistry Verification
        self.verify_stereochemistry()

        logging.info("Data curation process completed.")
        return self.cleaned_data

    def generate_sdf(self, output_file):
        """
        Generate an SDF file from the curated SMILES data.

        Args:
            output_file (str): The path to the output SDF file.
        """
        logging.info("Generating SDF file at %s", output_file)
        writer = SDWriter(output_file)
        for _, row in self.cleaned_data.iterrows():
            mol = Chem.MolFromSmiles(row["Cleaned_SMILES"])
            if mol is not None:
                mol.SetProp("InChIKey", row["InChIKey"])
                mol.SetProp("Status", row["Status"])
                writer.write(mol)
        writer.close()
        logging.info("SDF file generation complete.")


class BiologicalCuration:
    """
    A class to perform additional biological curation, including activity cliffs identification.
    """

    def __init__(self, df):
        self.df = df

    def identify_activity_cliffs(self, threshold=0.5, activity_threshold=0.2):
        """
        Identify activity cliffs using structural similarity and bioactivity values.
        Clustering compounds with similar structures and comparing their bioactivity.
        """
        try:
            logging.info("Identifying activity cliffs...")

            # Ensure "Cleaned_SMILES" column exists before proceeding
            if "Cleaned_SMILES" not in self.df.columns:
                logging.error("Missing 'Cleaned_SMILES' column in dataframe.")
                return self.df

            # Validate and generate fingerprints for all valid molecules
            mols = self.df["Cleaned_SMILES"].apply(Chem.MolFromSmiles)
            valid_mols = [mol for mol in mols if mol is not None]

            if not valid_mols:
                logging.warning(
                    "No valid molecules found for activity cliff identification."
                )
                return self.df

            fingerprints = rdFingerprintGenerator.GetFPs(valid_mols)

            # Calculate similarity matrix using Tanimoto similarity
            similarity_matrix = np.zeros((len(fingerprints), len(fingerprints)))
            for i, fp1 in enumerate(fingerprints):
                for j, fp2 in enumerate(fingerprints):
                    similarity_matrix[i, j] = DataStructs.TanimotoSimilarity(fp1, fp2)

            # Apply DBSCAN clustering to find potential activity cliffs
            clustering = DBSCAN(eps=threshold, min_samples=2, metric="precomputed")
            clusters = clustering.fit_predict(1 - similarity_matrix)

            self.df.loc[
                self.df["Cleaned_SMILES"].apply(Chem.MolFromSmiles).notnull(),
                "Activity_Cluster",
            ] = clusters

            # Filter clusters based on bioactivity values if available
            if "Bioactivity" not in self.df.columns:
                logging.warning(
                    "No 'Bioactivity' column found; skipping bioactivity-based cliff filtering."
                )
                return self.df

            # Proceed with filtering based on bioactivity if the column exists
            cliff_data = []
            for cluster in set(clusters):
                if cluster == -1:
                    continue  # Skip noise points
                cluster_df = self.df[self.df["Activity_Cluster"] == cluster]
                bioactivity_range = (
                    cluster_df["Bioactivity"].max() - cluster_df["Bioactivity"].min()
                )
                if bioactivity_range >= activity_threshold:
                    cliff_data.append(cluster_df)

            # Combine all the identified cliffs
            activity_cliffs_df = (
                pd.concat(cliff_data, ignore_index=True) if cliff_data else self.df
            )

            return activity_cliffs_df

        except Exception as e:
            logging.error(f"Failed to identify activity cliffs: {e}")
            return self.df


def main(input_file, csv_output_file, sdf_output_file, row_limit=None):
    try:
        # Validate input file existence
        if not os.path.isfile(input_file):
            logging.error(f"Input file not found: {input_file}")
            return

        # Read the input CSV file
        logging.info(f"Reading input file: {input_file}")
        df_with_inchikey = pd.read_csv(input_file)

        # Limit the number of rows if specified
        if row_limit:
            df_with_inchikey = df_with_inchikey.head(row_limit)

        # Create an instance of the ChemicalCuration class
        curator = ChemicalCuration(df_with_inchikey)

        # Run the data curation process
        curated_df = curator.curate_data()

        # # Perform additional biological curation
        bio_curator = BiologicalCuration(curated_df)
        curated_df = bio_curator.identify_activity_cliffs()

        # Check if the cleaned dataset has valid rows before saving
        if curated_df.empty:
            logging.warning(
                "No valid entries found after curation. Skipping file generation."
            )
            return

        # Save the curated DataFrame to the specified output CSV file
        output_dir = os.path.dirname(csv_output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        curated_df.to_csv(csv_output_file, index=False)
        logging.info(f"Data successfully saved to: {csv_output_file}")

        # Generate an SDF file from the curated data
        curator.generate_sdf(sdf_output_file)
        logging.info(f"SDF file successfully generated at: {sdf_output_file}")

    except Exception as e:
        logging.error(f"An error occurred during processing: {e}")


if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Chemical data curation script")
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument("csv_output_file", help="Path to the output CSV file")
    parser.add_argument("sdf_output_file", help="Path to the output SDF file")
    parser.add_argument(
        "--row_limit", type=int, default=None, help="Limit the number of rows processed"
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Call the main function with the provided arguments
    main(args.input_file, args.csv_output_file, args.sdf_output_file, args.row_limit)


"""
python .\Chemical_Curation_Workflow.py '..\DataSet\4.LifeStageData-InChIKeyRetrieved.csv' '..\DataSet\5.LifeStageData-CompoundsCurated.csv' '..\DataSet\5.LifeStageData-CompoundsCurated.sdf' --row_limit 100
"""
