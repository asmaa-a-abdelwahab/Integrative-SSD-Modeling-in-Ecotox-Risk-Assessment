import argparse
import logging
import os

import numpy as np
import pandas as pd
import requests
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, SDWriter, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.cluster import DBSCAN

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ChemicalCuration:
    def __init__(self, df, keep_manual_curation=True):
        """
        Initialize a ChemicalCuration object.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing the chemical data to be curated.
        keep_manual_curation : bool, optional
            Whether to retain rows that require manual curation or drop them.
            Defaults to True.

        Attributes
        ----------
        df : pandas.DataFrame
            The input DataFrame.
        keep_manual_curation : bool
            Whether to retain rows that require manual curation or drop them.
        cleaned_data : pandas.DataFrame or None
            The cleaned DataFrame after curation, or None if not yet curated.
        inchikey_smiles_map : dict
            Mapping of InChIKeys to SMILES strings, populated during curation.
        """
        self.df = df
        self.keep_manual_curation = keep_manual_curation
        self.cleaned_data = None
        self.df["Status"] = ""
        self.inchikey_smiles_map = {}
        logging.info(
            "ChemicalCuration initialized with DataFrame of shape %s", df.shape
        )

    def update_status(self, idx, message):
        """Update the 'Status' column of the input DataFrame at the given index.

        Parameters
        ----------
        idx : int
            Index of the row to update.
        message : str
            Message to append to the 'Status' column.

        Returns
        -------
        None
        """
        if self.df is not None and isinstance(self.df, pd.DataFrame):
            try:
                self.df.at[idx, "Status"] += f"{message}; "
            except Exception as e:
                logging.error(f"Failed to update status at index {idx}: {e}")

    def remove_duplicates(self):
        """
        Remove duplicates based on InChIKey and canonical Cleaned_SMILES.

        This method removes duplicates by canonicalizing the Cleaned_SMILES column
        and then grouping by InChIKey and canonical Cleaned_SMILES. The first entry
        in each group is kept, and the rest are dropped.

        If the 'Cleaned_SMILES' column is not found in the cleaned dataset, this
        method returns without making any changes.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        try:
            if "Cleaned_SMILES" not in self.cleaned_data.columns:
                logging.error(
                    "'Cleaned_SMILES' column not found in the cleaned dataset"
                )
                return

            logging.info(
                "Removing duplicates based on InChIKey and canonical Cleaned_SMILES..."
            )

            def canonicalize_smiles(smiles):
                """
                Canonicalize a SMILES string.

                Parameters
                ----------
                smiles : str
                    SMILES string to be canonicalized.

                Returns
                -------
                str or None
                    Canonicalized SMILES string or None if canonicalization fails.
                """
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    return Chem.MolToSmiles(mol, canonical=True) if mol else None
                except Exception as e:
                    logging.error(f"Failed to canonicalize SMILES '{smiles}': {e}")
                    return None

            unique_mapping = {}

            for inchikey, smiles in self.inchikey_smiles_map.items():
                if pd.notna(smiles):
                    canonical_smiles = canonicalize_smiles(smiles)
                    if (
                        canonical_smiles
                        and (inchikey, canonical_smiles) not in unique_mapping
                    ):
                        unique_mapping[(inchikey, canonical_smiles)] = canonical_smiles
                        self.update_status(
                            self.cleaned_data.index[
                                self.cleaned_data["InChIKey"] == inchikey
                            ][0],
                            "Passed duplicate check and canonicalization completed",
                        )

            self.inchikey_smiles_map = {k[0]: v for k, v in unique_mapping.items()}

            self.df["SMILES"] = self.df["InChIKey"].map(self.inchikey_smiles_map)
            self.cleaned_data = self.df.dropna(subset=["SMILES"]).copy()

            logging.info(
                f"Number of unique entries after duplicate removal: {len(self.cleaned_data)}"
            )

        except Exception as e:
            logging.error(f"An error occurred during duplicate removal: {e}")

    def remove_inorganics_and_mixtures(self):
        """
        Remove inorganics, organometallics, counterions, biologics, and mixtures by
        checking the SMILES string associated with each InChIKey. If the SMILES
        string is None or fails the checks, the entire row is dropped from the
        DataFrame. The cleaned DataFrame is stored in the 'cleaned_data' attribute.

        """
        if not isinstance(self.df, pd.DataFrame) or "InChIKey" not in self.df.columns:
            logging.error(
                "'InChIKey' column not found or DataFrame not initialized properly."
            )
            return

        unique_inchikeys = self.df["InChIKey"].drop_duplicates().unique()
        logging.info(
            "Processing %d unique InChIKeys to remove inorganics, organometallics, counterions, biologics, and mixtures.",
            len(unique_inchikeys),
        )

        if "Status" not in self.df.columns:
            self.df["Status"] = ""

        indices_to_drop = []

        for inchikey in unique_inchikeys:
            smiles = self.inchikey_to_smiles(inchikey)
            if smiles:
                try:
                    is_organic = self.is_organic(smiles)
                    is_counterion = self.is_counterion(smiles)
                    is_biologic = self.is_biologic(smiles)
                    is_mixture = self.is_mixture(smiles)

                    failure_reasons = []
                    if not is_organic:
                        failure_reasons.append("Not organic")
                    if is_counterion:
                        failure_reasons.append("Is counterion")
                    if is_biologic:
                        failure_reasons.append("Is biologic")
                    if is_mixture:
                        failure_reasons.append("Is mixture")

                    if not failure_reasons:
                        self.inchikey_smiles_map[inchikey] = smiles
                        self.df.loc[self.df["InChIKey"] == inchikey, "Status"] += (
                            "Passed Initial Structural Checks; "
                        )
                    else:
                        logging.info(
                            f"InChIKey {inchikey} failed checks: {', '.join(failure_reasons)}"
                        )
                        self.df.loc[self.df["InChIKey"] == inchikey, "Status"] += (
                            f"Failed checks: {', '.join(failure_reasons)}; "
                        )
                        indices_to_drop.extend(
                            self.df[self.df["InChIKey"] == inchikey].index.tolist()
                        )
                        self.inchikey_smiles_map[inchikey] = None

                except Exception as e:
                    logging.error(f"Error during checks for InChIKey {inchikey}: {e}")
                    self.df.loc[self.df["InChIKey"] == inchikey, "Status"] += (
                        f"Error: {e}; "
                    )
                    indices_to_drop.extend(
                        self.df[self.df["InChIKey"] == inchikey].index.tolist()
                    )
                    self.inchikey_smiles_map[inchikey] = None

            else:
                logging.info(f"Failed to retrieve SMILES for InChIKey {inchikey}")
                self.df.loc[self.df["InChIKey"] == inchikey, "Status"] += (
                    "Failed to retrieve SMILES; "
                )
                indices_to_drop.extend(
                    self.df[self.df["InChIKey"] == inchikey].index.tolist()
                )
                self.inchikey_smiles_map[inchikey] = None

        self.df.drop(indices_to_drop, inplace=True)
        self.df["SMILES"] = self.df["InChIKey"].map(self.inchikey_smiles_map)
        self.cleaned_data = self.df.dropna(subset=["SMILES"]).copy()
        self.cleaned_data["Cleaned_SMILES"] = None

        if self.cleaned_data.empty:
            logging.warning("No valid entries after filtering. Cleaned data is empty.")
            return

        logging.info(
            "Completed filtering of inorganics, organometallics, counterions, biologics, and mixtures. Cleaned data shape: %s",
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
        Perform structural cleaning on the SMILES strings in the cleaned data.

        This step applies RDKit's built-in tautomer rules, sanitizes the molecule, optimizes bond lengths and angles, aromatizes, normalizes,
        and applies canonicalization to the SMILES strings. The cleaned SMILES strings are then stored in the 'Cleaned_SMILES' column of the
        cleaned data.

        Args:

        Returns:
            None
        """

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
            """
            Perform structural cleaning on a single SMILES string.

            This function applies RDKit's built-in tautomer rules, sanitizes the molecule, optimizes bond lengths and angles, aromatizes, normalizes,
            and applies canonicalization to the SMILES string.

            Args:
                smiles (str): The SMILES string to be cleaned.
                idx (int): The index of the SMILES string in the original DataFrame.

            Returns:
                str: The cleaned SMILES string, or None if any step of the cleaning process fails.
            """
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    self.update_status(idx, "Invalid SMILES")
                    return None

                mol = Chem.AddHs(mol)

                # Call the pre_optimization_check method
                mol, pre_opt_status = pre_optimization_check(mol)
                if not mol:
                    self.update_status(
                        idx, f"Pre-optimization failed: {pre_opt_status}"
                    )
                    return None

                try:
                    Chem.SanitizeMol(mol)
                    self.update_status(idx, "Sanitization successful")
                except Chem.MolSanitizeException:
                    try:
                        Chem.SanitizeMol(
                            mol,
                            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
                            ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
                        )
                        self.update_status(idx, "Sanitization corrected")
                    except Exception as e:
                        self.update_status(idx, f"Sanitization error: {e}")
                        return None

                try:
                    # Bond length/angle optimization
                    if AllChem.MMFFOptimizeMolecule(mol) != 0:
                        if AllChem.UFFOptimizeMolecule(mol) != 0:
                            self.update_status(
                                idx, "Bond length/angle optimization failed"
                            )
                            return None  # Return None to indicate failure
                except Exception as e:
                    self.update_status(idx, f"Optimization error: {e}")
                    return None

                try:
                    Chem.Kekulize(mol, clearAromaticFlags=True)
                    Chem.SetAromaticity(mol)
                    self.update_status(idx, "Aromatization successful")
                except Exception as e:
                    self.update_status(idx, f"Aromatization error: {e}")
                    return None

                normalizer = rdMolStandardize.Normalizer()
                try:
                    mol = normalizer.normalize(mol)
                    self.update_status(idx, "Normalization successful")
                except Exception as e:
                    self.update_status(idx, f"Normalization error: {e}")
                    return None

                # Call the apply_tautomer_rules method
                try:
                    mol = apply_tautomer_rules(mol)
                    self.update_status(idx, "Tautomer rules applied")
                except Exception as e:
                    self.update_status(idx, f"Tautomer rules error: {e}")
                    return None

                if mol:
                    try:
                        cleaned_smiles = Chem.MolToSmiles(mol, canonical=True)
                        self.update_status(idx, "Structural cleaning successful")
                        return cleaned_smiles
                    except Exception as e:
                        self.update_status(idx, f"Failed to convert to SMILES: {e}")
                        return None
                else:
                    self.update_status(idx, "Molecule is None after processing")
                    return None

            except Exception as e:
                self.update_status(idx, f"Cleaning failed: {e}")
                return None

        # Now iterate over each row and clean the structure
        self.cleaned_data["Cleaned_SMILES"] = None

        for idx, row in self.cleaned_data.iterrows():
            cleaned_smiles = clean_structure(row["SMILES"], idx)
            self.cleaned_data.at[idx, "Cleaned_SMILES"] = cleaned_smiles

        # Drop rows with failed cleaning steps (i.e., missing Cleaned_SMILES)
        if "Cleaned_SMILES" not in self.cleaned_data.columns:
            logging.error("Cleaned_SMILES column missing after structural cleaning.")
            return

        self.cleaned_data.dropna(subset=["Cleaned_SMILES"], inplace=True)
        logging.info(
            "Structural cleaning completed. Cleaned data shape: %s",
            self.cleaned_data.shape,
        )
        self.cleaned_data["Status"] = self.df["Status"]

    def is_organic(self, smiles):
        """
        Determine if a given SMILES string represents an organic molecule.

        An organic molecule is defined as one that contains carbon and does not contain any metal atoms.
        The presence of metal atoms or absence of carbon will result in the molecule being classified as
        non-organic.

        Args:
            smiles (str): The SMILES string representation of the molecule to be evaluated.

        Returns:
            bool: True if the molecule is organic, False otherwise.
        """
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
        """
        Check if a SMILES string represents a counterion (defined as a molecule with a molecular weight < 100 Da).

        Parameters
        ----------
        smiles : str
            SMILES string to be checked.

        Returns
        -------
        bool
            True if the SMILES string represents a counterion, False otherwise.
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            return Descriptors.MolWt(mol) < 100
        except Exception as e:
            logging.error(f"Error processing SMILES {smiles} for counterion check: {e}")
            return False

    def is_biologic(self, smiles):
        """
        Determine if a SMILES string represents a biologic molecule.

        A biologic molecule is characterized by the presence of peptide bonds.
        This function assesses the count of peptide bonds (C(=O)N) in the SMILES
        string and classifies it as biologic if the count exceeds 5.

        Args:
            smiles (str): The SMILES string representation of the molecule to be evaluated.

        Returns:
            bool: True if the molecule is considered biologic based on peptide bond count, False otherwise.
        """
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
        """
        Determine if a SMILES string represents a mixture of molecules.

        A mixture is classified as a SMILES string containing a '.' character, which is used to separate individual molecule
        representations in a mixture.

        Args:
            smiles (str): The SMILES string representation of the molecule to be evaluated.

        Returns:
            bool: True if the SMILES string represents a mixture of molecules, False otherwise.
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            return "." in smiles
        except Exception as e:
            logging.error(f"Error processing SMILES {smiles} for mixture check: {e}")
            return False

    def get_stereocenters(self, mol):
        """
        Detect stereocenters in a molecule.

        Args:
            mol (rdkit.Chem.Mol): RDKit molecule object.

        Returns:
            list: A list of tuples where each tuple contains atom index and stereochemistry (R/S).
        """
        stereo_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        return stereo_centers

    def query_pubchem(self, smiles):
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
            logging.error(f"Failed to query PubChem for SMILES {smiles}. Reason: {e}")
            return None

    def manual_curation_needed(self, stereo_centers, pubchem_info):
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

    def verify_structure(self, smiles, idx):
        """
        Verify the stereochemistry of a single structure.

        Args:
            smiles (str): SMILES string of the molecule.

        Returns:
            dict: A dictionary with verification results.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            self.update_status(idx, "Invalid SMILES")
            return {"SMILES": smiles, "Valid": False, "Error": "Invalid SMILES"}

        # Step 1: Detect stereocenters in the molecule
        stereo_centers = self.get_stereocenters(mol)
        if not stereo_centers:
            self.update_status(idx, "No stereocenters detected")
            return {
                "SMILES": smiles,
                "Stereocenters": stereo_centers,
                "Manual_Curation_Required": False,
            }

        # Step 2: Query PubChem for stereochemistry information
        pubchem_info = self.query_pubchem(smiles)
        if not pubchem_info:
            self.update_status(idx, "PubChem query failed")
            return {
                "SMILES": smiles,
                "Stereocenters": stereo_centers,
                "Manual_Curation_Required": True,
            }

        # Step 3: Check for discrepancies
        needs_manual_curation = self.manual_curation_needed(
            stereo_centers, pubchem_info
        )

        if needs_manual_curation:
            self.update_status(idx, "Stereochemistry verification required")
        else:
            self.update_status(idx, "Stereochemistry verified")

        return {
            "SMILES": smiles,
            "Stereocenters": stereo_centers,
            "PubChem_Info": pubchem_info,
            "Manual_Curation_Required": needs_manual_curation,
        }

    def verify_stereochemistry(self):
        """
        Verify stereochemistry for each molecule in the cleaned dataset.

        This method iterates over the cleaned dataset and verifies the stereochemistry
        of each molecule based on its SMILES representation. It updates the status
        of each molecule with the result of the verification. If manual curation
        is required and `keep_manual_curation` is False, it drops those entries from
        the dataset. Logs the verification results and any errors encountered.

        Returns
        -------
        pandas.DataFrame
            The updated cleaned DataFrame with stereochemistry verification statuses.
        """
        indices_to_drop = []

        for idx, row in self.cleaned_data.iterrows():
            try:
                verification_result = self.verify_structure(row["SMILES"], idx)
                if verification_result["Manual_Curation_Required"]:
                    self.cleaned_data.at[idx, "Status"] += "Manual Curation Required; "
                    if not self.keep_manual_curation:
                        indices_to_drop.append(idx)
                else:
                    self.cleaned_data.at[idx, "Status"] += (
                        "Passed Stereochemistry Verification; "
                    )
                logging.info(
                    f"Stereochemistry verification result for row {idx}: {verification_result}"
                )
            except Exception as e:
                self.update_status(
                    idx, f"Stereochemistry verification failed due to error: {e}"
                )
                logging.error(f"Error verifying stereochemistry for row {idx}: {e}")
                indices_to_drop.append(idx)

        if not self.keep_manual_curation and indices_to_drop:
            self.cleaned_data.drop(indices_to_drop, inplace=True)

        logging.info("Stereochemistry verification completed.")
        return self.cleaned_data

    def curate_data(self):
        """
        Runs the entire chemical curation workflow.

        This method orchestrates the entire curation process, including initial
        data cleaning, structural cleaning and normalization, duplicate removal,
        and stereochemistry verification.

        Returns
        -------
        pandas.DataFrame
            The fully curated DataFrame.
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
        """
        Initialize a BiologicalCuration object with a given DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing the data to be used for biological curation.
        """
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


def main(
    input_file,
    csv_output_file,
    sdf_output_file,
    row_limit=None,
    keep_manual_curation=True,
):
    """
    Main function to curate the chemical data and generate output files.

    Parameters
    ----------
    input_file : str
        Path to the input CSV file.
    csv_output_file : str
        Path to the output CSV file.
    sdf_output_file : str
        Path to the output SDF file.
    row_limit : int, optional
        Maximum number of rows to read from the input CSV file. If not specified, all rows are read.
    keep_manual_curation : bool, optional
        Whether to retain rows that require manual curation or drop them. Defaults to True.

    Returns
    -------
    None
    """
    try:
        if not os.path.isfile(input_file):
            logging.error(f"Input file not found: {input_file}")
            return

        logging.info(f"Reading input file: {input_file}")
        df_with_inchikey = pd.read_csv(input_file)

        if row_limit:
            df_with_inchikey = df_with_inchikey.head(row_limit)

        curator = ChemicalCuration(df_with_inchikey, keep_manual_curation)
        curated_df = curator.curate_data()

        bio_curator = BiologicalCuration(curated_df)
        curated_df = bio_curator.identify_activity_cliffs()

        if curated_df.empty:
            logging.warning(
                "No valid entries found after curation. Skipping file generation."
            )
            return

        output_dir = os.path.dirname(csv_output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        curated_df.to_csv(csv_output_file, index=False)
        logging.info(f"Data successfully saved to: {csv_output_file}")

        curator.generate_sdf(sdf_output_file)
        logging.info(f"SDF file successfully generated at: {sdf_output_file}")

    except Exception as e:
        logging.error(f"An error occurred during processing: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chemical data curation script")
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument("csv_output_file", help="Path to the output CSV file")
    parser.add_argument("sdf_output_file", help="Path to the output SDF file")
    parser.add_argument(
        "--row_limit", type=int, default=None, help="Limit the number of rows processed"
    )
    parser.add_argument(
        "--keep_manual_curation",
        action="store_true",
        help="Keep compounds requiring manual curation",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    main(
        args.input_file,
        args.csv_output_file,
        args.sdf_output_file,
        args.row_limit,
        args.keep_manual_curation,
    )


"""
python .\Chemical_Curation_Workflow.py '..\DataSet\4.LifeStageData-InChIKeyRetrieved.csv' '..\DataSet\5.LifeStageData-CompoundsCurated.csv' '..\DataSet\5.LifeStageData-CompoundsCurated.sdf' --row_limit 100 -- keep_manual_curation 
"""
