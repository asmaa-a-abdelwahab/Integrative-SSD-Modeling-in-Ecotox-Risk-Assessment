# Execution time: 3h
import requests
import pubchempy as pcp
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from requests.exceptions import RequestException
from rdkit import Chem
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class InChIKeyStandardizer:
    def __init__(self):
        """
        Initialize the InChIKeyStandardizer without any specific chemical name.
        Counters Initialized in __init__:
          self.case_pubchem_only: Counts cases where InChIKey is found only in PubChem.
          self.case_opsin_only: Counts cases where InChIKey is found only in OPSIN.
          self.case_pubchem_opsin_match: Counts cases where PubChem and OPSIN InChIKeys match.
          self.case_pubchem_opsin_rdkit_match: Counts cases where PubChem, OPSIN, and RDKit InChIKeys all match.
          self.case_no_match: Counts cases where InChIKeys do not match across the sources.
          self.case_no_inchikey_found: Counts cases where no InChIKey could be found.
        """
        self.pubchem_inchikey = None
        self.opsin_inchikey = None
        self.standardized_inchikey = None

        # Counters for different cases
        self.case_pubchem_only = 0
        self.case_opsin_only = 0
        self.case_pubchem_opsin_match = 0
        self.case_pubchem_opsin_rdkit_match = 0
        self.case_no_match = 0
        self.case_no_inchikey_found = 0

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10), retry=retry_if_exception_type(RequestException))
    def retrieve_inchikey_pubchem(self, chemical_name):
        """
        Retrieve the InChIKey for a given chemical name using the PubChemPy API.

        Args:
            chemical_name (str): The name of the chemical.

        Returns:
            str: The InChIKey retrieved from PubChem, or None if not found.
        """
        try:
            compounds = pcp.get_compounds(chemical_name, 'name')
            if compounds:
                self.pubchem_inchikey = compounds[0].inchikey
            return self.pubchem_inchikey
        except Exception as e:
            logging.error(f"Failed to retrieve InChIKey from PubChem for {chemical_name}: {e}")
            return None

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10), retry=retry_if_exception_type(RequestException))
    def retrieve_inchikey_opsin(self, chemical_name):
        """
        Retrieve the InChIKey for a given chemical name using the OPSIN web service.

        Args:
            chemical_name (str): The name of the chemical.

        Returns:
            str: The InChIKey retrieved from OPSIN, or None if not found.
        """
        try:
            url = f"http://opsin.ch.cam.ac.uk/opsin/{chemical_name}.stdinchikey"
            response = requests.get(url)
            if response.status_code == 200:
                self.opsin_inchikey = response.text.strip()
            return self.opsin_inchikey
        except Exception as e:
            logging.error(f"Failed to retrieve InChIKey from OPSIN for {chemical_name}: {e}")
            return None

    def regenerate_inchikey_with_rdkit(self, inchi):
        """
        Regenerate the InChIKey using RDKit from an InChI string.

        Args:
            inchi (str): The InChI string to regenerate the InChIKey from.

        Returns:
            str: The regenerated InChIKey, or None if the process fails.
        """
        try:
            molecule = Chem.MolFromInchi(inchi)
            if molecule:
                regenerated_inchikey = Chem.InchiToInchiKey(inchi)
                return regenerated_inchikey
            else:
                logging.error("Failed to create a molecule from InChI string using RDKit.")
                return None
        except Exception as e:
            logging.error(f"RDKit failed to regenerate InChIKey: {e}")
            return None

    def retrieve_and_regenerate_inchikey_with_rdkit(self, chemical_name):
        """
        Retrieve the InChI string from PubChem and regenerate the InChIKey using RDKit.

        Args:
            chemical_name (str): The name of the chemical.

        Returns:
            str: The regenerated InChIKey, or None if the InChI retrieval or regeneration fails.
        """
        try:
            compounds = pcp.get_compounds(chemical_name, 'name')
            if compounds:
                inchi = compounds[0].inchi
                if inchi:
                    return self.regenerate_inchikey_with_rdkit(inchi)
                else:
                    logging.warning(f"InChI not found for {chemical_name} in PubChem.")
            return None
        except Exception as e:
            logging.error(f"Failed to retrieve InChI from PubChem for {chemical_name}: {e}")
            return None

    def standardize_inchikey(self, chemical_name):
        """
        Standardize the InChIKey by comparing PubChem and OPSIN results and regenerating with RDKit.

        Args:
            chemical_name (str): The name of the chemical.

        Returns:
            str: The standardized InChIKey, or None if all methods fail.
        """
        # Retrieve InChIKey from PubChem and OPSIN
        self.retrieve_inchikey_pubchem(chemical_name)
        self.retrieve_inchikey_opsin(chemical_name)

        if self.pubchem_inchikey and not self.opsin_inchikey:
            self.case_pubchem_only += 1
        elif not self.pubchem_inchikey and self.opsin_inchikey:
            self.case_opsin_only += 1

        # If both are the same, compare with the InChIKey regenerated by RDKit
        if self.pubchem_inchikey and self.opsin_inchikey and self.pubchem_inchikey == self.opsin_inchikey:
            regenerated_inchikey = self.retrieve_and_regenerate_inchikey_with_rdkit(chemical_name)
            if regenerated_inchikey == self.pubchem_inchikey:
                self.case_pubchem_opsin_rdkit_match += 1
                self.standardized_inchikey = self.pubchem_inchikey
                return self.standardized_inchikey
            else:
                logging.warning(f"PubChem and OPSIN InChIKeys match, but do not match RDKit regenerated InChIKey for {chemical_name}.")
                self.case_no_match += 1
                self.standardized_inchikey = regenerated_inchikey
                return self.standardized_inchikey

        # If they differ or RDKit regeneration is different, prioritize the RDKit regenerated InChIKey
        regenerated_inchikey = self.retrieve_and_regenerate_inchikey_with_rdkit(chemical_name)
        if regenerated_inchikey:
            self.case_no_match += 1
            self.standardized_inchikey = regenerated_inchikey
            return self.standardized_inchikey

        # If no consensus is reached, return the available one
        if self.pubchem_inchikey or self.opsin_inchikey:
            self.case_no_match += 1
            self.standardized_inchikey = self.pubchem_inchikey or self.opsin_inchikey
            return self.standardized_inchikey
        else:
            self.case_no_inchikey_found += 1
            return None

    def generate_standardized_inchikey(self, df, chemical_name_column='Chemical Name'):
        """
        Update a DataFrame with standardized InChIKeys based on chemical names.

        Args:
            df (pandas.DataFrame): The DataFrame containing chemical names.
            chemical_name_column (str): The column name in the DataFrame that contains chemical names.

        Returns:
            pandas.DataFrame: The updated DataFrame with a new column 'inchikey' containing the standardized InChIKeys.
        """
        # Extract unique chemical names
        unique_chemical_names = df[chemical_name_column].unique()

        # Create a dictionary to store InChIKeys for each unique chemical name
        inchikey_dict = {}

        # Populate the dictionary using the standardization process
        for idx, name in enumerate(unique_chemical_names, start=1):
            standardized_inchikey = self.standardize_inchikey(name)
            inchikey_dict[name] = standardized_inchikey

            # Log progress every 500 chemical names
            if idx % 500 == 0:
                logging.info(f"Processed {idx} chemical names")

        # Map the InChIKeys to the DataFrame
        df['InChIKey'] = df[chemical_name_column].map(inchikey_dict)
        df =  df_with_inchikey[~df['InChIKey'].isnull()]
        df.drop_duplicates(inplace=True)

        # Log final counts for each case
        logging.info(f"Case PubChem Only: {self.case_pubchem_only}")
        logging.info(f"Case OPSIN Only: {self.case_opsin_only}")
        logging.info(f"Case PubChem and OPSIN Match: {self.case_pubchem_opsin_match}")
        logging.info(f"Case PubChem, OPSIN, and RDKit Match: {self.case_pubchem_opsin_rdkit_match}")
        logging.info(f"Case No Match: {self.case_no_match}")
        logging.info(f"Case No InChIKey Found: {self.case_no_inchikey_found}")

        return df

# Assuming filtered_df is a DataFrame that already exists
standardizer = InChIKeyStandardizer()
df_with_inchikey = standardizer.generate_standardized_inchikey(filtered_df)

# Save the updated DataFrame to CSV
df_with_inchikey.to_csv('Analysis&Modeling/Integrated Analysis/1.Life_Stage_Analysis/DataSet/4.LifeStageData-SMILESRetrieved.csv', index=False)
