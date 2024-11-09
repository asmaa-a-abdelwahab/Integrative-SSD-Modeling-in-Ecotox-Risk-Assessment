import os
import pandas as pd
import logging
import sys
import json
import requests
from collections import defaultdict
import pubchempy as pcp
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from requests.exceptions import RequestException, Timeout
from bs4 import BeautifulSoup
from rdkit import Chem


# Clear any existing log handlers to prevent duplication
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("inchi_standardization_log.json"),
        logging.StreamHandler(sys.stdout),
    ],
)


class InChIStandardizer:
    def __init__(self):
        self.counters = defaultdict(int)
        self.inchi_map = {}  # Dictionary to map chemical name/CAS number to InChI/InChIKey
        logging.info("Initialized InChIStandardizer with counters and cache.")

    def format_cas_number(self, cas_number):
        """
        Formats a CAS number in the correct format.
        """
        cas_number = str(cas_number)
        check_digit = cas_number[-1]
        middle_part = cas_number[-3:-1]
        first_part = cas_number[:-3]
        return f"{first_part}-{middle_part}-{check_digit}"

    def format_chemical_name(self, chemical_name):
        """
        Formats the chemical name by removing the last part if it is in parentheses.
        Returns both the formatted chemical name and the last part inside the parentheses (if present).
        """
        smiles_candidate = None
        if " " in chemical_name:
            parts = chemical_name.split(" ")
            last_part = parts[-1]
            if "(" in last_part and ")" in last_part:
                smiles_candidate = last_part.strip("()")
                chemical_name = " ".join(parts[:-1])
                logging.info(
                    f"Formatted chemical name to: {chemical_name} and extracted potential SMILES: {smiles_candidate}"
                )
        return chemical_name, smiles_candidate

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((RequestException, Timeout)),
    )
    def retrieve_inchi_pubchem(self, chemical_name):
        """
        Retrieves the InChI string for a given chemical name from PubChem.
        """
        logging.info(f"Retrieving InChI from PubChem for {chemical_name}.")
        try:
            compounds = pcp.get_compounds(chemical_name, "name")
            inchi = compounds[0].inchi if compounds else None
            return inchi
        except Exception as e:
            logging.error(f"PubChem retrieval failed for {chemical_name}: {e}")
            return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((RequestException, Timeout)),
    )
    def retrieve_inchi_opsin(self, chemical_name):
        """
        Retrieves the InChI string for a given chemical name from OPSIN.
        """
        logging.info(f"Retrieving InChI from OPSIN for {chemical_name}.")
        try:
            url = f"http://opsin.ch.cam.ac.uk/opsin/{chemical_name}.inchi"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.text.strip()
            return None
        except Exception as e:
            logging.error(
                f"Failed to retrieve InChI from OPSIN for {chemical_name}: {e}"
            )
            return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((RequestException, Timeout)),
    )
    def retrieve_inchi_nci(self, chemical_name):
        """
        Retrieves the InChI string for a given chemical name from the NCI (National Cancer Institute) database.
        """
        logging.info(f"Retrieving InChI from NCI for {chemical_name}.")
        try:
            url = f"https://cactus.nci.nih.gov/chemical/structure/{chemical_name}/stdinchi"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.text.strip()
            return None
        except Exception as e:
            logging.error(f"Failed to retrieve InChI from NCI for {chemical_name}: {e}")
            return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((RequestException, Timeout)),
    )
    def retrieve_inchi_nist(self, cas_number):
        """
        Retrieves the InChI string for a given CAS number from the NIST Chemistry WebBook.
        """
        logging.info(f"Retrieving InChI from NIST for CAS {cas_number}.")
        try:
            url = f"https://webbook.nist.gov/cgi/cbook.cgi?ID={cas_number}&Units=SI"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                inchi_label = soup.find("strong", text="IUPAC Standard InChI:")
                if inchi_label:
                    inchi_element = inchi_label.find_next("span")
                    if inchi_element:
                        return inchi_element.text.strip()
            return None
        except Exception as e:
            logging.error(
                f"Failed to retrieve InChI from NIST for CAS {cas_number}: {e}"
            )
            return None

    def validate_inchi(self, inchi):
        """
        Validates an InChI string by converting it into an RDKit molecule.
        """
        logging.info(f"Validating InChI: {inchi}")
        mol = Chem.MolFromInchi(inchi)
        return mol is not None

    def cross_validate_inchi(self, inchi_data):
        """
        Cross-validates InChI data retrieved from multiple sources to enhance reliability.
        """
        valid_inchis = [inchi for inchi in inchi_data.values() if inchi]
        if len(valid_inchis) > 1 and len(set(valid_inchis)) == 1:
            logging.info(
                "Cross-validation succeeded with consistent InChI across sources."
            )
            return valid_inchis[0]
        elif len(valid_inchis) == 1:
            logging.warning("Only one valid InChI found, cross-validation limited.")
            return valid_inchis[0]
        else:
            logging.error("Cross-validation failed, no consistent InChI found.")
            return None

    def standardize_inchi(self, chemical_name, cas_number=None):
        # Check if already in cache (dictionary)
        key = chemical_name or cas_number
        if key in self.inchi_map:
            logging.info(f"Using cached InChI for {key}.")
            return self.inchi_map[key]

        # Simulate retrieval from multiple sources
        sources = {
            "PubChem": self.retrieve_inchi_pubchem(chemical_name),
            "OPSIN": self.retrieve_inchi_opsin(chemical_name),
            "NIST": self.retrieve_inchi_nist(cas_number),
            "NCI": self.retrieve_inchi_nci(chemical_name),
        }

        # Cross-validate InChI
        valid_inchi = next((v for v in sources.values() if v), None)
        if valid_inchi:
            self.inchi_map[key] = valid_inchi  # Cache the result
            logging.info(f"Retrieved and validated InChI: {valid_inchi}")
        else:
            logging.warning(f"No valid InChI found for {key}.")
            self.counters["no_inchi_found"] += 1
            valid_inchi = None

        return valid_inchi

    def log_counts(self):
        """
        Logs the counters for various outcomes during the standardization process.
        """
        with open("inchi_standardization_counts.json", "w") as f:
            json.dump(self.counters, f)
        for key, count in self.counters.items():
            logging.info(f"{key.replace('_', ' ').capitalize()}: {count}")


def process_chemical(chemical, standardizer):
    """
    Process a single chemical entry (chemical name or CAS number).
    Returns the chemical and the standardized InChI.
    """
    chemical_name, cas_number = chemical
    return chemical, standardizer.standardize_inchi(chemical_name, cas_number)


def main(input_file, output_file, delimiter=",", max_workers=5):
    if not os.path.isfile(input_file):
        logging.error(f"Input file not found: {input_file}")
        return

    logging.info(f"Reading input file: {input_file}")
    df = pd.read_csv(input_file, delimiter=delimiter)

    # Initialize the InChIStandardizer
    standardizer = InChIStandardizer()

    # Prepare a unique set of chemical names or CAS numbers
    unique_chemicals = set()
    for _, row in df.iterrows():
        chemical_name = row.get("Chemical Name")
        cas_number = row.get("CAS Number", None)
        key = chemical_name or cas_number
        if key:  # Ensure there is a valid chemical name or CAS number
            unique_chemicals.add((chemical_name, cas_number))

    # Process chemical names and CAS numbers in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_chemical, chemical, standardizer): chemical
            for chemical in unique_chemicals
        }

        for future in as_completed(futures):
            chemical, standardized_inchi = future.result()
            if standardized_inchi:
                standardizer.inchi_map[chemical[0] or chemical[1]] = standardized_inchi

    # Propagate InChI/InChIKey to the main dataframe
    df["Standardized_InChI"] = df.apply(
        lambda row: standardizer.inchi_map.get(
            row.get("Chemical Name") or row.get("CAS Number")
        ),
        axis=1,
    )

    # Save updated DataFrame
    df.to_csv(output_file, index=False)
    logging.info(f"Data successfully saved to: {output_file}")

    # Log final counts
    standardizer.log_counts()


if __name__ == "__main__":
    # Add argument parser for input/output files
    parser = argparse.ArgumentParser(
        description="Process chemical names and CAS numbers to retrieve InChI."
    )
    parser.add_argument(
        "input_file",
        help="Path to the input CSV file containing chemical names and/or CAS numbers",
    )
    parser.add_argument(
        "output_file", help="Path to save the output CSV file with InChI information"
    )
    parser.add_argument(
        "--delimiter", default=",", help="Delimiter for the CSV file (default: ',')"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Maximum number of parallel workers (default: 5)",
    )

    args = parser.parse_args()

    # Run the main function with command-line arguments
    main(args.input_file, args.output_file, args.delimiter, args.max_workers)


"""
Ex: python InChI_Standardisation_Workflow.py '..\DataSet\3.LifeStageData-InvertebratesMerged&SpeciesFiltered.csv' '..\DataSet\4.LifeStageData-InChIRetrieved.csv'
"""
