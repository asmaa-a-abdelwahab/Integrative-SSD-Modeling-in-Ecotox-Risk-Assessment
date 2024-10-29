import os
import csv
import argparse
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys, RDKFingerprint
import logging
import sys

# Clear any existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Reconfigure logging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Example usage
logger = logging.getLogger(__name__)


class FingerprintCalculator:
    """
    Processes molecular fingerprints using RDKit's FingerprintGenerator, formatting data for machine learning models
    and saving each type to separate CSV files directly after computation to ensure memory efficiency. This approach is
    particularly useful in environmental toxicology, where analyzing large datasets of chemical compounds is common.
    """

    def __init__(self, sdf_file, output_folder):
        """
        Initialize the calculator with paths to the input SDF file and the output directory for fingerprint files.
        Args:
            sdf_file (str): Path to the SDF file containing the chemical structures.
            output_folder (str): Path to the directory where fingerprint files will be saved.
        """
        self.sdf_file = sdf_file
        self.output_folder = output_folder
        os.makedirs(
            self.output_folder, exist_ok=True
        )  # Ensure the output directory exists
        # Initialize fingerprint generators with specific configurations.
        self.morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=2048
        )
        self.atom_pair_generator = rdFingerprintGenerator.GetAtomPairGenerator()
        self.torsion_generator = rdFingerprintGenerator.GetTopologicalTorsionGenerator()
        logging.info(
            "Initialized fingerprint generators and ensured output directory exists."
        )

    def molecule_supplier(self):
        """
        Generator to yield molecules one at a time from the SDF file, ensuring efficient memory usage.
        """
        with Chem.SDMolSupplier(self.sdf_file, sanitize=True) as suppl:
            for mol in suppl:
                if (
                    mol and mol.GetNumAtoms() > 0
                ):  # Check that molecule is not None and has atoms
                    yield mol
                else:
                    logging.warning(
                        "Skipped a molecule due to being None or having no atoms."
                    )

    def calculate_fingerprints(self, mol):
        """
        Calculate molecular fingerprints using configured generators and convert them to integer arrays.
        Each fingerprint type provides a different perspective on the molecular structure and potential biological activity,
        making them crucial for species sensitivity distribution (SSD) modeling in toxicological assessments.
        """
        try:
            smiles = Chem.MolToSmiles(mol)  # SMILES representation of the molecule
            return {
                "SMILES": smiles,
                "Morgan": self.morgan_generator.GetFingerprint(
                    mol
                ).ToBitString(),  # Circular fingerprint capturing local molecular environment
                "MACCS": MACCSkeys.GenMACCSKeys(
                    mol
                ).ToBitString(),  # 166 key bits representing presence/absence of certain chemical substructures
                "AtomPair": self.atom_pair_generator.GetFingerprint(
                    mol
                ).ToBitString(),  # Describes connections between pairs of atoms
                "Torsion": self.torsion_generator.GetFingerprint(
                    mol
                ).ToBitString(),  # Captures torsional angles to reflect molecular 3D aspects
                "RDKitFP": RDKFingerprint(
                    mol
                ).ToBitString(),  # A generic hash-based fingerprint provided by RDKit
            }
        except Exception as e:
            logging.error(
                f"Failed to process fingerprints for molecule due to an error: {e}"
            )
            return None

    def process_fingerprints(self):
        """
        Process each molecule to calculate fingerprints and directly append them to separate CSV files by fingerprint type.
        This method handles data efficiently by writing each entry as processed to minimize memory usage.
        """
        header_info = {
            key: ["SMILES"] + [f"{key}_{i}" for i in range(2048)]
            for key in ["Morgan", "MACCS", "AtomPair", "Torsion", "RDKitFP"]
        }
        file_handles = {
            key: open(
                os.path.join(self.output_folder, f"6.{key}_fingerprints.csv"),
                "a",
                newline="",
            )
            for key in header_info
        }
        writers = {key: csv.writer(file) for key, file in file_handles.items()}

        for key, writer in writers.items():
            writer.writerow(header_info[key])  # Write headers for each fingerprint file

        for mol in self.molecule_supplier():
            fingerprints = self.calculate_fingerprints(mol)
            if fingerprints:
                for key, value in fingerprints.items():
                    if key != "SMILES":  # Only process actual fingerprint data
                        writers[key].writerow(
                            [fingerprints["SMILES"]] + list(map(int, value))
                        )
        logging.info(f"All fingerprints are calculated and saved in the output folder")

        for file in file_handles.values():
            file.close()  # Close files after writing to ensure data integrity


def main():
    parser = argparse.ArgumentParser(
        description="Calculate molecular fingerprints using RDKit."
    )
    parser.add_argument(
        "sdf_file", help="Path to the input SDF file containing chemical structures."
    )
    parser.add_argument(
        "output_folder", help="Path to save the resulting CSV files with fingerprints."
    )

    args = parser.parse_args()

    calculator = FingerprintCalculator(args.sdf_file, args.output_folder)
    calculator.process_fingerprints()


if __name__ == "__main__":
    main()


"""
Ex:  python .\Fingerprinys_Calculation_Workflow.py '..\DataSet\5.LifeStageData-CompoundsCurated.sdf' '..\DataSet\'
"""
