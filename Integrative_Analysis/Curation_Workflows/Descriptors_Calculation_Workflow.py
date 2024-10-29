import argparse
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, rdmolops, AllChem
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DescriptorCalculator:
    """
    A class to calculate various molecular descriptors using RDKit,
    excluding properties already extracted by PhysChemPropertiesExtractor.
    """

    def __init__(self, sdf_file, output_file):
        """
        Initialize the DescriptorCalculator with an SDF file.

        Args:
            sdf_file (str): Path to the SDF file containing chemical structures.
            output_file (str): Path to save the resulting CSV file with descriptors.
        """
        self.sdf_file = sdf_file
        self.output_file = output_file
        self.smiles_df = self.read_smiles_from_sdf()
        self.descriptors_df = None

    def read_smiles_from_sdf(self):
        """
        Reads SMILES strings from an SDF file and returns them as a DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the SMILES strings.
        """
        logger.info(f"Reading SMILES from SDF file: {self.sdf_file}")
        suppl = Chem.SDMolSupplier(self.sdf_file)
        smiles_list = []

        for mol in suppl:
            if mol is not None:
                smiles = Chem.MolToSmiles(mol)
                smiles_list.append(smiles)

        logger.info(f"Read {len(smiles_list)} SMILES from the SDF file.")
        return pd.DataFrame(smiles_list, columns=["SMILES"])

    def calculate_constitutional_descriptors(self, mol):
        """
        Calculate constitutional descriptors for a given RDKit molecule.

        Args:
            mol (rdkit.Chem.rdchem.Mol): RDKit molecule object.

        Returns:
            dict: Dictionary of constitutional descriptors.
        """
        logger.debug("Calculating constitutional descriptors.")

        # These descriptors describe the basic structure of a molecule,
        # including the number of atoms, bonds, and aromaticity.
        # Important for understanding molecular size and complexity,
        # which can influence the bioavailability and toxicity of compounds.

        # Count the number of aromatic atoms manually
        num_aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())

        return {
            "NumAtoms": mol.GetNumAtoms(),  # Total number of atoms in the molecule
            "NumBonds": mol.GetNumBonds(),  # Total number of bonds in the molecule
            "NumAromaticAtoms": num_aromatic_atoms,  # Number of aromatic atoms, related to stability and reactivity
            "NumRotatableBonds": Descriptors.NumRotatableBonds(
                mol
            ),  # Flexibility of the molecule, affecting interaction with biological targets
        }

    def calculate_topological_descriptors(self, mol):
        """
        Calculate topological descriptors for a given RDKit molecule.

        Args:
            mol (rdkit.Chem.rdchem.Mol): RDKit molecule object.

        Returns:
            dict: Dictionary of topological descriptors.
        """
        logger.debug("Calculating topological descriptors.")

        # Topological descriptors capture the connectivity of the molecule,
        # providing insights into the molecular shape and branching, which
        # are crucial for understanding how a molecule interacts with its environment.

        # Calculate the Wiener Index
        wiener_index = self.calculate_wiener_index(mol)

        # Calculate the Balaban J Index
        balaban_j_index = self.calculate_balaban_j_index(mol)

        # Calculate the Harary Index
        harary_index = self.calculate_harary_index(mol)

        # Calculate the Zagreb Indices
        zagreb_indices = self.calculate_zagreb_indices(mol)

        return {
            "WienerIndex": wiener_index,  # Sum of all distances between atoms, related to molecular branching
            "BalabanJ": balaban_j_index,  # A measure of molecular connectivity and branching
            "HararyIndex": harary_index,  # Sum of reciprocal distances, indicating molecular compactness
            "FirstZagrebIndex": zagreb_indices[
                "FirstZagrebIndex"
            ],  # Reflects molecular branching
            "SecondZagrebIndex": zagreb_indices[
                "SecondZagrebIndex"
            ],  # Relates to the interaction of adjacent atoms
        }

    def calculate_wiener_index(self, mol):
        """
        Calculate the Wiener index for a given RDKit molecule.

        Args:
            mol (rdkit.Chem.rdchem.Mol): RDKit molecule object.

        Returns:
            float: Wiener index.
        """
        # The Wiener Index is a topological descriptor that sums the distances
        # between all pairs of atoms. It is related to molecular branching and
        # can be predictive of physical properties like boiling point.
        distance_matrix = rdmolops.GetDistanceMatrix(mol)
        wiener_index = (
            distance_matrix.sum() / 2
        )  # Sum of all distances divided by 2 to avoid double-counting
        return wiener_index

    def calculate_balaban_j_index(self, mol):
        """
        Calculate the Balaban J index for a given RDKit molecule.

        Args:
            mol (rdkit.Chem.rdchem.Mol): RDKit molecule object.

        Returns:
            float: Balaban J index.
        """
        # The Balaban J index is another measure of molecular connectivity and branching,
        # providing insights into molecular shape, which can influence how the molecule
        # interacts with biological targets.
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()

        if num_bonds == 0 or num_atoms <= 2:
            return 0.0

        distance_matrix = rdmolops.GetDistanceMatrix(mol)
        diameter = np.max(distance_matrix)  # Maximum distance in the distance matrix

        balaban_j = (
            (num_bonds / (num_atoms - 1))
            * (1 / diameter)
            * np.sum(1 / distance_matrix[distance_matrix > 0])
        )
        return balaban_j

    def calculate_harary_index(self, mol):
        """
        Calculate the Harary index for a given RDKit molecule.

        Args:
            mol (rdkit.Chem.rdchem.Mol): RDKit molecule object.

        Returns:
            float: Harary index.
        """
        # The Harary Index sums the reciprocals of all distances between atoms,
        # providing a measure of molecular compactness. Compact molecules may have
        # different biological interactions compared to more extended structures.
        distance_matrix = rdmolops.GetDistanceMatrix(mol)
        with np.errstate(
            divide="ignore", invalid="ignore"
        ):  # Handle division by zero and invalid values
            reciprocal_distances = np.where(distance_matrix > 0, 1 / distance_matrix, 0)
        harary_index = (
            np.sum(reciprocal_distances) / 2
        )  # Sum of reciprocals divided by 2 to avoid double-counting
        return harary_index

    def calculate_zagreb_indices(self, mol):
        """
        Calculate the first and second Zagreb indices for a given RDKit molecule.

        Args:
            mol (rdkit.Chem.rdchem.Mol): RDKit molecule object.

        Returns:
            dict: Dictionary containing the first and second Zagreb indices.
        """
        # The Zagreb Indices reflect the molecular branching and are related
        # to the stability and reactivity of the molecule. These indices can
        # help predict how a molecule behaves in different environments.

        # First Zagreb Index (M1): Sum of the squares of the degrees of the vertices
        first_zagreb_index = 0
        for atom in mol.GetAtoms():
            degree = atom.GetDegree()
            first_zagreb_index += degree**2

        # Second Zagreb Index (M2): Sum of the products of degrees of adjacent vertices
        second_zagreb_index = 0
        for bond in mol.GetBonds():
            degree1 = bond.GetBeginAtom().GetDegree()
            degree2 = bond.GetEndAtom().GetDegree()
            second_zagreb_index += degree1 * degree2

        return {
            "FirstZagrebIndex": first_zagreb_index,
            "SecondZagrebIndex": second_zagreb_index,
        }

    def calculate_electronic_descriptors(self, mol):
        """
        Calculate electronic descriptors for a given RDKit molecule.

        Args:
            mol (rdkit.Chem.rdchem.Mol): RDKit molecule object.

        Returns:
            dict: Dictionary of electronic descriptors.
        """
        logger.debug("Calculating electronic descriptors.")

        # Electronic descriptors, like partial charges, provide information on
        # how electrons are distributed in the molecule. These descriptors are
        # crucial for understanding how the molecule will interact with biological
        # systems, as charge distribution affects binding to proteins and other targets.
        return {
            "MaxAbsPartialCharge": Descriptors.MaxAbsPartialCharge(
                mol
            ),  # Maximum absolute partial charge
            "MinAbsPartialCharge": Descriptors.MinAbsPartialCharge(
                mol
            ),  # Minimum absolute partial charge
            "MaxPartialCharge": Descriptors.MaxPartialCharge(
                mol
            ),  # Maximum partial charge
            "MinPartialCharge": Descriptors.MinPartialCharge(
                mol
            ),  # Minimum partial charge
        }

    def calculate_bcut_descriptors(self, mol):
        """
        Calculate BCUT descriptors for a given RDKit molecule.

        Args:
            mol (rdkit.Chem.rdchem.Mol): RDKit molecule object.

        Returns:
            dict: Dictionary of BCUT descriptors.
        """
        logger.debug("Calculating BCUT descriptors.")

        # BCUT descriptors capture both the atomic properties (like charges) and
        # the molecular connectivity. These descriptors help predict how the molecule
        # will interact with biological systems and environmental factors.

        # Generate atomic charges (e.g., Gasteiger charges)
        AllChem.ComputeGasteigerCharges(mol)

        # Retrieve atomic charges
        atomic_charges = np.array(
            [float(atom.GetProp("_GasteigerCharge")) for atom in mol.GetAtoms()]
        )

        # Check for NaNs or infinities in atomic charges
        if not np.all(np.isfinite(atomic_charges)):
            logger.error(
                "Invalid atomic charges (NaNs or infinities) detected in the molecule. Skipping BCUT calculation."
            )
            return {"BCUT2D_MWLOW": np.nan, "BCUT2D_MWUP": np.nan}

        # Build an adjacency matrix
        adjacency_matrix = rdmolops.GetAdjacencyMatrix(mol)

        # Create a diagonal matrix of atomic properties
        atomic_property_matrix = np.diag(atomic_charges)

        # Calculate the BCUT matrix: adjacency matrix weighted by atomic properties
        bcut_matrix = np.dot(atomic_property_matrix, adjacency_matrix)

        # Compute eigenvalues of the BCUT matrix
        try:
            eigenvalues = np.linalg.eigvals(bcut_matrix)
        except np.linalg.LinAlgError as e:
            logger.error(f"Linear algebra error during BCUT calculation: {e}")
            return {"BCUT2D_MWLOW": np.nan, "BCUT2D_MWUP": np.nan}

        # Sort eigenvalues to obtain the low and high values
        bcut_mwlow = np.min(eigenvalues)
        bcut_mwup = np.max(eigenvalues)

        return {
            "BCUT2D_MWLOW": bcut_mwlow,  # Lower eigenvalue of the BCUT matrix, related to molecular interaction properties
            "BCUT2D_MWUP": bcut_mwup,  # Upper eigenvalue of the BCUT matrix, related to molecular interaction properties
        }

    def calculate_hybrid_descriptors(self, mol):
        """
        Calculate hybrid descriptors for a given RDKit molecule.

        Args:
            mol (rdkit.Chem.rdchem.Mol): RDKit molecule object.

        Returns:
            dict: Dictionary of hybrid descriptors.
        """
        logger.debug("Calculating hybrid descriptors.")

        # Hybrid descriptors combine multiple types of molecular properties,
        # like electronic and steric properties, providing a comprehensive
        # view of how the molecule might interact in biological systems.

        # Calculate BCUT descriptors using custom implementation
        bcut_descriptors = self.calculate_bcut_descriptors(mol)

        return {
            "EState_VSA1": Descriptors.EState_VSA1(
                mol
            ),  # Electronic state surface area descriptor
            "EState_VSA2": Descriptors.EState_VSA2(
                mol
            ),  # Another electronic state surface area descriptor
            "BCUT2D_MWLOW": bcut_descriptors["BCUT2D_MWLOW"],  # Lower BCUT eigenvalue
            "BCUT2D_MWUP": bcut_descriptors["BCUT2D_MWUP"],  # Upper BCUT eigenvalue
        }

    def calculate_fragment_based_descriptors(self, mol):
        """
        Calculate fragment-based descriptors for a given RDKit molecule.

        Args:
            mol (rdkit.Chem.rdchem.Mol): RDKit molecule object.

        Returns:
            dict: Dictionary of fragment-based descriptors.
        """
        logger.debug("Calculating fragment-based descriptors.")

        # Fragment-based descriptors count specific substructures, such as
        # aromatic and aliphatic rings, which are important for understanding
        # molecular stability and interaction with biological systems.
        return {
            "NumAromaticRings": Descriptors.NumAromaticRings(
                mol
            ),  # Number of aromatic rings, related to molecular stability and reactivity
            "NumAliphaticRings": Descriptors.NumAliphaticRings(
                mol
            ),  # Number of aliphatic rings, related to molecular flexibility and reactivity
        }

    def calculate_all_descriptors(self, smiles):
        """
        Calculate all descriptors for a given SMILES string.

        Args:
            smiles (str): SMILES string of the molecule.

        Returns:
            dict: Dictionary of all descriptors or None if molecule is invalid.
        """
        logger.info(f"Calculating descriptors for SMILES: {smiles}")
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            logger.warning(f"Invalid SMILES string: {smiles}")
            return None

        descriptors = {}
        descriptors.update(self.calculate_constitutional_descriptors(mol))
        descriptors.update(self.calculate_topological_descriptors(mol))
        descriptors.update(self.calculate_electronic_descriptors(mol))
        descriptors.update(self.calculate_hybrid_descriptors(mol))
        descriptors.update(self.calculate_fragment_based_descriptors(mol))
        print(descriptors)

        return descriptors

    def process_smiles(self):
        """
        Calculate descriptors for each SMILES string in the DataFrame and store the results.

        Returns:
            None
        """
        logger.info("Processing SMILES to calculate descriptors.")
        smiles_list = self.smiles_df["SMILES"]
        descriptors_list = [
            self.calculate_all_descriptors(smiles) for smiles in smiles_list
        ]
        self.descriptors_df = (
            pd.DataFrame(descriptors_list, index=smiles_list)
            .reset_index()
            .rename(columns={"index": "SMILES"})
        )
        logger.info("Descriptor calculation completed.")

    def get_final_dataframe(self):
        """
        Run the full process and return the final DataFrame with SMILES and their descriptors.

        Returns:
            pd.DataFrame: DataFrame containing SMILES and their corresponding descriptors.
        """
        logger.info("Generating final DataFrame with descriptors.")
        self.process_smiles()
        logger.info(f"Saving final DataFrame to {self.output_file}.")
        self.descriptors_df.to_csv(self.output_file, index=False)
        return self.descriptors_df


def main():
    parser = argparse.ArgumentParser(
        description="Calculate molecular descriptors using RDKit from an SDF file."
    )
    parser.add_argument(
        "sdf_file", help="Path to the input SDF file containing chemical structures."
    )
    parser.add_argument(
        "output_file", help="Path to save the resulting CSV file with descriptors."
    )

    args = parser.parse_args()

    calculator = DescriptorCalculator(args.sdf_file, args.output_file)
    final_df = calculator.get_final_dataframe()
    print(final_df)


if __name__ == "__main__":
    main()

"""
Ex: python .\Descriptors_Calculation_Workflow.py '..\DataSet\5.LifeStageData-CompoundsCurated.sdf' '..\DataSet\5.LifeStageData-Descriptors.csv'              
"""
