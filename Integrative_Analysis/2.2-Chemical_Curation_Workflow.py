import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import SDWriter
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChemicalCuration:
    """
    A class to perform chemical data curation, including the removal of inorganics, organometallics, and counterions,
    structure normalization, handling tautomers and isomers, and final validation.
    """

    def __init__(self, df):
        """
        Initialize the ChemicalCuration class with a DataFrame.

        Args:
            df (pandas.DataFrame): DataFrame containing chemical data with at least an 'InChIKey' column.
        """
        self.df = df
        self.cleaned_data = None
        self.inchikey_smiles_map = {}
        logging.info("ChemicalCuration initialized with DataFrame of shape %s", df.shape)

    def remove_inorganics_and_mixtures(self):
        """
        Remove inorganic compounds, organometallics, counterions, biologics, and mixtures from the DataFrame.
        Organic compounds must contain carbon and at least one other element, and should not contain metals or counterions.
        """
        def is_organic(smiles):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            elements = {atom.GetSymbol() for atom in mol.GetAtoms()}
            metals = {'Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                      'Ga', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Cs',
                      'Ba', 'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi'}
            # Check for the presence of metals (organometallics) or the absence of carbon
            if elements & metals or 'C' not in elements or len(elements) <= 1:
                return False
            return True

        def is_counterion(smiles):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            # Identify small molecules or ions typically used as counterions (e.g., Na+, Cl-)
            return Chem.Descriptors.MolWt(mol) < 100  # Example threshold for small ions

        unique_inchikeys = self.df['InChIKey'].unique()
        logging.info("Processing %d unique InChIKeys to remove inorganics, organometallics, counterions, biologics, and mixtures.", len(unique_inchikeys))

        for inchikey in unique_inchikeys:
            smiles = self.inchikey_to_smiles(inchikey)
            if smiles and is_organic(smiles) and not is_counterion(smiles):
                self.inchikey_smiles_map[inchikey] = smiles

        # Map the InChIKeys to their corresponding SMILES and filter out inorganics/mixtures
        self.df['SMILES'] = self.df['InChIKey'].map(self.inchikey_smiles_map)
        self.cleaned_data = self.df.dropna(subset=['SMILES']).copy()
        logging.info("Removed inorganics, organometallics, counterions, biologics, and mixtures. Cleaned data shape: %s", self.cleaned_data.shape)

    def inchikey_to_smiles(self, inchikey):
        """
        Convert an InChIKey to SMILES by querying an external chemical database.

        Args:
            inchikey (str): The InChIKey to convert.

        Returns:
            str: The corresponding SMILES string, or None if retrieval fails.
        """
        try:
            url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchikey}/property/CanonicalSMILES/TXT'
            response = requests.get(url)
            response.raise_for_status()
            smiles = response.text.strip()
            logging.info("Retrieved SMILES for InChIKey %s", inchikey)
            return smiles
        except Exception as e:
            logging.error("Failed to retrieve SMILES for InChIKey %s: %s", inchikey, e)
            return None

    def normalize_structures(self):
        """
        Normalize chemical structures by sanitizing molecules and preserving aromaticity.
        """
        def normalize_smiles(smiles):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            Chem.SanitizeMol(mol)  # Ensure the molecule is clean
            # Skip Kekulization to preserve aromaticity
            return Chem.MolToSmiles(mol, kekuleSmiles=False)

        self.cleaned_data['Normalized SMILES'] = self.cleaned_data['SMILES'].apply(normalize_smiles)
        logging.info("Normalized chemical structures. Data shape after normalization: %s", self.cleaned_data.shape)

    def handle_tautomers_and_isomers(self):
        """
        Detect and standardize tautomers and isomers using RDKit's MolStandardize.
        """
        def standardize_tautomer(smiles):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return None
                enumerator = rdMolStandardize.TautomerEnumerator()
                tautomer = enumerator.Canonicalize(mol)
                return Chem.MolToSmiles(tautomer)
            except Exception as e:
                logging.error("Failed to standardize tautomer for SMILES %s: %s", smiles, e)
                return None

        def handle_isomers(smiles):
            """
            Handle stereoisomers by generating canonical SMILES that include stereochemistry.
            """
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return None
                # Generate the canonical SMILES with stereochemistry included
                return Chem.MolToSmiles(mol, isomericSmiles=True)
            except Exception as e:
                logging.error("Failed to handle isomer for SMILES %s: %s", smiles, e)
                return None

        # Standardize tautomers first
        self.cleaned_data['Standardized SMILES'] = self.cleaned_data['Normalized SMILES'].apply(standardize_tautomer)
        logging.info("Standardized tautomers. Data shape after tautomer standardization: %s", self.cleaned_data.shape)

        # Then handle stereoisomers
        self.cleaned_data['Isomeric SMILES'] = self.cleaned_data['Standardized SMILES'].apply(handle_isomers)
        logging.info("Handled isomers. Data shape after isomer handling: %s", self.cleaned_data.shape)

    def validate_and_finalize(self):
        """
        Final validation of the curated dataset, including:
        - Removing duplicate SMILES.
        - Ensuring all SMILES strings are valid.
        - Cross-referencing with external databases.
        """
        # Re-check for duplicates after standardization and isomer handling
        self.cleaned_data.drop_duplicates(subset=['Isomeric SMILES'], inplace=True)

        # Validate that all SMILES are valid molecules
        def is_valid_smiles(smiles):
            return Chem.MolFromSmiles(smiles) is not None

        self.cleaned_data = self.cleaned_data[self.cleaned_data['Isomeric SMILES'].apply(is_valid_smiles)]

        # Cross-reference with external databases (e.g., PubChem) for additional validation
        def cross_reference_smiles(smiles):
            try:
                url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/cids/TXT'
                response = requests.get(url)
                return response.status_code == 200
            except:
                return False

        self.cleaned_data = self.cleaned_data[self.cleaned_data['Isomeric SMILES'].apply(cross_reference_smiles)]
        logging.info("Final validation complete. Total valid entries: %d", len(self.cleaned_data))

    def curate_data(self):
        """
        Run the full curation workflow on the DataFrame.

        Returns:
            pandas.DataFrame: The curated chemical data.
        """
        logging.info("Starting data curation process.")
        # Step 1: Initial Data Cleaning
        self.remove_inorganics_and_mixtures()

        # Step 2: Structural Normalization
        self.normalize_structures()

        # Step 3: Handle Tautomers and Isomers
        self.handle_tautomers_and_isomers()

        # Step 4: Final Review and Validation
        self.validate_and_finalize()

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
            mol = Chem.MolFromSmiles(row['Isomeric SMILES'])
            if mol is not None:
                mol.SetProp("InChIKey", row["InChIKey"])
                writer.write(mol)
        writer.close()
        logging.info("SDF file generation complete.")

if __name__ == "__main__":
    # Load the input DataFrame from a CSV file
    df_with_inchikey = pd.read_csv('Analysis&Modeling/Integrated Analysis/1.Life_Stage_Analysis/DataSet/4.LifeStageData-InChIKeyRetrieved.csv')
    
    # Create an instance of the ChemicalCuration class
    curator = ChemicalCuration(df_with_inchikey)
    
    # Run the data curation process
    curated_df = curator.curate_data()
    
    # Save the curated DataFrame to a CSV file
    curated_df.to_csv('Analysis&Modeling/Integrated Analysis/1.Life_Stage_Analysis/DataSet/5.LifeStageData-CompoundsCurated.csv', index=False)
    
    # Generate an SDF file from the curated data
    curator.generate_sdf('Analysis&Modeling/Integrated Analysis/1.Life_Stage_Analysis/DataSet/5.LifeStageData-CompoundsCurated.sdf')
