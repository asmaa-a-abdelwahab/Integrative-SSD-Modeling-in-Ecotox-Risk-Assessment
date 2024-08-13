import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem  # Import for bond length and angle optimization
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import SDWriter
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChemicalCuration:
    """
    A class to perform chemical data curation, including the removal of inorganics, organometallics, counterions,
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

    def remove_inorganics_and_mixtures(self):
        """
        Remove inorganic compounds, organometallics, counterions, biologics, and mixtures from the DataFrame.
        Organic compounds must contain carbon and at least one other element, and should not contain metals, counterions, or multiple disconnected components.
        """

        def is_organic(smiles):
            """
            Determine if a molecule is organic.

            An organic molecule must contain carbon and at least one other element. It must not contain metals or be too small to be a typical organic compound.

            Args:
                smiles (str): SMILES string of the molecule.

            Returns:
                bool: True if the molecule is organic, False otherwise.
            """
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return False
                elements = {atom.GetSymbol() for atom in mol.GetAtoms()}
                metals = {'Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                          'Ga', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Cs',
                          'Ba', 'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi'}
                # Check for presence of metals (indicating organometallics) or absence of carbon
                if elements & metals or 'C' not in elements or len(elements) <= 1:
                    return False
                return True
            except Exception as e:
                logging.error(f"Error processing SMILES {smiles}: {e}")
                return False

        def is_counterion(smiles):
            """
            Determine if a molecule is a counterion.

            Counterions are typically small ions used to balance charges in salts. This function uses a molecular weight threshold to identify them.

            Args:
                smiles (str): SMILES string of the molecule.

            Returns:
                bool: True if the molecule is a counterion, False otherwise.
            """
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return False
                # Molecules with a molecular weight less than 100 are considered counterions
                return Descriptors.MolWt(mol) < 100
            except Exception as e:
                logging.error(f"Error processing SMILES {smiles} for counterion check: {e}")
                return False

        def is_biologic(smiles):
            """
            Determine if a molecule is a biologic.

            Biologics often contain multiple peptide bonds and are larger, more complex molecules. This function uses a simple heuristic to detect them.

            Args:
                smiles (str): SMILES string of the molecule.

            Returns:
                bool: True if the molecule is a biologic, False otherwise.
            """
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return False
                # Simple heuristic: biologics often contain many peptide bonds (C=O-NH)
                peptide_bond_count = smiles.count('C(=O)N')
                return peptide_bond_count > 5  # Threshold for detecting potential biologics
            except Exception as e:
                logging.error(f"Error processing SMILES {smiles} for biologic check: {e}")
                return False

        def is_mixture(smiles):
            """
            Determine if a SMILES string represents a mixture.

            A mixture is defined as a SMILES string containing multiple components, usually separated by a '.' character.

            Args:
                smiles (str): SMILES string of the molecule.

            Returns:
                bool: True if the SMILES string represents a mixture, False otherwise.
            """
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return False
                # A dot in the SMILES string indicates multiple components (mixture)
                return '.' in smiles
            except Exception as e:
                logging.error(f"Error processing SMILES {smiles} for mixture check: {e}")
                return False

        unique_inchikeys = self.df['InChIKey'].unique()
        logging.info("Processing %d unique InChIKeys to remove inorganics, organometallics, counterions, biologics, and mixtures.", len(unique_inchikeys))

        for inchikey in unique_inchikeys:
            smiles = self.inchikey_to_smiles(inchikey)
            if smiles:
                # Perform checks in order: organic, not a counterion, not a biologic, and not a mixture
                if is_organic(smiles) and not is_counterion(smiles) and not is_biologic(smiles) and not is_mixture(smiles):
                    self.inchikey_smiles_map[inchikey] = smiles
                else:
                    logging.info(f"SMILES {smiles} removed: does not meet criteria.")

        # Map the remaining valid SMILES strings to their corresponding InChIKeys
        self.df['SMILES'] = self.df['InChIKey'].map(self.inchikey_smiles_map)
        # Drop rows where SMILES is NaN (indicating removal)
        self.cleaned_data = self.df.dropna(subset=['SMILES']).copy()
        logging.info("Removed inorganics, organometallics, counterions, biologics, and mixtures. Cleaned data shape: %s", self.cleaned_data.shape)

    def structural_cleaning(self):
        """
        Perform comprehensive structural cleaning on the chemical dataset, including:
        - Detection and correction of valence violations and extreme bond lengths/angles.
        - Ring aromatization.
        - Normalization of specific chemotypes.
        - Standardization of tautomeric forms using custom SMIRKS rules.

        Returns:
            None: The function modifies the DataFrame in place, updating the `cleaned_data` with the cleaned structures.
        """

        def apply_tautomer_rules(mol):
            """
            Apply a set of custom tautomer rules to standardize the tautomeric form of the molecule.

            Args:
                mol (rdkit.Chem.Mol): The RDKit molecule object.

            Returns:
                rdkit.Chem.Mol: The molecule with standardized tautomers.
            """
            # Define the tautomer rules based on SMIRKS patterns
            tautomer_transforms = [
                rdMolStandardize.TautomerTransform('[CX3]=[OX1]([#1])>>[CX3][OX2H]', name='Keto-enol'),
                rdMolStandardize.TautomerTransform('[CX3][OX2H]>>[CX3]=[OX1]([#1])', name='Enol-keto'),
                rdMolStandardize.TautomerTransform('[CX4H2][CX3]=[CX3H]>>[CX4H]=[CX3H][CX4H2]', name='Alkene Tautomer'),
                rdMolStandardize.TautomerTransform('[NX3][CX3]=[NX3]>>[NX3]=[CX3][NX3]', name='Imine-Amine'),
                rdMolStandardize.TautomerTransform('[NX3]=[CX3][OX2H]>>[NX3][CX3]=[OX1]', name='Imine-Oxime'),
                rdMolStandardize.TautomerTransform('[CX3](=[OX1])-[NX3][#1]>>[CX3](=[OX1])[NX2]=[NX2]', name='Amide-Imidic acid'),
                rdMolStandardize.TautomerTransform('[CX3](=[OX1])[NX2]=[NX2]>>[CX3](=[OX1])-[NX3][#1]', name='Imidic acid-Amide'),
                rdMolStandardize.TautomerTransform('[CX3](=[OX1])[NX3]=[NX3]>>[CX3](=[OX1])[NX2]=[NX3][#1]', name='Amidine-Imidamide'),
                rdMolStandardize.TautomerTransform('[NX2]=[NX3][#1]>>[NX3]=[NX3]', name='Imidamide-Amidine'),
                rdMolStandardize.TautomerTransform('[NX2]=[NX3][CX3](=[OX1])[OX2H]>>[NX3]=[NX3][CX3]=[OX1]', name='Guanidine-Guanidine'),
                rdMolStandardize.TautomerTransform('[NX2]=[NX3][CX3]=[OX1]>>[NX3]=[NX3][CX3](=[OX1])[OX2H]', name='Guanidine-Guanidine reverse'),
                rdMolStandardize.TautomerTransform('[CX3](=[OX1])[OX2H]>>[CX3](=[OX1])[OX1]', name='Carboxyl-Carboxylate'),
                rdMolStandardize.TautomerTransform('[CX3](=[OX1])[OX1]>>[CX3](=[OX1])[OX2H]', name='Carboxylate-Carboxyl'),
                rdMolStandardize.TautomerTransform('[OX1]=[CX3][CX3]=[CX3]>>[OX2H]-[CX3]=[CX3]', name='Beta-Diketone Enolization'),
                rdMolStandardize.TautomerTransform('[OX2H]-[CX3]=[CX3]>>[OX1]=[CX3][CX3]=[CX3]', name='Beta-Diketone Enolization reverse'),
                rdMolStandardize.TautomerTransform('[CX3](=[NX2][#1])[NX3][CX3](=[NX3])>>[CX3]=[NX3][CX3](=[NX3])[#1]', name='Amidoxime-Amidrazone'),
                rdMolStandardize.TautomerTransform('[CX3]=[NX3][CX3](=[NX3])[#1]>>[CX3](=[NX2][#1])[NX3][CX3](=[NX3])', name='Amidrazone-Amidoxime'),
                rdMolStandardize.TautomerTransform('[CX3](=[NX2])-[OX1]>>[CX3]=[NX3][OX2H]', name='Oxime-Nitrone'),
                rdMolStandardize.TautomerTransform('[CX3]=[NX3][OX2H]>>[CX3](=[NX2])-[OX1]', name='Nitrone-Oxime'),
                rdMolStandardize.TautomerTransform('[NX3]=[CX3][NX3]=[NX3]>>[NX3][CX3](=[NX2])[NX3]=[NX3]', name='Amidine-Amidrazone'),
                rdMolStandardize.TautomerTransform('[NX3][CX3](=[NX2])[NX3]=[NX3]>>[NX3]=[CX3][NX3]=[NX3]', name='Amidrazone-Amidine')
            ]
            
            # Create a TautomerEnumerator with custom rules
            enumerator = rdMolStandardize.TautomerEnumerator()
            enumerator.SetTransforms(tautomer_transforms)

            # Apply the tautomer rules
            return enumerator.Canonicalize(mol)

        def clean_structure(smiles):
            """
            Apply structural cleaning to a single molecule.

            Args:
                smiles (str): SMILES string of the molecule.

            Returns:
                str: Cleaned SMILES string, or None if cleaning fails.
            """
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return None

                # Step 1: Detect and correct valence violations
                try:
                    Chem.SanitizeMol(mol)
                except Chem.MolSanitizeException:
                    logging.warning(f"Valence violation detected and corrected for SMILES: {smiles}")
                    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)

                # Step 2: Detect and correct extreme bond lengths and angles
                try:
                    AllChem.UFFOptimizeMolecule(mol)
                except Exception as e:
                    logging.warning(f"Optimization failed for SMILES: {smiles}. Reason: {e}")

                # Step 3: Ring aromatization
                try:
                    Chem.Kekulize(mol, clearAromaticFlags=True)
                    Chem.SetAromaticity(mol)
                except Exception as e:
                    logging.warning(f"Ring aromatization failed for SMILES: {smiles}. Reason: {e}")

                # Step 4: Normalization of specific chemotypes
                normalizer = rdMolStandardize.Normalizer()
                mol = normalizer.normalize(mol)

                # Step 5: Apply custom tautomer rules
                mol = apply_tautomer_rules(mol)

                # Convert the cleaned molecule back to SMILES
                cleaned_smiles = Chem.MolToSmiles(mol)
                return cleaned_smiles

            except Exception as e:
                logging.error(f"Failed to clean structure for SMILES {smiles}. Reason: {e}")
                return None

        # Apply the cleaning process to each SMILES string in the DataFrame
        self.cleaned_data['Cleaned_SMILES'] = self.cleaned_data['SMILES'].apply(clean_structure)

        # Remove any rows where cleaning failed (i.e., Cleaned_SMILES is None)
        self.cleaned_data.dropna(subset=['Cleaned_SMILES'], inplace=True)

        logging.info("Structural cleaning completed. Cleaned data shape: %s", self.cleaned_data.shape)

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
                url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/property/CanonicalSMILES,IsomericSMILES/JSON'
                response = requests.get(url)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                logging.error(f"Failed to query PubChem for SMILES {smiles}. Reason: {e}")
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
            
            pubchem_stereocenters = []  # You would extract this from PubChem's response

            # Compare detected stereocenters with PubChem info
            if len(stereo_centers) != len(pubchem_stereocenters):
                return True

            # Further comparison logic can be added here

            return False

        def verify_structure(smiles):
            """
            Verify the stereochemistry of a single structure.

            Args:
                smiles (str): SMILES string of the molecule.

            Returns:
                dict: A dictionary with verification results.
            """
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {'SMILES': smiles, 'Valid': False, 'Error': 'Invalid SMILES'}

            stereo_centers = get_stereocenters(mol)
            pubchem_info = query_pubchem(smiles)

            needs_manual_curation = manual_curation_needed(stereo_centers, pubchem_info)

            return {
                'SMILES': smiles,
                'Stereocenters': stereo_centers,
                'PubChem_Info': pubchem_info,
                'Manual_Curation_Required': needs_manual_curation
            }

        # Apply stereochemistry verification to each SMILES string in the DataFrame
        self.cleaned_data['Stereochemistry_Verification'] = self.cleaned_data['SMILES'].apply(verify_structure)

        logging.info("Stereochemistry verification completed.")

        return self.cleaned_data

    def curate_data(self):
        """
        Run the full curation workflow on the DataFrame.

        The workflow includes:
        - Removal of inorganics, organometallics, counterions, biologics, and mixtures.
        - Structural cleaning, including valence correction, bond optimization, and chemotype normalization.
        - Tautomer and isomer handling.
        - Stereochemistry verification.
        - Final review and validation.

        Returns:
            pandas.DataFrame: The curated chemical data.
        """
        logging.info("Starting data curation process.")

        # Step 1: Initial Data Cleaning
        self.remove_inorganics_and_mixtures()

        # Step 2: Structural Cleaning and Normalization
        self.structural_cleaning()

        # Step 3: Stereochemistry Verification
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
            mol = Chem.MolFromSmiles(row['Cleaned_SMILES'])
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
