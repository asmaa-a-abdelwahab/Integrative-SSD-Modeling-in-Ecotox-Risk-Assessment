import argparse
import requests
import pandas as pd
from rdkit import Chem
from io import StringIO
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class ServerBusyException(Exception):
    """Custom exception to handle server busy responses."""

    pass


class PhysChemPropertiesCalculator:
    """
    A class to calculate physicochemical properties for molecules from SMILES strings
    using OPERA and PubChem APIs.
    """

    def __init__(self):
        """Initialize the PhysChemPropertiesCalculator with predefined URLs and property names."""
        self.opera_url = "https://opera.saferworldbydesign.com/opera/global-descriptors"
        self.pubchem_base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
        self.properties = (
            "MolecularFormula,MolecularWeight,CanonicalSMILES,IsomericSMILES,InChI,InChIKey,"
            "IUPACName,Title,XLogP,ExactMass,MonoisotopicMass,TPSA,Complexity,Charge,"
            "HBondDonorCount,HBondAcceptorCount,RotatableBondCount,HeavyAtomCount,"
            "IsotopeAtomCount,AtomStereoCount,DefinedAtomStereoCount,UndefinedAtomStereoCount,"
            "BondStereoCount,DefinedBondStereoCount,UndefinedBondStereoCount,CovalentUnitCount"
        )

    def read_smiles_from_sdf(self, sdf_file):
        """Reads SMILES strings from an SDF file and returns them as a DataFrame."""
        logger.info(f"Reading SMILES from SDF file: {sdf_file}")
        suppl = Chem.SDMolSupplier(sdf_file)
        smiles_list = [Chem.MolToSmiles(mol) for mol in suppl if mol is not None]
        logger.info(f"Extracted {len(smiles_list)} SMILES from the SDF file.")
        return pd.DataFrame(smiles_list, columns=["Cleaned_SMILES"])

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(
            (requests.exceptions.RequestException, ServerBusyException)
        ),
    )
    def get_opera_data(self, smiles):
        """Fetches chemical properties from OPERA for a given SMILES string."""
        logger.info(f"Fetching OPERA data for SMILES: {smiles}")
        headers = {"Content-Type": "application/json"}
        props = {}

        try:
            response = requests.post(
                self.opera_url, json={"smiles": smiles}, headers=headers
            )
            logger.debug(f"Raw OPERA response for {smiles}: {response.text}")

            # Check if the response is successful
            if response.status_code == 200:
                try:
                    data = response.json()
                except ValueError:
                    logger.error(f"Failed to parse JSON response for SMILES: {smiles}")
                    return props  # Return empty properties if JSON parsing fails

                if data and "data" in data and smiles in data["data"]:
                    raw_props = data["data"].get(smiles, {})
                    for prop, value in raw_props.items():
                        if prop not in [
                            "MolWeight",
                            "TopoPolSurfAir",
                            "nbHBdAcc",
                            "ndHBdDon",
                        ]:
                            new_key = self._rename_opera_property(prop)
                            props[new_key] = value
                else:
                    logger.warning(f"No valid data found for SMILES: {smiles}")
            else:
                logger.error(
                    f"Failed to fetch OPERA data for {smiles}. Status code: {response.status_code}"
                )
                if response.status_code == 503:
                    logger.warning("Server is busy. Retrying...")
                    raise ServerBusyException("Server is busy. Retrying...")
                else:
                    logger.error(
                        f"Server error for SMILES: {smiles} - Status code: {response.status_code}"
                    )

            return props
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error while fetching OPERA data for {smiles}: {e}")
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error while fetching OPERA data for {smiles}: {e}"
            )
            return props  # Return empty properties in case of unexpected errors

    def _rename_opera_property(self, prop):
        """
        Rename OPERA property keys to more human-readable names.

        Args:
            prop (str): Original OPERA property key.

        Returns:
            str: Renamed property key.
        """
        rename_dict = {
            "AD_FUB": "Fraction Unbound (FUB)",
            "AD_LogD": "LogD",
            "AD_LogP": "LogP",
            "AD_MP": "Melting Point (MP)",
            "AD_VP": "Vapor Pressure (VP)",
            "AD_WS": "Water Solubility (WS)",
            "AD_index_FUB": "Fraction Unbound Index",
            "AD_index_LogD": "LogD Index",
            "AD_index_LogP": "LogP Index",
            "AD_index_MP": "Melting Point Index",
            "AD_index_VP": "Vapor Pressure Index",
            "AD_index_WS": "Water Solubility Index",
            "AD_index_pKa": "pKa Index",
            "AD_pKa": "pKa",
            "CombDipolPolariz": "Combined Dipole Polarizability",
            "Conf_index_FUB": "Confidence Fraction Unbound",
            "Conf_index_LogD": "Confidence LogD Index",
            "Conf_index_LogP": "Confidence LogP Index",
            "Conf_index_MP": "Confidence Melting Point Index",
            "Conf_index_VP": "Confidence Vapor Pressure Index",
            "Conf_index_WS": "Confidence Water Solubility Index",
            "Conf_index_pKa": "Confidence pKa Index",
            "FUB_pred": "Predicted Fraction Unbound",
            "FUB_predRange": "Predicted Fraction Unbound Range",
            "LogD55_pred": "Predicted LogD at pH 5.5",
            "LogD55_predRange": "Predicted LogD at pH 5.5 Range",
            "LogD74_pred": "Predicted LogD at pH 7.4",
            "LogD74_predRange": "Predicted LogD at pH 7.4 Range",
            "LogP_pred": "Predicted LogP",
            "LogP_predRange": "Predicted LogP Range",
            "LogVP_pred": "Predicted Vapor Pressure",
            "LogWS_pred": "Predicted Water Solubility",
            "MP_pred": "Predicted Melting Point",
            "MP_predRange": "Predicted Melting Point Range",
            "MolWeight": "Molecular Weight",
            "MolarRefract": "Molar Refractivity",
            "Sp3Sp2HybRatio": "SP3 SP2 Hybridization Ratio",
            "TopoPolSurfAir": "Topological Polar Surface Area",
            "VP_predRange": "Predicted Vapor Pressure Range",
            "WS_predRange": "Predicted Water Solubility Range",
            "ionization": "Ionization Potential",
            "nbAromAtom": "Number Aromatic Atoms",
            "nbAtoms": "Number Atoms",
            "nbC": "Number Carbon Atoms",
            "nbHBdAcc": "Number HBond Acceptors",
            "nbHeavyAtoms": "Number Heavy Atoms",
            "nbHeteroRing": "Number Heterocyclic Rings",
            "nbLipinskiFailures": "Number Lipinski Rule Failures",
            "nbN": "Number Nitrogen Atoms",
            "nbO": "Number Oxygen Atoms",
            "nbRing": "Number Rings",
            "nbRotBd": "Number Rotatable Bonds",
            "ndHBdDon": "Number HBond Donors",
            "pKa_a_pred": "Predicted Acidic pKa",
            "pKa_a_predRange": "Predicted Acidic pKa Range",
            "pKa_b_pred": "Predicted Basic pKa",
            "pKa_b_predRange": "Predicted Basic pKa Range",
        }
        return rename_dict.get(
            prop, prop
        )  # Return renamed key if available, else original key

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(
            (requests.exceptions.RequestException, ServerBusyException)
        ),
    )
    def fetch_pubchem_properties(self, canonical_smiles):
        """Fetches chemical properties from PubChem for a given canonical SMILES string."""
        logger.info(f"Fetching PubChem data for SMILES: {canonical_smiles}")
        url = f"{self.pubchem_base_url}/smiles/{canonical_smiles}/property/{self.properties}/CSV"
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raises an exception for HTTP errors
            data = pd.read_csv(StringIO(response.text), sep=",")
            data.rename(columns=self._rename_pubchem_property, inplace=True)
            return data
        except requests.exceptions.RequestException as e:
            logger.error(
                f"Error fetching PubChem properties for SMILES {canonical_smiles}: {e}"
            )
            if response.status_code == 503:
                raise ServerBusyException("Server is busy. Retrying...")
            return pd.DataFrame()

    def _rename_pubchem_property(self, prop):
        """
        Rename PubChem property keys to more human-readable names.

        Args:
            prop (str): Original PubChem property key.

        Returns:
            str: Renamed property key.
        """
        pubchem_rename_dict = {
            "MolecularFormula": "Molecular Formula",
            "MolecularWeight": "Molecular Weight",
            "CanonicalSMILES": "Canonical SMILES",
            "IsomericSMILES": "Isomeric SMILES",
            "InChI": "InChI",
            "InChIKey": "InChI Key",
            "IUPACName": "IUPAC Name",
            "Title": "Title",
            "XLogP": "XLogP",
            "ExactMass": "Exact Mass",
            "MonoisotopicMass": "Monoisotopic Mass",
            "TPSA": "Topological Polar Surface Area",
            "Complexity": "Complexity",
            "Charge": "Charge",
            "HBondDonorCount": "HBond Donor Count",
            "HBondAcceptorCount": "HBond Acceptor Count",
            "RotatableBondCount": "Rotatable Bond Count",
            "HeavyAtomCount": "Heavy Atom Count",
            "IsotopeAtomCount": "Isotope Atom Count",
            "AtomStereoCount": "Atom Stereo Count",
            "DefinedAtomStereoCount": "Defined Atom Stereo Count",
            "UndefinedAtomStereoCount": "Undefined Atom Stereo Count",
            "BondStereoCount": "Bond Stereo Count",
            "DefinedBondStereoCount": "Defined Bond Stereo Count",
            "UndefinedBondStereoCount": "Undefined Bond Stereo Count",
            "CovalentUnitCount": "Covalent Unit Count",
            "Volume3D": "Volume 3D",
            "XStericQuadrupole3D": "X Steric Quadrupole 3D",
            "YStericQuadrupole3D": "Y Steric Quadrupole 3D",
            "ZStericQuadrupole3D": "Z Steric Quadrupole 3D",
            "FeatureCount3D": "Feature Count 3D",
            "FeatureAcceptorCount3D": "Feature Acceptor Count 3D",
            "FeatureDonorCount3D": "Feature Donor Count 3D",
            "FeatureAnionCount3D": "Feature Anion Count 3D",
            "FeatureCationCount3D": "Feature Cation Count 3D",
            "FeatureRingCount3D": "Feature Ring Count 3D",
            "FeatureHydrophobeCount3D": "Feature Hydrophobe Count 3D",
            "ConformerModelRMSD3D": "Conformer Model RMSD 3D",
            "EffectiveRotorCount3D": "Effective Rotor Count 3D",
            "ConformerCount3D": "Conformer Count 3D",
        }
        return pubchem_rename_dict.get(
            prop, prop
        )  # Return renamed key if available, else original key

    def process_dataframe_for_opera(self, df, column_name):
        """Processes the DataFrame by adding OPERA properties for each SMILES string."""
        logger.info("Processing DataFrame for OPERA properties.")
        unique_smiles = df[column_name].unique()
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self.get_opera_data, smiles): smiles
                for smiles in unique_smiles
            }
            for future in as_completed(futures):
                smiles = futures[future]
                props = future.result()
                if props:
                    for prop_name, prop_value in props.items():
                        df.loc[df[column_name] == smiles, prop_name] = prop_value
        return df

    def process_dataframe_for_pubchem(self, df, column_name):
        """Processes the DataFrame by adding PubChem properties for each SMILES string."""
        logger.info("Processing DataFrame for PubChem properties.")
        unique_smiles = df[column_name].unique()
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self.fetch_pubchem_properties, smiles): smiles
                for smiles in unique_smiles
            }
            for future in as_completed(futures):
                smiles = futures[future]
                props = future.result()
                if not props.empty:
                    for prop_name, prop_value in props.iloc[0].items():
                        df.loc[df[column_name] == smiles, prop_name] = prop_value
        return df

    def read_sdf_and_extract_properties(self, sdf_file, output_csv):
        """Main method to read SMILES from an SDF file, process them for both OPERA and PubChem properties, and return the final DataFrame."""
        logger.info(f"Starting the process of reading SDF and extracting properties.")
        df = self.read_smiles_from_sdf(sdf_file)  # Read SMILES strings from SDF file
        df = self.process_dataframe_for_opera(
            df, "Cleaned_SMILES"
        )  # Add OPERA properties
        df = self.process_dataframe_for_pubchem(
            df, "Cleaned_SMILES"
        )  # Add PubChem properties
        df.to_csv(output_csv, index=False)  # Save the final DataFrame
        logger.info(f"Properties extraction completed and saved to '{output_csv}'.")
        return df


def main():
    parser = argparse.ArgumentParser(
        description="Extract Physico-Chemical properties using OPERA and PubChem APIs."
    )
    parser.add_argument("sdf_file", help="Path to the input SDF file")
    parser.add_argument("output_csv", help="Path to save the resulting CSV file")

    args = parser.parse_args()

    calculator = PhysChemPropertiesCalculator()
    df = calculator.read_sdf_and_extract_properties(args.sdf_file, args.output_csv)
    print(df)


if __name__ == "__main__":
    main()


"""
Ex: python PhysChemProperties_Calculation_Workflow.py '..\DataSet\5.LifeStageData-CompoundsCurated.sdf' '..\DataSet\5.LifeStageData-PhysicoChemicalProperties.csv'
"""
