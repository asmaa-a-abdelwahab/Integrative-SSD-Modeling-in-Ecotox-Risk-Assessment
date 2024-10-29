### README for Species Sensitivity Distribution Modeling: Data Curation Workflows

This repository provides a comprehensive set of data curation workflows tailored for Species Sensitivity Distribution (SSD) modeling in environmental toxicology. The workflows include steps for data standardization, physicochemical property extraction, descriptor calculation, and molecular fingerprint generation, preparing the datasets for further analysis in SSD and predictive modeling studies.

---

## Overview

The following workflows are provided as Python scripts:

- [Overview](#overview)
  - [1. **InChIKey Standardization Workflow (`InChIKey_Standardisation_Workflow.py`)**](#1-inchikey-standardization-workflow-inchikey_standardisation_workflowpy)
  - [2. **Chemical Curation Workflow (`Chemical_Curation_Workflow.py`)**](#2-chemical-curation-workflow-chemical_curation_workflowpy)
  - [3. **Physicochemical Properties Calculation Workflow (`PhysChemProperties_Calculation_Workflow.py`)**](#3-physicochemical-properties-calculation-workflow-physchemproperties_calculation_workflowpy)
  - [4. **Descriptors Calculation Workflow (`Descriptors_Calculation_Workflow.py`)**](#4-descriptors-calculation-workflow-descriptors_calculation_workflowpy)
  - [5. **Fingerprints Calculation Workflow (`Fingerprints_Calculation_Workflow.py`)**](#5-fingerprints-calculation-workflow-fingerprints_calculation_workflowpy)
- [Installation](#installation)
- [Contributing](#contributing)

---

### 1. **InChIKey Standardization Workflow (`InChIKey_Standardisation_Workflow.py`)**

- **Purpose**: Standardizes InChIKeys using multiple data sources including PubChem, OPSIN, NCI, and RDKit. Validates and resolves inconsistencies between sources.
- **Key Features**:
  - Cross-references chemical databases such as PubChem, OPSIN, NCI, and NIST.
  - Regenerates InChIKeys using RDKit to ensure consistency.
  - Addresses variations in chemical names and formatting to find the most accurate InChIKey.

- **Command Line Usage**:
  ```bash
  python InChIKey_Standardisation_Workflow.py <input_csv_file> <output_csv_file>
  ```
  Example:
  ```bash
  python InChIKey_Standardisation_Workflow.py '../DataSet/3.LifeStageData-InvertebratesMerged&SpeciesFiltered.csv' '../DataSet/4.LifeStageData-InChIKeyRetrieved.csv'
  ```

---

### 2. **Chemical Curation Workflow (`Chemical_Curation_Workflow.py`)**

- **Purpose**: Cleans and curates chemical datasets by removing duplicates, filtering invalid structures, and standardizing key chemical properties and identifiers.
- **Key Features**:
  - Filters out inorganics, counterions, biologics, and mixtures based on structural checks.
  - Performs structural cleaning and normalization using RDKit’s built-in tools.
  - Validates stereochemistry and flags discrepancies for manual curation.
  - Identifies activity cliffs using structural similarity and bioactivity thresholds with clustering techniques.

- **Command Line Usage**:
  ```bash
  python Chemical_Curation_Workflow.py <input_csv_file> <output_csv_file> <sdf_output_file>
  ```
  Example:
  ```bash
  python Chemical_Curation_Workflow.py '../DataSet/4.LifeStageData-InChIKeyRetrieved.csv' '../DataSet/5.LifeStageData-CompoundsCurated.csv' '../DataSet/5.LifeStageData-CompoundsCurated.sdf'
  ```

---

### 3. **Physicochemical Properties Calculation Workflow (`PhysChemProperties_Calculation_Workflow.py`)**

- **Purpose**: Extracts physicochemical properties for a set of chemical compounds using OPERA and PubChem APIs.
- **Key Features**:
  - Fetches essential properties such as molecular weight, LogP, water solubility, and polar surface area.
  - Utilizes the OPERA and PubChem APIs for comprehensive property extraction.
  - Supports large datasets with efficient parallel processing.

- **Command Line Usage**:
  ```bash
  python PhysChemProperties_Calculation_Workflow.py <input_sdf_file> <output_csv_file>
  ```
  Example:
  ```bash
  python PhysChemProperties_Calculation_Workflow.py '../DataSet/5.LifeStageData-CompoundsCurated.sdf' '../DataSet/6.LifeStageData-PhysicoChemicalProperties.csv'
  ```

---

### 4. **Descriptors Calculation Workflow (`Descriptors_Calculation_Workflow.py`)**

- **Purpose**: Computes a variety of molecular descriptors using RDKit, capturing structural and electronic features of the compounds.
- **Key Features**:
  - Calculates constitutional, topological, electronic, hybrid, and fragment-based descriptors.
  - Provides detailed information on molecular size, complexity, electronic distribution, and branching.
  - Facilitates descriptor-based modeling for QSAR studies and predictive toxicology.

- **Command Line Usage**:
  ```bash
  python Descriptors_Calculation_Workflow.py <input_sdf_file> <output_csv_file>
  ```
  Example:
  ```bash
  python Descriptors_Calculation_Workflow.py '../DataSet/5.LifeStageData-CompoundsCurated.sdf' '../DataSet/6.LifeStageData-Descriptors.csv'
  ```

---

### 5. **Fingerprints Calculation Workflow (`Fingerprints_Calculation_Workflow.py`)**

- **Purpose**: Generates various types of molecular fingerprints using RDKit, essential for machine learning and similarity-based modeling tasks.
- **Key Features**:
  - Computes Morgan, MACCS, AtomPair, Torsion, and RDKit-based fingerprints.
  - Saves each fingerprint type into separate CSV files for efficient data management.
  - Captures different aspects of molecular structure and reactivity for SSD modeling.

- **Command Line Usage**:
  ```bash
  python Fingerprints_Calculation_Workflow.py <input_sdf_file> <output_folder>
  ```
  Example:
  ```bash
  python Fingerprints_Calculation_Workflow.py '../DataSet/5.LifeStageData-CompoundsCurated.sdf' '../DataSet/'
  ```

  **Details**:
  - Reads chemical structures from an SDF file.
  - Generates multiple fingerprints including Morgan, MACCS, AtomPair, Torsion, and RDKit-based fingerprints.
  - Saves each type of fingerprint in a separate CSV file with consistent naming and indexing for easy reference.

---

## Installation

1. **Dependencies**: Install the required dependencies using:
   ```bash
   pip install -r requirements.txt
   ```

2. **Workflow Execution**: The workflows are designed as standalone Python scripts. You can run these workflows directly via the command line.

3. **Input Data**: Each workflow expects specific file formats for input, such as `.sdf` files for chemical structures or `.csv` files for chemical metadata. Ensure that the input files are correctly formatted as per the requirements specified in each workflow.

---

## Contributing

Contributions to this project are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

