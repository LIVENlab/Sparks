
# Sparks

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

**Sparks** is an open-source Python library designed to bridge the gap between energy system modeling outputs and Life Cycle Assessment (LCA) workflows. It specifically facilitates the integration of [Calliope](https://calliope.readthedocs.io/) energy system model results with LCA data, creating standardized inputs for [ENBIOS](https://pypi.org/project/enbios/) analysis.

## Key Features

- 🔄 **Data Integration**: Seamlessly connects Calliope energy system outputs with LCA inventory data
- 📊 **Unit Conversion**: Automatically adapts energy units to match LCA database requirements
- 🌍 **Regional Aggregation**: Supports both national and subnational regional analysis
- 🏗️ **Hierarchical Structure**: Creates dendrogram-like hierarchies for ENBIOS compatibility
- ⚡ **Performance Optimized**: Efficient data processing with pandas and caching mechanisms
- 🔧 **Flexible Configuration**: Supports multiple databases and custom conversion factors

## Installation

### Prerequisites

- Python 3.11 or higher
- Brightway2 database with ecoinvent or similar LCA data
- Required Python packages (see `environment.yaml`)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/LIVENlab/Sparks.git
   cd Sparks
   ```

2. **Install dependencies using conda (recommended):**
   ```bash
   conda env create -f environment.yaml
   conda activate sparks
   ```

3. **Alternative pip installation:**
   ```bash
   pip install -r requirements_fixed.txt
   ```

## Quick Start

### Basic Usage

```python
from Sparks.util.base import SoftLink

# Initialize Sparks with your data directory and Brightway project
enbios_mod = SoftLink(
    path='path/to/your/data',
    bw_project='your_brightway_project_name'
)

# Preprocess and transform the data
enbios_mod.preprocess()

# Generate ENBIOS input file
enbios_mod.data_for_ENBIOS(
    smaller_vers=False,
    path_save='output.json'
)
```

### Complete Workflow Example

```python
from Sparks.util.base import SoftLink
import pandas as pd

# 1. Prepare your data directory with:
#    - basefile.xlsx (configuration file)
#    - energy system output files (CSV format)

# 2. Initialize Sparks
enbios_mod = SoftLink(
    path='testing/data_test',
    bw_project='Hydrogen_SEEDS'
)

# 3. Preprocess data (maps LCI data with energy system data)
enbios_mod.preprocess()

# 4. Check preprocessed units
print(enbios_mod.preprocessed_units)

# 5. Generate ENBIOS input
enbios_mod.data_for_ENBIOS(
    smaller_vers=False,
    path_save='test.json'
)
```

## Input Requirements

### Basefile Structure

The core configuration file (`basefile.xlsx`) must contain a **Processors** sheet with the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| `Processor` | Technology name in energy system model | `"CHP_hydrogen"` |
| `Region` | Geographic region identifier | `"Spain"` or `"Spain_1"` |
| `@SimulationCarrier` | Energy carrier type | `"HEAT"`, `"ELECTRICITY"` |
| `ParentProcessor` | Hierarchical parent technology | `"Power_Generation"` |
| `@SimulationToEcoinventFactor` | Unit conversion factor | `1.0` or `0.001` |
| `@Ecoinvent_key_code` | Brightway database activity code | `"CHP_hydrogen_2050"` |
| `File_source` | Source CSV file name | `"flow_out_sum.csv"` |

### Data File Requirements

- **Format**: CSV files with comma separation
- **Naming**: Main data column must match the filename
  - File: `flow_out_sum.csv` → Column: `flow_out_sum`
- **Required Columns**: 
  - `techs`: Technology identifiers
  - `locs`: Location identifiers  
  - `carriers`: Energy carrier types
  - `energy_value`: Main energy values
  - `spores` or `scenario`: Scenario identifiers

### Directory Structure

```
your_data_directory/
├── basefile.xlsx          # Configuration file
├── flow_out_sum.csv       # Energy flow data
├── energy_cap.csv         # Capacity data
├── nameplate_capacity.csv # Nameplate capacity data
└── other_data_files.csv   # Additional data files
```

## Core Components

### 1. Cleaner Class

The `Cleaner` class handles data preprocessing and validation:

```python
from Sparks.util.preprocess.cleaner import Cleaner

cleaner = Cleaner(
    motherfile='basefile.xlsx',
    file_handler={'flow_out_sum.csv': 'path/to/file.csv'},
    national=False,  # Set to True for national aggregation
    specify_database=False,
    additional_columns=['optimization', 'year']
)
```

**Key Methods:**
- `preprocess_data()`: Main preprocessing pipeline
- `adapt_units()`: Unit conversion and adaptation
- `_verify_national()`: Regional granularity validation

### 2. SoftLink Class

The main interface for transforming data into ENBIOS format:

```python
from Sparks.util.preprocess.SoftLink import SoftLinkCalEnb

softlink = SoftLinkCalEnb(
    calliope=energy_data,
    mother_data=basefile_activities,
    sublocations=region_list,
    motherfile='basefile.xlsx'
)
```

**Key Methods:**
- `run()`: Execute the transformation pipeline
- `_generate_scenarios()`: Create scenario structures
- `_get_methods()`: Extract LCA methods from basefile

### 3. Data Classes

Core data structures for LCA activities and scenarios:

```python
from Sparks.generic.generic_dataclass import BaseFileActivity, Scenario

# Base activity with LCA data
activity = BaseFileActivity(
    name="CHP_hydrogen",
    region="Spain",
    carrier="HEAT",
    parent="Power_Generation",
    code="CHP_hydrogen_2050",
    factor=1.0
)

# Scenario structure
scenario = Scenario(
    name="scenario1",
    activities=[activity_scenario1, activity_scenario2]
)
```

## Configuration Options

### National vs. Subnational Analysis

```python
# For national-level aggregation
cleaner = Cleaner(
    motherfile='basefile.xlsx',
    file_handler=file_dict,
    national=True  # Aggregates subnational regions
)

# For subnational detail
cleaner = Cleaner(
    motherfile='basefile.xlsx', 
    file_handler=file_dict,
    national=False  # Preserves regional detail
)
```

### Database Specification

```python
# When using multiple databases
cleaner = Cleaner(
    motherfile='basefile.xlsx',
    file_handler=file_dict,
    specify_database=True  # Enables database column validation
)
```

### Additional Columns

```python
# For creating subscenarios
cleaner = Cleaner(
    motherfile='basefile.xlsx',
    file_handler=file_dict,
    additional_columns=['optimization', 'year', 'policy']
)
# Creates scenarios like: "scenario1_optimization1_year2025_policyA"
```

## Advanced Features

### Regional Pattern Detection

Sparks automatically detects regional naming patterns:

- **National**: `"Spain"`, `"Germany"`
- **Subnational**: `"Spain_Madrid"`, `"Germany_Berlin"`

### Unit Conversion Pipeline

1. **Input Validation**: Ensures data structure consistency
2. **Technology Filtering**: Maps energy technologies to LCA activities
3. **Unit Adaptation**: Applies conversion factors from basefile
4. **Regional Aggregation**: Optionally combines subnational data

### Performance Optimizations

- **Activity Caching**: Reduces database queries for repeated activities
- **Batch Processing**: Efficient handling of large datasets
- **Memory Management**: Optimized pandas operations

## Troubleshooting

### Common Issues

1. **Empty DataFrame after dropna()**
   - Check merge keys in basefile
   - Verify column names match between files
   - Ensure no NaN values in essential columns

2. **Missing Activities**
   - Verify Brightway database codes exist
   - Check database project selection
   - Review basefile activity mappings

3. **Regional Aggregation Errors**
   - Set `national=True` for subnational data
   - Check regional naming conventions
   - Verify `_verify_national()` warnings

### Debug Mode

```python
# Enable detailed logging
import warnings
warnings.filterwarnings('always')

# Check data at each step
print(f"Data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"NaN summary: {df.isna().sum()}")
```

## API Reference

### Cleaner Class

| Method | Description | Returns |
|--------|-------------|---------|
| `preprocess_data()` | Main preprocessing pipeline | `pd.DataFrame` |
| `adapt_units()` | Unit conversion and adaptation | `pd.DataFrame` |
| `_verify_national()` | Regional validation | `None` |
| `_input_checker()` | Data structure validation | `pd.DataFrame` |

### SoftLink Class

| Method | Description | Returns |
|--------|-------------|---------|
| `run()` | Execute transformation | `dict` |
| `_generate_scenarios()` | Create scenario structures | `list` |
| `_get_methods()` | Extract LCA methods | `dict` |

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use Sparks in your research, please cite:

```bibtex
@software{sparks2024,
  title={Sparks: Energy System to LCA Integration Tool},
  author={de Tomás, Alexander and LIVENlab},
  year={2024},
  url={https://github.com/LIVENlab/Sparks}
}
```

## Contact

- **Maintainer**: [Alexander de Tomás](mailto:alexander.detomas@uab.cat)
- **Organization**: [LIVENlab](https://github.com/LIVENlab)
- **Repository**: [https://github.com/LIVENlab/Sparks](https://github.com/LIVENlab/Sparks)

## Acknowledgments

- [Calliope](https://calliope.readthedocs.io/) - Energy system modeling framework
- [ENBIOS](https://pypi.org/project/enbios/) - LCA analysis platform
- [Brightway2](https://brightway.dev/) - LCA database framework
- [pandas](https://pandas.pydata.org/) - Data manipulation library


