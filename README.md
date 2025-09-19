# Sparks

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Welcome to Sparks!

**Sparks** is your open-source Python toolkit for connecting energy system modeling results with Life Cycle Assessment (LCA) workflows. If you use [Calliope](https://calliope.readthedocs.io/) for energy system modeling and want to analyze your results with [ENBIOS](https://pypi.org/project/enbios/), Sparks makes the process smooth and standardized.

---

## Why Sparks?

Bridging energy system outputs and LCA data can be tricky. Sparks automates the  conversions, data mapping, and regional aggregation—so you can focus on insights, not data wrangling. All the outputs can be directly used in [ENBIOS](https://pypi.org/project/enbios/), following its hierarchical structure.

---

## Key Features

- 🔄 **Effortless Data Integration**: Connects ESM outputs with LCA inventory data in a snap.
- 📊 **Automatic Unit Conversion**: Sparks adapts energy units for you in a flexible way.
- 🌍 **Flexible Regional Aggreagtion**: Analyze at national or subnational levels.
- 🏗️ **Hierarchical Data Structure**: Outputs are ready for [ENBIOS](https://pypi.org/project/enbios/), with dendrogram-like hierarchies.
- ⚡ **Optimized Performance**: Fast processing with pandas and smart caching.
- 🔧 **Customizable**: Supports multiple databases and your own conversion factors.

---

## Getting Started

### Prerequisites

- Python 3.11 or newer
- A Brightway2 database (with ecoinvent or similar LCA data)
- Required Python packages (see `environment.yaml`)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/LIVENlab/Sparks.git
   cd Sparks
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
sparks_object = SoftLink(
    path='testing/data_test',
    bw_project='Hydrogen_SEEDS'
)

# 3. Preprocess data (maps LCI data with energy system data)
sparks_object.preprocess()

# 4. Check preprocessed units
print(sparks_object.preprocessed_units)

# 5. Generate ENBIOS input
sparks_object.data_for_ENBIOS(
    smaller_vers=False,
    path_save='test.json'
)
```

## Input Requirements

### Basefile Structure

The core configuration file (`basefile.xlsx`) must contain a **Processors** sheet with the following columns:

| Column                         | Description                                                    | Example                   |
|--------------------------------|----------------------------------------------------------------|---------------------------|
| `Processor`                    | Technology name in energy system model                         | `"CHP_hydrogen"`          |
| `Region`                       | Geographic region identifier                                   | `"Spain"` or `"Spain_1"`  |
| `@SimulationCarrier`           | Energy carrier type                                            | `"HEAT"`, `"ELECTRICITY"` |
| `ParentProcessor`              | Hierarchical parent technology                                 | `"Power_Generation"`      |
| `@SimulationToEcoinventFactor` | Unit conversion factor                                         | `1.0` or `0.001`          |
| `@Ecoinvent_key_code`          | Brightway database activity code                               | `"CHP_hydrogen_2050"`     |
| `File_source`                  | Source CSV file name                                           | `"flow_out_sum.csv"`      |
| `geo_loc`                      | Distinguish betwen onsite-offsite, operation or infrastructure | `"onsite"`                |

See the [Pandera](https://pandera.readthedocs.io/en/stable/index.html) validations schemas [here](https://github.com/LIVENlab/Sparks/blob/main/Sparks/generic/basefile_schema.py)

### Energy Data File Requirements

- **Format**: CSV files with comma separation
- **Naming**: Main data column (energy value) must match the filename
  - File: `flow_out_sum.csv` → Column: `flow_out_sum`
- **Required Columns**: 
  - `techs`: Technology identifiers (they must match the `Processor` in basefile)
  - `locs`: Location identifiers
  - `energy_value`: Main energy values
  
- **Optional Columns**:
  - `carriers`: Energy carrier types
  - - `spores` or `scenario`: Scenario identifiers
  - 
### Directory Structure

```
your_data_directory/
├── basefile.xlsx          # Configuration file
├── flow_out_sum.csv       # Energy flow data
├── energy_cap.csv         # Capacity data
├── nameplate_capacity.csv # Nameplate capacity data
└── other_data_files.csv   # Additional data files
```



## Advanced Features

### Regional Pattern Detection

Sparks automatically detects regional naming patterns. An aggregated analysis can be activated using :
```python
sparks_object.preprocess(national=True)
```
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


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use Sparks in your research, please cite:

```bibtex
@software{sparks2024,
  title={Sparks: Energy System to LCA Integration Tool},
  author={de Tomás, Alexander},
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
- [Brightway25](https://brightway.dev/) - LCA database framework
- [pandas](https://pandas.pydata.org/) - Data manipulation library


