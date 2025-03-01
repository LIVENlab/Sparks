
# Sparks

## Introduction

[Sparks](https://github.com/LIVENlab/Sparks) is an open-source repository designed to facilitate the integration of life cycle assessment (LCA) data with energy system configurations generated by [Calliope](https://calliope.readthedocs.io/en/stable/#). It creates an input file for [ENBIOS](https://pypi.org/project/enbios/), adapting units, structuring data into a dendrogram tree-like format, and optionally aggregating regions and subregions.

## Installation

Ensure you have Python installed, then install the required dependencies:

```sh
pip install -r requirements.txt
```

## Usage

### Generating Inputs for ENBIOS

```python
from Sparks.util.base import SoftLink

enbios_mod = SoftLink(r'testing/data_test', 'BW25_project_name')

enbios_mod.preprocess()
enbios_mod.data_for_ENBIOS(smaller_vers=False, path_save=r'test.json')
```

### Input Requirements

The `SoftLink` class requires a `path` pointing to a directory containing the relevant energy system files. The main input is a **base file** (Excel format), which defines:

- Energy technologies
- Inventory data
- File sources
- LCA methods
- Conversion factors
- Dendrogram structure

A structured example can be found in the [testing/data_test](https://github.com/LIVENlab/Sparks/tree/sparks-times/testing/data_test).

### Workflow Example

See an example of the workflow [here](https://github.com/LIVENlab/Sparks/blob/main/demo.ipynb)


### Contact

- [Alexander de Tomás](mailto:alexander.detomas@uab.cat)


