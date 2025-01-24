# Introduction
ENBIOS is an analytical framework for the estimation of environmental impacts of energy system configurations as calculated by energy system optimization models. More information about ENBIOS can be found [here](https://pypi.org/project/enbios/)

Sparks is an open source repository to soft-link LCA data and energy system configurations from [Calliope](https://calliope.readthedocs.io/en/stable/#), generating an input file for ENBIOS.


## Install requirements
```console
pip install -r /path/to/requirements.txt

```

## Create inputs for enbios

```python
from Sparks.util.base import SoftLink

enbios_mod = SoftLink(r'testing/data_test',
                    'BW25_project name')

enbios_mod.preprocess()
enbios_mod.data_for_ENBIOS(smaller_vers=False, path_save=r'test.json')
```

The *SoftLink* class requires a path to a folder with different files. First, you need to include the basefile (see an example [here](https://github.com/LIVENlab/Sparks/tree/sparks-times/testing/data_test). It's an Excel file describing the energy technologies, inventory data, file sources, methods, conversion factors and the dendrogram.

run the *preprocess* function will transform the data into an enbios like-file, while also adapting the units according to the conversion factors.


## License
[![Creative Commons License](https://i.creativecommons.org/l/by-sa/4.0/88x31.png)](https://creativecommons.org/licenses/by-sa/4.0/)



