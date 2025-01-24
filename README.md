# Introduction
ENBIOS is an analytical framework for the estimation of environmental impacts of energy system configurations as calculated by energy system optimization models. More information about ENBIOS can be found [here](https://pypi.org/project/enbios/)

## Generating JSON input file
The first step to run ENBIOS is to generate the input files. You can do this by running the sparks module. Sparks will transform your excel inputs into the ENBIOS dendrogram.  For an example of the ENBIOS excel basefile you can go to the ENBIOS github site [here](https://github.com/LIVENlab/Sparks/blob/main/testing/data_test/basefile.xlsx) 

You can generate the JSON input file by running the demo_test.py script. If you are a Brightway2 user SPARKs will crete a BW2 project for you on the run in the 




![Example](example.png)

## Install requirements
```console
pip install enbios==2.2.10
pip install -r /path/to/requirements.txt
```
## Create inputs for enbios

```python
from Sparks.util.base import SoftLink
enbios_mod = SoftLink(r'C:\Users\Administrator\Downloads\flow_out_sum (1).csv',
                      r'C:\Users\Administrator\Documents\Alex\BASEFILE\basefile_off_heat.xlsx',
                       'github', 
                      'ecoinvent')
enbios_mod.preprocess(subregions=False)
enbios_mod.data_for_ENBIOS(smaller_vers=True, path_save=r'data/test.json')
```
Please note that this repository is currently undergoing several updates.
The following functions have not been implemented yet:
- Double counting

### Output Example
```json
{
    "adapters": [
        {
            "adapter_name": "brightway-adapter",
            "config": {
                "bw_project": "github"
            },
            "methods": {
                "GWP1000": [
                    "ReCiPe 2016 v1.03, midpoint (H)",
                    "climate change",
                    "global warming potential (GWP1000)"
                ],
                "LOP": [
                    "ReCiPe 2016 v1.03, midpoint (H)",
                    "land use",
                    "agricultural land occupation (LOP)"
                ],
                "WCP": [
                    "ReCiPe 2016 v1.03, midpoint (H)",
                    "water use",
                    "water consumption potential (WCP)"
                ],
                "FEP": [
                    "ReCiPe 2016 v1.03, midpoint (H)",
                    "eutrophication: freshwater",
                    "freshwater eutrophication potential (FEP)"
                ],
                "SOP": [
                    "ReCiPe 2016 v1.03, midpoint (H)",
                    "material resources: metals/minerals",
                    "surplus ore potential (SOP)"
                ]
            }
        }
    ],
    "hierarchy": {
        "name": "Energysystem",
        "aggregator": "sum",
        "children": [
            {
                "name": "Generation",
                "aggregator": "sum",
                "children": [
                    {
                        "name": "Electricity_generation",
                        "aggregator": "sum",
                        "children": [
                            {
                                "name": "wind_onshore__electricity___PRT",
                                "adapter": "bw",
                                "config": {
                                    "code": "81174ec2c20931c1a36f65c654bbd11e"
                                }
                            },
                            {
                                "name": "hydro_run_of_river__electricity___PRT",
                                "adapter": "bw",
                                "config": {
                                    "code": "b870e3d3ddd5b634d016940064b27532"
                                }
                            },
                            {
                                "name": "hydro_reservoir__electricity___PRT",
                                "adapter": "bw",
                                "config": {
                                    "code": "c868c4688fbf78f5ca3787ac3d83312b"
                                }
                            },
                            {
                                "name": "ccgt__electricity___PRT",
                                "adapter": "bw",
                                "config": {
                                    "code": "4833b6352dfe15c95ae46fd280371cd3"
                                }
                            },
                            {
                                "name": "chp_biofuel_extraction__electricity___PRT",
                                "adapter": "bw",
                                "config": {
                                    "code": "1952fb157a18463b0028629917057bcf"
                                }
                            },
                            {
                                "name": "open_field_pv__electricity___PRT",
                                "adapter": "bw",
                                "config": {
                                    "code": "206c2e2c30f45d47ea2e9a3701f8ecc5"
                                }
                            },
                            {
                                "name": "existing_wind__electricity___PRT",
                                "adapter": "bw",
                                "config": {
                                    "code": "81174ec2c20931c1a36f65c654bbd11e"
                                }
                            },
                            {
                                "name": "existing_pv__electricity___PRT",
                                "adapter": "bw",
                                "config": {
                                    "code": "206c2e2c30f45d47ea2e9a3701f8ecc5"
                                }
                            },
                            {
                                "name": "roof_mounted_pv__electricity___PRT",
                                "adapter": "bw",
                                "config": {
                                    "code": "1d7a3591e7a3d033a25e0eaf699bb50b"
                                }
                            },
                            {
                                "name": "chp_wte_back_pressure__electricity___PRT",
                                "adapter": "bw",
                                "config": {
                                    "code": "07d4d5423ba8379958ee341ada3a6ef5"
                                }
                            },
                            {
                                "name": "chp_methane_extraction__electricity___PRT",
                                "adapter": "bw",
                                "config": {
                                    "code": "a0d0ab65435d0487d33b4be142483d76"
                                }
                            },
                            {
                                "name": "waste_supply__waste___PRT",
                                "adapter": "bw",
                                "config": {
                                    "code": "07d4d5423ba8379958ee341ada3a6ef5"
                                }
                            }
                        ]
                    },
                    {
                        "name": "Thermal_generation",
                        "aggregator": "sum",
                        "children": [
                            {
                                "name": "chp_biofuel_extraction__heat___PRT",
                                "adapter": "bw",
                                "config": {
                                    "code": "413bc4617794c6e8b07038dbeca64adb"
                                }
                            },
                            {
                                "name": "chp_wte_back_pressure__heat___PRT",
                                "adapter": "bw",
                                "config": {
                                    "code": "bbcc24a62a7a425e708fa8967f0df2ee"
                                }
                            },
                            {
                                "name": "chp_methane_extraction__heat___PRT",
                                "adapter": "bw",
                                "config": {
                                    "code": "a3039e0a35ed472774d6ed66262c67f3"
                                }
                            },
                            {
                                "name": "biofuel_boiler__heat___PRT",
                                "adapter": "bw",
                                "config": {
                                    "code": "413bc4617794c6e8b07038dbeca64adb"
                                }
                            },
                            {
                                "name": "methane_boiler__heat___PRT",
                                "adapter": "bw",
                                "config": {
                                    "code": "a3039e0a35ed472774d6ed66262c67f3"
                                }
                            }
                        ]
                    }
                ]
            },
            {
                "name": "Storage",
                "aggregator": "sum",
                "children": [
                    {
                        "name": "Electricity_storage",
                        "aggregator": "sum",
                        "children": [
                            {
                                "name": "battery__electricity___PRT",
                                "adapter": "bw",
                                "config": {
                                    "code": "5649dd9b7aa34f1a93f87fabfb7ea516"
                                }
                            },
                            {
                                "name": "pumped_hydro__electricity___PRT",
                                "adapter": "bw",
                                "config": {
                                    "code": "309ff1f1194f2b5ea34d8c077103b1e3"
                                }
                            }
                        ]
                    },
                    {
                        "name": "Thermal_storage",
                        "aggregator": "sum",
                        "children": [
                            {
                                "name": "heat_storage_big__heat___PRT",
                                "adapter": "bw",
                                "config": {
                                    "code": "07db2c419abe9cffeb033aa0bed7f0ba"
                                }
                            },
                            {
                                "name": "heat_storage_small__heat___PRT",
                                "adapter": "bw",
                                "config": {
                                    "code": "14e0227e764a4a098e5fd34525ccaf3f"
                                }
                            },
                            {
                                "name": "methane_storage__methane___PRT",
                                "adapter": "bw",
                                "config": {
                                    "code": "da586847dea594ccd75188627f456d03"
                                }
                            }
                        ]
                    }
                ]
            },
            {
                "name": "Conversions",
                "aggregator": "sum",
                "children": [
                    {
                        "name": "Conversions",
                        "aggregator": "sum",
                        "children": [
                            {
                                "name": "biofuel_to_diesel__diesel___PRT",
                                "adapter": "bw",
                                "config": {
                                    "code": "3a6586952eff793738eef72743703e90"
                                }
                            },
                            {
                                "name": "biofuel_to_liquids__diesel___PRT",
                                "adapter": "bw",
                                "config": {
                                    "code": "3a6586952eff793738eef72743703e90"
                                }
                            },
                            {
                                "name": "biofuel_to_methane__methane___PRT",
                                "adapter": "bw",
                                "config": {
                                    "code": "7d82a9b0a28dc30058260b5c68fbad95"
                                }
                            },
                            {
                                "name": "biofuel_to_methanol__methanol___PRT",
                                "adapter": "bw",
                                "config": {
                                    "code": "df54a237474430370bb580c2451d405d"
                                }
                            }
                        ]
                    }
                ]
            },
            {
                "name": "Imports",
                "aggregator": "sum",
                "children": [
                    {
                        "name": "Imports",
                        "aggregator": "sum",
                        "children": [
                            {
                                "name": "el_import__electricity___ESP",
                                "adapter": "bw",
                                "config": {
                                    "code": "1373b63314d03c42313a372bca1bd648"
                                }
                            }
                        ]
                    }
                ]
            }
        ]
    },
    "scenarios": [
        {
            "name": "0",
            "nodes": {
                "battery__electricity___PRT": [
                    "kilogram",
                    44134.16580352975
                ],
                "biofuel_boiler__heat___PRT": [
                    "megajoule",
                    8443951.632043561
                ],
                "biofuel_to_diesel__diesel___PRT": [
                    "kilogram",
                    226141452.37025702
                ],
                "biofuel_to_liquids__diesel___PRT": [
                    "kilogram",
                    26776.950120781345
                ],
                "biofuel_to_methane__methane___PRT": [
                    "cubic meter",
                    240964492.04181156
                ],
                "biofuel_to_methanol__methanol___PRT": [
                    "kilogram",
                    776498342.6121527
                ],
                "ccgt__electricity___PRT": [
                    "kilowatt hour",
                    23211.124751186187
                ],
                "chp_biofuel_extraction__electricity___PRT": [
                    "kilowatt hour",
                    158962.74804907836
                ],
                "chp_biofuel_extraction__heat___PRT": [
                    "megajoule",
                    941785.54105752
                ],
                "chp_methane_extraction__electricity___PRT": [
                    "kilowatt hour",
                    114044.33980325895
                ],
                "chp_methane_extraction__heat___PRT": [
                    "megajoule",
                    147518.0183137525
                ],
                "chp_wte_back_pressure__electricity___PRT": [
                    "kilowatt hour",
                    558172222.4720111
                ],
                "chp_wte_back_pressure__heat___PRT": [
                    "megajoule",
                    94617.4063238808
                ],
                "el_import__electricity___ESP": [
                    "kilowatt hour",
                    4829248004.934705
                ],
                "existing_pv__electricity___PRT": [
                    "kilowatt hour",
                    1706580759.1818905
                ],
                "existing_wind__electricity___PRT": [
                    "kilowatt hour",
                    16975519247.851534
                ],
                "heat_storage_big__heat___PRT": [
                    "unit",
                    27.51374451906663
                ],
                "heat_storage_small__heat___PRT": [
                    "unit",
                    676.8670753979804
                ],
                "hydro_reservoir__electricity___PRT": [
                    "kilowatt hour",
                    7034481419.237691
                ],
                "hydro_run_of_river__electricity___PRT": [
                    "kilowatt hour",
                    8217716366.150942
                ],
                "methane_boiler__heat___PRT": [
                    "megajoule",
                    847413.1280672245
                ],
                "methane_storage__methane___PRT": [
                    "unit",
                    21.26796402586542
                ],
                "open_field_pv__electricity___PRT": [
                    "kilowatt hour",
                    100417087878.80571
                ],
                "pumped_hydro__electricity___PRT": [
                    "kilowatt hour",
                    9180843870.236614
                ],
                "roof_mounted_pv__electricity___PRT": [
                    "kilowatt hour",
                    842273.1367537137
                ],
                "waste_supply__waste___PRT": [
                    "kilowatt hour",
                    2572222223.3733234
                ],
                "wind_onshore__electricity___PRT": [
                    "kilowatt hour",
                    57514890465.62103
                ]
            }
        }
    ]
}

```
