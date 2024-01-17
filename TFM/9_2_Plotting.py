import pandas as pd
import numpy as np
# Copied from enbios' input
from collections import OrderedDict
import json


hierarchy = {
    "Energysystem": {
        "Generation": {
            "Electricity_generation": [
                "hydro_run_of_river__electricity___PRT",
                "hydro_reservoir__electricity___PRT",
                "ccgt__electricity___PRT",
                "chp_biofuel_extraction__electricity___PRT",
                "chp_wte_back_pressure__electricity___PRT",
                "chp_methane_extraction__electricity___PRT",
                "waste_supply__waste___PRT"
            ],
            "Thermal_generation": [
                "biofuel_supply__biofuel___PRT",
                "biofuel_boiler__heat___PRT",
                "methane_boiler__heat___PRT"
            ]
        },
        "Storage": {
            "Electricity_storage": [
                "battery__electricity___PRT",
                "pumped_hydro__electricity___PRT"
            ],
            "Thermal_storage": [
                "heat_storage_big__heat___PRT",
                "heat_storage_small__heat___PRT",
                "methane_storage__methane___PRT"
            ]
        },
        "Conversions": {
            "Conversions": [
                "biofuel_to_diesel__diesel___PRT",
                "biofuel_to_methane__methane___PRT",
                "biofuel_to_methanol__methanol___PRT"
            ]
        },
        "Imports": {
            "Imports": ["el_import__electricity___ESP-sink"]
        }
    }}

class Hierarchy:

    def __init__(self,hierarchy_tree: dict):
        self.hierarchy=hierarchy
        self.df_hierarchy=self.hierarchy_to_df(hierarchy_tree)
        self._energy_pivot=self.join_impacts_input()

    @staticmethod
    def hierarchy_to_df(hierarchy):
        """
        assumed: the hierarchy has **4** levels
        """
        df = pd.DataFrame.from_dict({
            'Level_1': i,
            'Level_2': j,
            'Level_3': k,
            'Activity': l
        } for i in hierarchy.keys() for j in hierarchy[i].keys() for k in hierarchy[i][j].keys() for l in hierarchy[i][j][k])
        return df

    def group_by_level(self,
                       level:str)->dict:

        hierarchy=self.df_hierarchy
        energy=self._energy_pivot
        keys=hierarchy[level].unique().tolist()

        output = {}
        for key in keys:
            filtered_hierarchy = hierarchy[hierarchy[level] == key]
            activities_to_filter=filtered_hierarchy['Activity'].unique().tolist()
            pass
            energy_filtered=energy.filter(activities_to_filter)
            energy_filtered[key]=energy_filtered.sum(axis=1)
            output[key]=energy_filtered[key].tolist()


        output = pd.DataFrame.from_dict(output)
        self.energy_by_level = output
        return output


    def join_impacts_input(self):
        """
        Join in the same df the energy inputs
        """
        df_energy = pd.read_csv(r'flow_out_clean.csv', delimiter=',')
        pass
        df_energy = df_energy.drop(df_energy.columns[0], axis=1)
        df_energy = df_energy.groupby(['scenarios', 'aliases'])['flow_out_sum'].sum().reset_index()
        df_energy = df_energy.pivot(index='scenarios', columns='aliases', values='flow_out_sum')
        self.__energy_pivot=df_energy
        return df_energy

    def join_impacts_levels(self):
        pass





    @staticmethod
    def get_general_results():

        out_results = OrderedDict()
        with open('results_TFM.json') as file:
            d = json.load(file, object_pairs_hook=OrderedDict)
        pass
        for scen in range(len(d)):
            scen_name = scen
            out_results[scen_name] = d[scen]['results']
            a = pd.DataFrame.from_dict(out_results)
            pass
            a = a.T
        a = a.drop('global temperature change potential (GTP100)', axis=1)
        pass
        return a



aaa=Hierarchy(hierarchy_tree=hierarchy)
aaa.group_by_level('Level_2')
aaa.get_general_results()

pass
