
import pandas as pd


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


    def group_by_level(level:str)->pd.DataFrame:


        pass


    def join_impacts_input(self):
        """
        Join in the same df the energy inputs + results
        """
        df_energy = pd.read_csv(r'flow_out_clean.csv', delimiter=',')
        df_energy = df_energy.drop(df_energy.columns[0], axis=1)

        df_energy = df_energy.groupby(['scenarios', 'techs'])['flow_out_sum'].sum().reset_index()
        df_energy = df_energy.pivot(index='scenarios', columns='techs', values='flow_out_sum')
        self.__energy_pivot=df_energy
        return df_energy




aaa=Hierarchy(hierarchy_tree=hierarchy)
pass
