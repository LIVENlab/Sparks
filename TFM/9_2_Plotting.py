import pandas as pd
import numpy as np
# Copied from enbios' input
from collections import OrderedDict
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from scipy.stats import pearsonr
import seaborn as sns
from SALib.analyze import sobol
from SALib.sample import saltelli



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

        total=self.get_general_results()
        pass
        self.join_df=pd.concat([total,output],axis=1)
        self.join_df=self.get_normalized_value(self.join_df)

        return output


    def get_normalized_value(self,df):
        """
        Normalize the results between 0 and 1
        -------
        Return: modified DF
        """

        res = df
        info_norm = {}
        pass
        for col in res.columns:
            name = str(col)
            name_modified = name + "_"
            min_ = np.min(res[col])
            max_ = np.max(res[col])
            res[name_modified] = (res[col] - min_) / (max_ - min_)
            info_norm[col] = {
                "max": max_,
                "min": min_,
                "scenario_max": res[col].idxmax().item(),
                "scenario_min": res[col].idxmin().item()
            }
            # drop the original cols
            res = res.drop(col, axis=1)
        pass
        return res


    def get_general_results(self):

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
        self.general_impacts=a
        pass
        return a

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




    def _2(self):

        out_results = OrderedDict()
        with open('results_TFM.json') as file:
            d = json.load(file, object_pairs_hook=OrderedDict)

        pass
        for scen in range(len(d)):
            scen_name = scen
            pass
            bb=self.recrusive_dict(d[scen],2)
            pass
            out_results[scen_name] = bb

        self.dict_to_df(out_results)
            # TODO: add function that allows to transform the dataframe

        #a = a.drop('global temperature change potential (GTP100)', axis=1)
    """"
    PER 18/01:
        -Recorda: impacte total
        -Només has d'agregar la PRODUCCIÓ

    """

    def dict_to_df(self,
                   dictionary):
        df_join=pd.DataFrame()
        for k, value in dictionary.items():
            pass


            #a = pd.DataFrame.from_dict(out_results)
        pass
        #a = a.T

        pass

    def recrusive_dict(self,
                       dictionary,
                       depth,
                       count_depth=0)->dict:

        """ Only allows to get proper results till level 2""" # TODO

        if count_depth==depth-1:
            dictionary_iter=dictionary['children']
            result_final={}
            for element in dictionary_iter:
                result_final[element['alias']]=element['results']
            return result_final

        elif "children" in dictionary:
            for child in dictionary['children']:
                result=self.recrusive_dict(child,depth,count_depth+1)
                if result is not None:
                    return result




    def plot_pearsons_heatmap(self):
            """
            Compute the pearson correlation between imputs and ouputs

            https://en.wikipedia.org/wiki/Pearson_correlation_coefficient:
                The correlation coefficient ranges from −1 to 1.
                An absolute value of exactly 1 implies that a linear equation describes the relationship between X and Y perfectly, with all data points lying on a line.
                The correlation sign is determined by the regression slope: a value of +1 implies that all data points lie on a line for which Y increases as X increases, and vice versa for −1.
                A value of 0 implies that there is no linear dependency between the variables
            --------------------------
            Assumptions:
                -Data follow a linear relation
                -Both variables are quantitative
                -Normally distributed
                -Have no outliers
            """
            df = self.join_df
            pass
            X = df.drop(columns=df.columns[:11], axis=1)
            X = X.drop(columns=['Imports_'], axis=1)
            Y = df[df.columns[:11]]
            pass
            correlation_matrix = pd.DataFrame(index=X.columns, columns=Y.columns)

            for col in Y.columns:
                var_y = Y[col]
                for cols in X.columns:
                    var_x = X[cols]
                    correlation, _ = pearsonr(var_x, var_y)
                    correlation_matrix.loc[cols, col] = correlation
            plt.figure(figsize=(10, 6))
            sns.heatmap(correlation_matrix.astype(float), annot=True, cmap='coolwarm', linewidths=0.5)
            plt.title('Pearson Correlation Matrix (Impacts-Technology)', fontsize=18)
            plt.savefig("plots/pearson_2.png", dpi=800, bbox_inches='tight')

    def sensitivity_index(self):

        df = self.join_df
        X = df.drop(columns=df.columns[:11], axis=1)
        X = X.drop(columns=['Imports_'], axis=1)

        Y = df[df.columns[:11]]


        problem = {
            'num_vars': len(X.columns),
            'names': X.columns.tolist(),
            'bounds': [[X[col].min(), X[col].max()] for col in X.columns]
        }
        resultados_df = pd.DataFrame(columns=['Indicador_Impacto', 'Tecnologia', 'Indice_Sensibilidad_Global'])
        pass
        # Realizar el Análisis de Sensibilidad de Sobol para cada indicador de impacto y cada tecnología
        for impacto_column in Y.columns:
            # Obtener los resultados de impacto para el indicador actual
            resultados_impacto_actual = Y[impacto_column].values
            pass

            # Realizar el Análisis de Sensibilidad de Sobol utilizando Monte Carlo
            sensitivity_indices = sobol.analyze(problem, resultados_impacto_actual)
            print(f"\nÍndices de Sensibilidad Global para {impacto_column}:")
            print(f'    Índices de primer orden: {sensitivity_indices["S1"]}')
            print(f'    Índices de segundo orden: {sensitivity_indices["S2"]}')
            for i, tecnologia in enumerate(problem['names']):
                resultados_df = resultados_df.append({
                    'Indicador_Impacto': impacto_column,
                    'Tecnologia': tecnologia,
                    'Indice_Sensibilidad_Global': sensitivity_indices['ST'][i]
                }, ignore_index=True)

            # Do something with sensitivity_results (e.g., store in sensitivity_index DataFrame)

        # Return or use sensitivity_index as needed




                #correlation_matrix.loc[cols, col] = correlation
        pass

aaa=Hierarchy(hierarchy_tree=hierarchy)
aaa.group_by_level('Level_2')
aaa.get_general_results()
aaa.plot_pearsons_heatmap()
#aaa.sensitivity_index()

pass
