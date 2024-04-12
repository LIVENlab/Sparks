import pandas as pd
import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt
from collections import OrderedDict

from scipy.stats import spearmanr


class PlottingEnbios:

    # TODO: Plots intensivos
    def __init__(self, results_path, flow_out_sum, starter, units):
        """
        -results_path: str
        """

        self._path=results_path
        self._flow_out_path=flow_out_sum
        self._starter_path=starter
        self._units_path=units
        self._raw_data=None
        # Basic actions
        self.data=self._get_general_results(self._path)
        self.normal_data=self._normalize_values()


    def _get_general_results(self, *args):

        out_results = OrderedDict()
        with open(*args) as file:
            self._raw_data = json.load(file, object_pairs_hook=OrderedDict)
        for scen in range(len(self._raw_data)):
            scen_name = scen
            out_results[scen_name] = self._raw_data[scen]['results']
            data = pd.DataFrame.from_dict(out_results).T
        return data

    def _normalize_values(self) -> pd.DataFrame:
        """ Normalize values between 0 and 1"""
        res = self.data
        info_norm = {}
        for col in res.columns:
            name = str(col)
            min_ = np.min(res[col])
            max_ = np.max(res[col])
            res[name] = (res[col] - min_) / (max_ - min_)
            info_norm[col] = {
                "max": max_,
                "min": min_,
                "scenario_max": res[col].idxmax().item(),
                "scenario_min": res[col].idxmin().item()
            }
        return res

    def _join_impacts(self):
        """
        Join in the same df the energy inputs + results
        """
        df_energy = pd.read_csv(self._starter_path, delimiter=',')
        #df_energy = df_energy.drop(df_energy.columns[0], axis=1)
        df_energy = df_energy.groupby(['scenarios', 'techs'])['flow_out_sum'].sum().reset_index()
        df_energy = df_energy.pivot(index='scenarios', columns='techs', values='flow_out_sum')

        return pd.concat([self._normalize_values(), df_energy], axis=1)




    def load_hierarchy(self):
        pass


    def hierarchy_to_df(self,hierarchy):
        """
        assumed: the hierarchy has **4** levels
        """
        df = pd.DataFrame.from_dict({
                                        'Level_1': i,
                                        'Level_2': j,
                                        'Level_3': k,
                                        'Activity': l
                                    } for i in hierarchy.keys() for j in hierarchy[i].keys() for k in
                                    hierarchy[i][j].keys() for l in hierarchy[i][j][k])
        pass
        return df







    def violin_plot(self, spore: None, path_save:  str):
        """
        General Violin plot of results
        """
        df = self.normal_data

        mi_paleta = sns.color_palette("coolwarm", n_colors=5)
        sns.set(style="whitegrid", palette=mi_paleta)
        legend_labels = ['Spore', 'Cost-Optimized Spore']


        plt.figure(figsize=(18, 12))
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        sns.violinplot(data=df, orient='h', inner='quart', cut=0, palette='Set2', alpha=0.6)
        # Configurar etiquetas y título
        plt.xlabel("Valor Impact")
        plt.ylabel("Indicator")
        sns.stripplot(data=df, orient='h', color='black', alpha=0.4, jitter=0.2, label=legend_labels[0])

        # Add the selected spore. Spore 0 is the one that we would get with TIMES
        if spore is not None:
            df_selected = df.iloc[[spore]]
            sns.stripplot(data=df_selected, orient='h', color='red', jitter=1, size=13, marker='*',
                          label=legend_labels[1])
        plt.title("Normalized environmental impacts", fontsize=18)

        for l in plt.gca().lines:
            l.set_linewidth(2)

        plt.grid(True)
        handles, labels = plt.gca().get_legend_handles_labels()
        unique_labels = list(set(labels))
        unique_handles = [handles[labels.index(label)] for label in unique_labels]
        plt.legend(unique_handles, unique_labels, loc='upper right', fontsize='medium')

        plt.savefig(path_save, dpi=900, bbox_inches='tight')

    def correlation_matrix(self, path_save:str):
        """
        Study the correlation among the outputs (impacts)
        (Simple correlation)

        """
        df = self._join_impacts()
        df = df.drop(columns=df.columns[5:])
        correlation_matrix = df.corr()

        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', cbar=True)
        plt.title('Correlation index')
        plt.savefig(path_save, dpi=800, bbox_inches='tight')

    def spearman_correlation(self, path_save):
        """
        Compute the Spearman correlation between inputs and outputs.

        https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient:
            The correlation coefficient ranges from −1 to 1.
            An absolute value of exactly 1 implies that a monotonic relationship exists between X and Y.
            A value of 0 implies that there is no monotonic dependency between the variables.
        --------------------------
        Assumptions:
            - Data follow a monotonic relation.
            - Both variables are quantitative.
            - Have no outliers.
        """
        df = self._join_impacts()


        # Splitting the DataFrame into predictor variables (X) and output variables (Y)
        X = df.drop(columns=df.columns[:5], axis=1)  # Predictor features
        Y = df[df.columns[:5]]  # Output variables

        correlation_matrix = pd.DataFrame(index=X.columns, columns=Y.columns)
        for col in Y.columns:
            var_y = Y[col]
            for col_x in X.columns:
                var_x = X[col_x]
                correlation, _ = spearmanr(var_x, var_y)
                # Store the coefficient in the correlation matrix
                correlation_matrix.loc[col_x, col] = correlation

        plt.figure(figsize=(10, 6))
        sns.heatmap(correlation_matrix.astype(float), annot=True, cmap='coolwarm', linewidths=0.5)
        plt.xticks(rotation=45)
        plt.title('Spearman Correlation Matrix (Impacts-Technology)', fontsize=18)
        plt.savefig(path_save, dpi=800, bbox_inches='tight')

        return correlation_matrix


    def _edit_units(self):
        data = pd.read_csv(self._units_path, delimiter=',')
        pass
        data = data.drop(columns=data.columns[1:5])
        data = data.drop(columns=data.columns[4:])
        data=data.drop(columns=['alias_carrier'])

        df_transpuesto = data.set_index(['spores', 'names2']).unstack(fill_value=-11414)
        df_transpuesto.columns = df_transpuesto.columns.droplevel(0)
        return df_transpuesto


    def correlations_calliope(self, path_save):
        df=self._edit_units()
        correlation_matrix = df.corr()

        fig = plt.figure(figsize=(25, 20))

        sns.set(style="whitegrid")

        # heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm',linewidths=.5,
        #                    linecolor='white')
        heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5,
                              linecolor='white', fmt='.2f', annot_kws={"size": 8})


        plt.xticks(rotation=15, ha="right", )
        plt.tick_params(axis='x', labelsize=8)
        plt.tick_params(axis='y', labelsize=8)
        heatmap.collections[0].colorbar.remove()
        plt.ylabel('')
        plt.xlabel('')

        cbar = fig.colorbar(heatmap.collections[0], fraction=0.046, pad=0.04)
        cbar.set_label('Correlation Index r', rotation=270, labelpad=15)
        plt.tight_layout()
        plt.savefig(path_save, dpi=800, bbox_inches='tight')
        plt.show()




if __name__ =='__main__':

    plot=PlottingEnbios(r'C:\Users\Administrator\PycharmProjects\SEEDS\results\results_260.json',
                        flow_out_sum=r'C:\Users\Administrator\PycharmProjects\SEEDS\data_sentinel\flow_out_sum.csv',
                        starter=r'C:\Users\Administrator\PycharmProjects\SEEDS\Data_enbios_paper\units_260.csv',
                        units=r'C:\Users\Administrator\PycharmProjects\SEEDS\Data_enbios_paper\starter_260.csv')

    #plot.violin_plot(spore=0, path_save='violin_260')
    #plot.correlation_matrix("plots/correlation.png")
    #plot.spearman_correlation("plots/spearman.png")
    #plot.correlations_calliope(r"plots/correlation_matrix_calliope.png")


    hierarchy={
        "Energysystem": {
            "Generation": {
                "Electricity_generation": [
                    "wind_onshore__electricity___PRT",
                    "wind_offshore__electricity___PRT",
                    "hydro_run_of_river__electricity___PRT",
                    "hydro_reservoir__electricity___PRT",
                    "ccgt__electricity___PRT",
                    "chp_biofuel_extraction__electricity___PRT",
                    "open_field_pv__electricity___PRT",
                    "chp_hydrogen__electricity___PRT",
                    "existing_wind__electricity___PRT",
                    "existing_pv__electricity___PRT",
                    "roof_mounted_pv__electricity___PRT",
                    "chp_wte_back_pressure__electricity___PRT",
                    "chp_methane_extraction__electricity___PRT",
                    "waste_supply__waste___PRT"
                ],
                "Thermal_generation": [
                    "chp_biofuel_extraction__heat___PRT",
                    "chp_hydrogen__heat___PRT",
                    "chp_wte_back_pressure__heat___PRT",
                    "chp_methane_extraction__heat___PRT",
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
                    "biofuel_to_liquids__diesel___PRT",
                    "biofuel_to_methane__methane___PRT",
                    "biofuel_to_methanol__methanol___PRT",
                    "electrolysis__hydrogen___PRT"
                ]
            },
            "Imports": {
                "Imports": [
                    "el_import__electricity___ESP"
                ]
            }
        }}

    plot.hierarchy_to_df(hierarchy=hierarchy)



