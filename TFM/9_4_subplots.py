import pandas as pd

from collections import OrderedDict

import matplotlib.pyplot as plt

from scipy.stats import pearsonr, ttest_ind

import shap
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import metrics
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from statsmodels.stats.outliers_influence  import variance_inflation_factor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, davies_bouldin_score, accuracy_score, \
    precision_score, recall_score, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
hierarchy={
            "Generation": [
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
                    "waste_supply__waste___PRT",
                    "biofuel_supply__biofuel___PRT",
                    "chp_biofuel_extraction__heat___PRT",
                    "chp_wte_back_pressure__heat___PRT",
                    "chp_methane_extraction__heat___PRT",
                    "biofuel_boiler__heat___PRT",
                    "methane_boiler__heat___PRT"
                ]
            ,
            "Storage":
                 [
                    "battery__electricity___PRT",
                    "pumped_hydro__electricity___PRT" 
                    "heat_storage_big__heat___PRT",
                    "heat_storage_small__heat___PRT",
                    "methane_storage__methane___PRT"
                ],

            "Conversions":
                 [
                    "biofuel_to_diesel__diesel___PRT",
                    "biofuel_to_methane__methane___PRT",
                    "biofuel_to_methanol__methanol___PRT",
                    "electrolysis__hydrogen___PRT"
                ],

            "Imports":  [
                    "el_import__electricity___ESP"
                ]}




hierarchy_2= {
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
                    "waste_supply__waste___PRT"],

                "Thermal_generation": [
                    "biofuel_supply__biofuel___PRT",
                    "chp_biofuel_extraction__heat___PRT",
                    "chp_wte_back_pressure__heat___PRT",
                    "chp_methane_extraction__heat___PRT",
                    "biofuel_boiler__heat___PRT",
                    "methane_boiler__heat___PRT"
                ],
                "Electricity_storage": [
                    "battery__electricity___PRT",
                    "pumped_hydro__electricity___PRT"
                ],
                "Thermal_storage": [
                    "heat_storage_big__heat___PRT",
                    "heat_storage_small__heat___PRT",
                    "methane_storage__methane___PRT"
                ],
                "Conversions": [
                    "biofuel_to_diesel__diesel___PRT",
                    "biofuel_to_methane__methane___PRT",
                    "biofuel_to_methanol__methanol___PRT",
                    "electrolysis__hydrogen___PRT"
                ],
                "Imports": [
                    "el_import__electricity___ESP"
                ]
            }
class Subplots:
    def __init__(self):

        self.levels_1_df=self.get_general_results_n1()
        self.level1_normalized=self.get_normalized_value_general(self.levels_1_df)
        self.level_2_dfs=self.get_general_results_n2()
        self.level_2_dfs_normalized=self.get_normalized_value()
        self.energy_input_categories=self.energy_categories()

        self.get_level_n3()
        self.energy_categories_n3()
        self.charachteristics_l_4()

    def get_general_results_n1(self):
        out_results = OrderedDict()
        with open('results_TFM_update.json') as file:
            d = json.load(file, object_pairs_hook=OrderedDict)
        pass
        for scen in range(len(d)):
            scen_name = scen
            out_results[scen_name] = d[scen]['results']
            a = pd.DataFrame.from_dict(out_results)
            a = a.T
        a = a.drop(
            ['exergy content', "global hectares", "water pollutants", "global temperature change potential (GTP100)"],
            axis=1)
        return a


    def get_general_results_n2(self):
            out_results = {}
            generation_list = []
            storage_list = []
            conversions_list = []
            imports_list = []

            with open('results_TFM_update.json') as file:
                d = json.load(file)
            columns_to_drop = ['exergy content', "global hectares", "water pollutants",
                               "global temperature change potential (GTP100)"]
            for scen in range(len(d)):
                scen_name = scen

                out_results[scen_name] = d[scen]['children'][0]['children']
                out_results_ = out_results[scen_name]

                generation = pd.DataFrame.from_dict(out_results_[0]['results'], orient='index')
                generation = generation.T
                generation['scen_name'] = scen_name
                generation_list.append(generation)

                storage = pd.DataFrame.from_dict(out_results_[1]['results'], orient='index')
                pass
                storage = storage.T
                storage['scen_name'] = scen_name
                storage_list.append(storage)

                conversions = pd.DataFrame.from_dict(out_results_[2]['results'], orient='index')
                conversions = conversions.T
                conversions['scen_name'] = scen_name
                conversions_list.append(conversions)

                imports = pd.DataFrame.from_dict(out_results_[3]['results'], orient='index')
                imports = imports.T
                imports['scen_name'] = scen_name
                imports_list.append(imports)


            all_generations = pd.concat(generation_list)
            all_storage = pd.concat(storage_list)
            all_conversions = pd.concat(conversions_list)
            all_imports = pd.concat(imports_list)

            all_generations = all_generations.drop(columns=columns_to_drop, errors='ignore')
            all_storage = all_storage.drop(columns=columns_to_drop, errors='ignore')
            all_conversions = all_conversions.drop(columns=columns_to_drop, errors='ignore')
            all_imports = all_imports.drop(columns=columns_to_drop, errors='ignore')

            all_generations = all_generations.set_index('scen_name')
            all_storage = all_storage.set_index('scen_name')
            all_conversions = all_conversions.set_index('scen_name')
            all_imports = all_imports.set_index('scen_name')


            final_list={}
            final_list['generation']=all_generations
            final_list['storage']=all_storage
            final_list['conversions']=all_conversions
            final_list['imports']=all_imports

            self.level_2_dfs=final_list
            return final_list

    def get_level_n3(self):
        out_results = {}
        generation_electricity_list= []
        generation_thermal_list=[]
        storage_electricity_list = []
        storage_thermal_list=[]
        conversions_list = []
        imports_list = []

        with open('results_TFM_update.json') as file:
            d = json.load(file)
        columns_to_drop = ['exergy content', "global hectares", "water pollutants",
                           "global temperature change potential (GTP100)"]

        for scen in range(len(d)):
            scen_name = scen

            out_results[scen_name] = d[scen]['children'][0]['children']
            out_results_ = out_results[scen_name]

            generation_electricity = pd.DataFrame.from_dict(out_results_[0]['children'][0]['results'], orient='index')
            generation_electricity=generation_electricity.T
            generation_electricity['scen_name'] = scen_name
            generation_electricity_list.append(generation_electricity)

            generation_thermal = pd.DataFrame.from_dict(out_results_[0]['children'][1]['results'], orient='index')
            generation_thermal = generation_thermal.T
            generation_thermal['scen_name'] = scen_name
            generation_thermal_list.append(generation_electricity)



            pass
            storage_electricity = pd.DataFrame.from_dict(out_results_[1]['children'][0]['results'], orient='index')
            storage_electricity = storage_electricity.T
            storage_electricity['scen_name'] = scen_name
            storage_electricity_list.append(generation_electricity)


            storage_thermal = pd.DataFrame.from_dict(out_results_[1]['children'][1]['results'], orient='index')
            storage_thermal = storage_thermal.T
            storage_thermal['scen_name'] = scen_name
            storage_thermal_list.append(generation_electricity)

            conversions = pd.DataFrame.from_dict(out_results_[2]['results'], orient='index')
            conversions = conversions.T
            conversions['scen_name'] = scen_name
            conversions_list.append(conversions)

            imports = pd.DataFrame.from_dict(out_results_[3]['results'], orient='index')
            imports = imports.T
            imports['scen_name'] = scen_name
            imports_list.append(imports)


        all_generation_electricity = pd.concat(generation_electricity_list)
        all_generation_thermal = pd.concat(generation_thermal_list)
        all_storage_electricity = pd.concat(storage_electricity_list)
        all_storage_thermal = pd.concat(storage_thermal_list)
        all_conversions = pd.concat(conversions_list)
        all_imports = pd.concat(imports_list)

        all_generation_electricity = all_generation_electricity.drop(columns=columns_to_drop, errors='ignore')
        all_generation_thermal = all_generation_thermal.drop(columns=columns_to_drop, errors='ignore')
        all_storage_electricity = all_storage_electricity.drop(columns=columns_to_drop, errors='ignore')
        all_storage_thermal = all_storage_thermal.drop(columns=columns_to_drop, errors='ignore')
        all_conversions = all_conversions.drop(columns=columns_to_drop, errors='ignore')
        all_imports = all_imports.drop(columns=columns_to_drop, errors='ignore')

        all_generation_electricity = all_generation_electricity.set_index('scen_name')
        all_generation_thermal = all_generation_thermal.set_index('scen_name')
        all_storage_electricity = all_storage_electricity.set_index('scen_name')
        all_storage_thermal = all_storage_thermal.set_index('scen_name')
        all_conversions = all_conversions.set_index('scen_name')
        all_imports = all_imports.set_index('scen_name')

        final_list = {}
        final_list['generation_electricity'] = all_generation_electricity
        final_list['generation_thermal'] = all_generation_thermal
        final_list['storage_electricity'] = all_storage_electricity
        final_list['storage_thermal'] = all_storage_thermal
        final_list['conversions'] = all_conversions
        final_list['imports'] = all_imports
        pass
        self.level_n3_dfs = final_list


    def get_normalized_value(self):
        """
        Normalize the results between 0 and 1
        -------
        Return: modified dictionary of DataFrames
        """
        data = self.level_2_dfs

        for k, v in data.items():
            res = v
            info_norm = {}
            normalized_df = pd.DataFrame()

            for col in res.columns:
                name = str(col)
                name_modified = name + "_"

                min_ = np.min(res[col])
                max_ = np.max(res[col])


                normalized_df[name_modified] = (res[col] - min_) / (max_ - min_)

                info_norm[col] = {
                    "max": max_,
                    "min": min_,
                    "scenario_max": normalized_df[name_modified].idxmax(),
                    "scenario_min": normalized_df[name_modified].idxmin()
                }

            # Drop the original columns from the new DataFrame
            res = res.drop(res.columns, axis=1)
            normalized_df.columns = [col.rstrip('_') for col in normalized_df.columns]


            data[k] = pd.concat([res, normalized_df], axis=1)



        return data


    def get_normalized_value_general(self,df):
        """
        Normalize the results between 0 and 1
        -------
        Return: modified dictionary of DataFrames
        """



        res = df
        info_norm = {}
        normalized_df = pd.DataFrame()

        for col in res.columns:
            name = str(col)
            name_modified = name + "_"

            min_ = np.min(res[col])
            max_ = np.max(res[col])


            normalized_df[name_modified] = (res[col] - min_) / (max_ - min_)

            info_norm[col] = {
                "max": max_,
                "min": min_,
                "scenario_max": normalized_df[name_modified].idxmax(),
                "scenario_min": normalized_df[name_modified].idxmin()
            }


        res = res.drop(res.columns, axis=1)
        normalized_df.columns = [col.rstrip('_') for col in normalized_df.columns]


        data= pd.concat([res, normalized_df], axis=1)

        return data


    def plot_correlation_matrices(self):
        """
        Plot correlation matrices for DataFrames in a dictionary
        """
        data_dict = self.level_2_dfs_normalized
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12), sharex=True, sharey=True)



        plt.subplots_adjust(wspace=0.2, hspace=0.2)


        for (key, df), ax in zip(data_dict.items(), axes.flatten()):

            df = df.drop(columns=df.columns[8:])
            correlation_matrix = df.corr()
            sns.set(style="whitegrid")
            sns.heatmap(correlation_matrix, cmap='coolwarm', cbar=False, vmin=-1, vmax=1, linewidths=.5,
                        square=True, ax=ax, annot=True, fmt=".2f", annot_kws={"size": 12})


            ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
            ax.tick_params(axis='x', labelsize=15)
            ax.tick_params(axis='y',labelsize=15)
            ax.set_title(f'n-1 {key}', fontsize=16)


        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(ax.collections[0], cax=cbar_ax)
        cbar.set_label('Correlation Index r', rotation=270, labelpad=15)


        plt.savefig("plots/correlation_matrices.png", dpi=1500, bbox_inches='tight', format='png')







    def marginal_contribution(self):
        data_dict = self.level_2_dfs
        df_total=self.levels_1_df
        df_contribuciones = pd.DataFrame(index=self.levels_1_df.index)
        dict_contributions={}
        for columna in df_total.columns:
            contribuciones_columna = {}
            pass
            for key, dataframe in data_dict.items():
                contribucion = dataframe[columna] / df_total[columna]
                contribuciones_columna[key] = contribucion


            df_contribuciones_columna = pd.DataFrame(contribuciones_columna)


            dict_contributions[columna] = df_contribuciones_columna
        return dict_contributions
        pass



    def energy_categories(self):
        df=pd.read_csv(r'flow_out_processed.csv')


        df['category'] = None


        for main_category, techs in hierarchy.items():
            df.loc[df['aliases'].isin(techs), 'category'] = main_category

        pass
        result_df = df.groupby(['category', 'scenarios']).agg({'flow_out_sum': 'sum'}).reset_index()
        result_df_pivoted = result_df.pivot_table(index='scenarios', columns='category', values='flow_out_sum',
                                                  fill_value=0)

        self.general_check=result_df
        #result_df_pivoted=result_df_pivoted.set_index('scenarios')
        pass
        self.energy_input_categories_=result_df_pivoted


        result_df_pivoted_normalized = result_df_pivoted.apply(lambda col: (col - col.min()) / (col.max() - col.min()))
        self.energy_input_categories_normalized=result_df_pivoted_normalized
        pass

        return result_df_pivoted_normalized

    def energy_categories_n3(self):
        df=pd.read_csv(r'/home/lex/PycharmProjects/Sparks_2/TFM/flow_cru.csv')
        df['aliases']=df['techs']+'__'+df['carriers']+'___'+df['locs']
        pass

        df['category'] = None  #


        for main_category, techs in hierarchy_2.items():
            df.loc[df['aliases'].isin(techs), 'category'] = main_category
        pass

        result_df = df.groupby(['category', 'spores'], sort=False).agg({'flow_out_sum': 'sum'}).reset_index()
        result_df.rename(columns={'spores':'scenarios'}, inplace=True)

        pass
        self.lvl_3_check=result_df
        result_df_pivoted = result_df.pivot_table(index='scenarios', columns='category', values='flow_out_sum',
                                                  fill_value=0)

        #result_df_pivoted=result_df_pivoted.set_index('scenarios')
        self.energy_input_categories_n3=result_df_pivoted


        result_df_pivoted_normalized = result_df_pivoted.apply(lambda col: (col - col.min()) / (col.max() - col.min()))
        self.energy_input_categories_normalized_n3=result_df_pivoted_normalized
        pass

        return result_df_pivoted_normalized


    def plot_heatmap_3rows_with_index_and_scale(self, cmap):
        """

        """
        data = self.energy_input_categories
        categories = data.columns

        # Crear el objeto de la figura y el eje automáticamente
        plt.figure(figsize=(18, 12))

        # Ajustes para las bolas y las filas separadas
        for i, category in enumerate(categories):
            x = data.index
            y_top = [i + 0.25] * 87
            y_middle = [i] * 87
            y_bottom = [i - 0.25] * 87
            colors = cmap(data[category].values)

            # Espaciar los círculos para las tres filas
            x_spaced = np.linspace(min(x), max(x), 87)


            for xi, y_topi, color, index in zip(x_spaced, y_top, colors[:87], range(87)):
                plt.scatter(xi, y_topi, c=color, s=60, marker='o', edgecolors='none')
                plt.text(xi, y_topi - 0.1, str(index), ha='center', va='center', fontsize=7.5, color='black', rotation=90)

            # Plotear cada círculo individualmente para la fila del medio
            for xi, y_middlei, color, index in zip(x_spaced, y_middle, colors[87:174], range(87, 174)):
                plt.scatter(xi, y_middlei, c=color, s=60, marker='o', edgecolors='none')
                plt.text(xi, y_middlei - 0.1, str(index), ha='center', va='center', fontsize=7.5, color='black',
                         rotation=90)

            # Plotear cada círculo individualmente para la fila inferior
            for xi, y_bottomi, color, index in zip(x_spaced, y_bottom, colors[174:], range(174, 261)):
                plt.scatter(xi, y_bottomi, c=color, s=60, marker='o', edgecolors='none')
                plt.text(xi, y_bottomi - 0.1, str(index), ha='center', va='center', fontsize=7.5, color='black',
                         rotation=90)

            # ...

            # Añadir título de columna centrado
        for i, category in enumerate(categories):
            plt.text(np.min(x_spaced) -5, i, category, ha='right', va='center', fontsize=13, color='black',
                     rotation=90)

        plt.gca().set_frame_on(False)
        plt.grid(axis='x', alpha=0.4)
        plt.gca().set_axisbelow(True)

        plt.yticks([])
        # Ajustes para las xticks
        xticks_location = np.arange(0, len(x_spaced) * 3, 30) - 0.5
        plt.xticks(xticks_location, labels=np.arange(0, len(x_spaced), 10))
        # Eliminar xticks
        plt.xticks([])
        plt.tick_params(size=0, colors="0.3")

        cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), orientation='vertical', fraction=0.025, pad=0.07)
        cbar.set_label('Relative input value', fontsize=12)

        cax = cbar.ax
        cax.set_position([0.85, 0.1, 0.02, 0.8])
        cbar.ax.yaxis.set_tick_params(width=0)
        cbar.ax.yaxis.label.set_size(10)

        plt.subplots_adjust(hspace=0.5)
        plt.margins(x=0.05)
        # Desplazar los subplots hacia la derecha
        plt.subplots_adjust(left=0.1, right=0.95)
        plt.tight_layout(pad=2)
        plt.savefig("plots/heat_entrance_3rows_with_index_and_scale", dpi=700, bbox_inches='tight')
        plt.show()




    def join_for_caracteristics_n2(self):
        b,df=self.clusters()
        pass
        b=b['cluster']
        energy=self.energy_input_categories
        df=pd.concat([energy,b],axis=1)
        return df

    def join_caracteristics_n3(self):
        b, df = self.clusters()
        b = b['cluster']
        energy = self.energy_input_categories_normalized_n3
        df = pd.concat([energy, b], axis=1)
        return df

    from scipy.stats import ttest_ind
    def caracteristics(self):
        df=self.join_for_caracteristics_n2()
        pass

        cluster_0_data = df[df['cluster'] == 0]
        cluster_1_data = df[df['cluster'] == 1]
        pass

        means_0 = cluster_0_data[['Storage', 'Generation', 'Conversions', 'Imports']].mean()
        means_1 = cluster_1_data[['Storage', 'Generation', 'Conversions', 'Imports']].mean()

        means_df = pd.DataFrame({'Cluster 0': means_0, 'Cluster 1': means_1})
        for feature in ['Storage', 'Generation', 'Conversions', 'Imports']:
            t_stat, p_value = ttest_ind(cluster_0_data[feature], cluster_1_data[feature])
            print(f'Prueba t para {feature}: Estadístico t={t_stat}, Valor p={p_value}')


        means_df['Diferencia'] = means_df['Cluster 1'] - means_df['Cluster 0']
        sns.set_palette("colorblind")

        means_df[['Cluster 0', 'Cluster 1']].plot(kind='bar', colormap='viridis', figsize=(10, 6))
        plt.bar(means_df.index, means_df['Diferencia'], color='black', alpha=0.3, label='Diferencia')
        plt.title('Diferencias Absolutas entre Medias en Cluster 0 y 1')
        plt.ylabel('Media')
        plt.show()
        plt.savefig('plots/means.png',dpi=300)

    def caracteristics_n3(self):
        df = self.join_caracteristics_n3()
        cluster_0_data = df[df['cluster'] == 0]
        cluster_1_data = df[df['cluster'] == 1]
        pass

        means_0 = cluster_0_data.iloc[:, :-1].mean()
        means_1 = cluster_1_data.iloc[:, :-1].mean()
        means_df = pd.DataFrame({'Cluster 0': means_0, 'Cluster 1': means_1})
        for feature in cluster_0_data.iloc[:, :-1].columns:
            t_stat, p_value = ttest_ind(cluster_0_data[feature], cluster_1_data[feature])
            print(f'Prueba t para {feature}: Estadístico t={t_stat}, Valor p={p_value}')

        means_df = means_df.reset_index().melt(id_vars='index', var_name='Cluster', value_name='Media')


        colores_personalizados = {'Cluster 0': '#4CAF50', 'Cluster 1': '#2196F3'}
        plt.figure(figsize=(12, 8))

        ax = sns.barplot(x='index', y='Media', hue='Cluster', data=means_df,
                         palette=colores_personalizados, saturation=0.8, linewidth=0.1, edgecolor='black',alpha=0.7)

        ax.legend(title='Cluster', loc='upper right', bbox_to_anchor=(1, 1), frameon=True, fontsize='medium',
                  handlelength=0.1, edgecolor='black', framealpha=0.1)


        plt.ylabel('Mean value', fontsize=12)
        plt.xlabel('')
        plt.xticks(rotation=35, ha='right')
        ax.tick_params(axis='x', which='major', labelsize=12)

        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)


        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False, fontsize='medium')
        plt.tight_layout()
        plt.savefig('plots/means_lvl3.png', dpi=900, bbox_inches='tight')
        plt.show()

    def clusters(self):
        df = self.level1_normalized
        pass
        df = df.drop('biotic resources', axis=1)
        X = df.values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        n_clusters = 2

        k_means = KMeans(n_clusters=n_clusters, random_state=42)
        df['cluster'] = k_means.fit_predict(X_scaled)
        inertia = k_means.inertia_

        # Calcular la silueta
        silhouette_score = metrics.silhouette_score(X_scaled, df['cluster'])
        davies_bouldin = davies_bouldin_score(X_scaled, df['cluster'])
        # Visualizar los resultados
        plt.figure(figsize=(12, 6))

        # Visualizar los clusters en un gráfico
        sns.scatterplot(data=df, x=df.index, y='cluster', palette='viridis', marker='o', hue='cluster')
        plt.xlabel('Escenario')
        plt.ylabel('Cluster')
        plt.title('Clustering de Escenarios')

        # Mostrar la inercia y la silueta en el gráfico
        plt.text(1, -0.5, f'Inercia: {inertia:.2f}', fontsize=12)
        plt.text(1, -0.75, f'Silueta: {silhouette_score:.2f}', fontsize=12)

        print(f'Silueta: {silhouette_score:.2f}')
        print(f'Davies-Bouldin: {davies_bouldin:.2f}')
        plt.savefig(r'plots/clustering.png')


        # Verificar los resultados
        b = df.groupby('cluster').mean()
        return df, b



    def charachteristics_l_4(self):

            """
            Join in the same df the energy inputs + results
            """
            b, df = self.clusters()
            b = b['cluster']
            df_energy = pd.read_csv(r'flow_out_processed.csv', delimiter=',')
            pass
            # df_energy=df_energy.drop(df_energy.columns[0],axis=1)
            df_energy = df_energy.groupby(['scenarios', 'techs'])['flow_out_sum'].sum().reset_index()
            df_energy = df_energy.pivot(index='scenarios', columns='techs', values='flow_out_sum')

            result = pd.concat([df_energy,b], axis=1)

            X = result.drop('cluster', axis=1)
            y = result['cluster']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            clf = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=42)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)


            print(f'Accuracy: {accuracy:.4f}')
            print(f'Precision: {precision:.4f}')
            print(f'Recall: {recall:.4f}')
            print(f'Confusion Matrix:\n{conf_matrix}')


            importances = clf.feature_importances_
            feature_names = X.columns


            feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

            feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')

            plt.xlabel('Importance')
            plt.ylabel('')
            plt.xticks(ha='right', fontsize=14)
            plt.yticks(ha='right',fontsize=12)

            plt.tight_layout()

            plt.savefig('plots/feature_importance.png', dpi=800, bbox_inches='tight')
            plt.show()


            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_test)


            shap.summary_plot(shap_values[1], X_test, plot_type='bar', show=False)


            plt.savefig('plots/shapley_values.png', dpi=300, bbox_inches='tight')
            plt.show()


    def get_best_values(self):
        data=self.level1_normalized
        top_10={}
        for col in data.columns:
            top_10[col]=data.nsmallest(10,col).sort_values(by=col).index.tolist()
        print("\n Top 10 best for category:")
        for column,scenario in top_10.items():
            print(f"{column}: {scenario}")

        # Add here some rules to find the best scenario
        data['sum_rule_1'] = data['water consumption potential (WCP)'] + data['agricultural land occupation (LOP)']
        best_water_land = data['sum_rule_1'].idxmin()

        data['sum_rule_2'] = data['water consumption potential (WCP)'] + data['biotic resources']
        best_water_biotic = data['sum_rule_2'].idxmin()

        data['sum_rule_3'] = data['freshwater ecotoxicity potential (FETP)'] + data[
            'marine ecotoxicity potential (METP)'] + data['biotic resources']
        best_biotic_toxic = data['sum_rule_3'].idxmin()

        print(f"rule 1: {best_water_land}, best rule 2: {best_water_biotic}, rule 3: {best_biotic_toxic}")

        # find the best spore
        data['sum'] = data.drop(['biotic resources', 'sum_rule_1', 'sum_rule_2', 'sum_rule_3'], axis=1).sum(axis=1)
        best_scenario = data['sum'].idxmin()

        #
        print(f"rule 1: {best_water_land}, best rule 2: {best_water_biotic}, rule 3: {best_biotic_toxic}")
        print(f'best overall scenario {best_scenario}')

        # find the best spore
        data['sum_2'] = data.drop(['sum_rule_1', 'sum_rule_2', 'sum_rule_3','sum'], axis=1).sum(axis=1)
        best_scenario_2 = data['sum_2'].idxmin()

        #
        print(f"rule 1: {best_water_land}, best rule 2: {best_water_biotic}, rule 3: {best_biotic_toxic}")
        print(f'best overall scenario {best_scenario}')
        print(f'best overall scenario 2 {best_scenario_2}')

        # Worst
        data['sum_3'] = data.drop(['sum_rule_1', 'sum_rule_2', 'sum_rule_3', 'sum', 'sum_2'], axis=1).sum(axis=1)
        worst_scen= data['sum_3'].idxmax()
        print(f" Worst scenario {worst_scen}")
        pass








plotty=Subplots()
bby=plotty.get_general_results_n2()
a=plotty.plot_correlation_matrices()
plotty.marginal_contribution()

cmap = sns.color_palette("viridis", as_cmap=True)
#plotty.plot_heatmap_3rows_with_index_and_scale(cmap)
#plotty.random_forest_analyzer()
#plotty.random_forest_analyzer2()
plotty.caracteristics()
#plotty.caracteristics_n3()
plotty.get_best_values()
#plotty.crear_pie_chart([83,152,0])
