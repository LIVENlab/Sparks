import pandas as pd
import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy.stats import spearmanr, pearsonr

data_group={
    'wind': ['wind_onshore','wind_offshore','existing_wind'],
    'solar':['roof_mounted_pv', 'open_field_pv','existing_pv'],
    'hydrogen':['electrolysis'],
    'storage': ['pumped_hydro', 'battery'],
    'imports': ['el_import'],
    'biofuel': ['chp_biofuel_extraction']
}


class PlottingEnbios:

    def __init__(self, results_path, flow_out_sum, starter, units):
        """
        -results_path: str
        """

        self._path=results_path
        self._flow_out_path=flow_out_sum
        self._starter_path=starter
        self._units_path=units
        self._raw_data=None
        self.results_csv= None

        # Basic actions
        self.data=self._get_general_results(self._path)
        self.normal_data=self._normalize_values()

        self.starter=self._get_amounts_starter()
        self.df = self._join_impacts()

        self.grouped=self.define_groups()

    def _get_general_results(self, *args):

        out_results = OrderedDict()
        with open(*args) as file:
            self._raw_data = json.load(file, object_pairs_hook=OrderedDict)
        for scen in range(len(self._raw_data)):
            scen_name = scen
            out_results[scen_name] = self._raw_data[scen]['results']
            data = pd.DataFrame.from_dict(out_results).T
        return data

    def _normalize_values(self, data=None) -> pd.DataFrame:
        """ Normalize values between 0 and 1"""
        if data is None:
            res = self.data
        else:
            res = data
        pass
        for col in res.columns:
            min_spore_0 = res[col][0]
            name = str(col)
            min_ = np.min(res[col])
            max_ = np.max(res[col])
            res[name] = (res[col] - min_) / (max_ - min_)
            # res[name]=res[col] / min_spore_0
        pass
        return res

    def _interquartiles(self, data=None) -> pd.DataFrame:
        """ Normalize values between 0 and 1"""

        """Assign values to 1 if below mode, and 2 if above mode."""
        if data is None:
            res = self.data
        else:
            res = data.copy()

        for col in res.columns:
            mode_val = res[col].mode().iloc[0]  # Moda de la columna
            name = str(col)

            # Asignar 1 si el valor está por debajo de la moda, 2 si está por encima o igual
            res[name] = res[col].apply(lambda x: 1 if x < mode_val else 2)

        return res

    def _join_impacts(self):
        """
        Join in the same df the energy inputs + results
        """

        # df_energy = pd.read_csv(self._starter_path, delimiter=',')
        df_energy = pd.read_csv(self._units_path)
        pass
        # df_energy = df_energy.drop(df_energy.columns[0], axis=1)
        df_energy = df_energy.groupby(['spores', 'techs'])['flow_out_sum'].sum().reset_index()
        self.energy_straight = df_energy
        df_energy = df_energy.pivot(index='spores', columns='techs', values='flow_out_sum')
        pass

        return pd.concat([self.normal_data, df_energy], axis=1)

    def _load_csv(self, results_path: str):
        return pd.read_csv(results_path)

    def _handle_nan_columns(self, col_name: pd.Series = 'scenario'):
        val_prev = np.nan
        pass
        filled_colum = []
        for value in col_name:
            if pd.isnull(value):
                filled_colum.append(val_prev)
            else:
                filled_colum.append(int(value))
                val_prev = int(value)
        return pd.Series(filled_colum)

    def _get_amounts_starter(self):
        data=pd.read_csv(self._units_path)
        data['names_official']=data['techs']+'__'+data['carriers']+'___'+data['locs']

        return data


    def define_groups(self):

        #split impacts and data

        techs=self.df.drop(columns=self.df.columns[:5], axis=1)
        impacts = self.df[self.df.columns[:5]]  # Output variables


        self.info={}
        grouped_df=pd.DataFrame()
        for group, columns in data_group.items():
            grouped_df[group] = techs[columns].sum(axis=1)

            self.info[group] = {
                'max': grouped_df[group].max(),
                'min': grouped_df[group].min(),
                'mean': grouped_df[group].mean(),
                'max_index': grouped_df[group].idxmax(),
                'min_index': grouped_df[group].idxmin(),

            }


        return pd.concat([impacts,grouped_df],axis=1)



    def plot(self):
        df=self.grouped[self.grouped.columns[:5]]  # Output variables
        colums=[x.split('(')[1].split(')')[0] for x in df.columns]
        df.columns=colums
        for k, v in self.info.items():
            mi_paleta = sns.color_palette("coolwarm", n_colors=5)
            sns.set(style="whitegrid", palette=mi_paleta)
            legend_labels = ['Spores']

            plt.figure(figsize=(18, 12))
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

            sns.stripplot(data=df, orient='v', color='grey', alpha=0.8, jitter=0.1, label=legend_labels[0])

            # Add the selected spore. Spore 0 is the one that we would get with TIMES
            max_index = v['max_index']
            min_index = v['min_index']
            df_max_selected = df.iloc[[max_index]]
            df_min_selected = df.iloc[[min_index]]

            # Graficar el valor máximo
            sns.stripplot(data=df_max_selected, orient='v', color='orange', jitter=0.2, size=9, marker='o',
                          label=f'{k} (Max)')

            # Graficar el valor mínimo
            sns.stripplot(data=df_min_selected, orient='v', color='purple', jitter=0.2, size=9, marker='o',
                          label=f'{k} (Min)')

            selected_indices = [min_index, max_index]
            df_selected = df.iloc[selected_indices]
            colors = ['purple', 'orange']

            for idx, row in zip(selected_indices, df_selected.iterrows()):
                row_index, row_data = row
                color = colors.pop(0)
                plt.plot(df.columns, row_data, marker='o', color=color, linestyle='--', linewidth=0.6, markersize=7,
                         alpha=0.8)

            # Configurar etiquetas y título
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)

            plt.xlabel("Impact Value", fontsize=18)
            plt.ylabel("Indicator", fontsize=18)
            plt.title(f" Tradeoffs {k}", fontsize=19)

            for l in plt.gca().lines:
                l.set_linewidth(2)

            plt.grid(False)
            handles, labels = plt.gca().get_legend_handles_labels()
            unique_labels = list(set(labels))
            unique_handles = [handles[labels.index(label)] for label in unique_labels]
            plt.legend(unique_handles, unique_labels, loc='upper right', fontsize='large')
            plt.tight_layout()
            plt.savefig(f'{k}.png', dpi=500, bbox_inches='tight')
            #plt.close()

    def plot_wind_solar(self):
        """Same as plot but specific for wind and solar in the same figure."""

        df = self.grouped[self.grouped.columns[:5]]  # Output variables
        colums = [x.split('(')[1].split(')')[0] for x in df.columns]
        df.columns = colums

        items_to_plot = ['wind', 'solar']
        color_mapping = {
            'wind': ('#add8e6', '#0000ff'),  # light blue for min, blue for max
            'solar': ('#ffcccb', '#ff4500')  # light orange for min, orange-red for max
        }

        sns.set(style="whitegrid")
        plt.figure(figsize=(18, 12))
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        legend_labels = ['Spores']
        sns.stripplot(data=df, orient='v', color='grey', alpha=0.8, jitter=0.1, size=9, label=legend_labels[0])

        for k in items_to_plot:
            if k in self.info:
                v = self.info[k]
                light_color, dark_color = color_mapping[k]

                # Add the selected spore. Spore 0 is the one that we would get with TIMES
                max_index = v['max_index']
                min_index = v['min_index']
                df_max_selected = df.iloc[[max_index]]
                df_min_selected = df.iloc[[min_index]]

                # Graficar el valor máximo
                sns.stripplot(data=df_max_selected, orient='v', color=dark_color, jitter=0.2, size=9, marker='o',
                              label=f'{k} (Max)')

                # Graficar el valor mínimo
                sns.stripplot(data=df_min_selected, orient='v', color=light_color, jitter=0.2, size=9, marker='o',
                              label=f'{k} (Min)')

                selected_indices = [min_index, max_index]
                df_selected = df.iloc[selected_indices]
                colors = [light_color, dark_color]

                for idx, row in zip(selected_indices, df_selected.iterrows()):
                    row_index, row_data = row
                    color = colors.pop(0)
                    plt.plot(df.columns, row_data, marker='o', color=color, linestyle='--', linewidth=0.6, markersize=7,
                             alpha=0.8)

        # Configurar etiquetas y título
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)

        plt.xlabel("Impact Value", fontsize=18)
        plt.ylabel("Indicator", fontsize=18)
        plt.title(f"Tradeoffs Wind and Solar", fontsize=19)

        for l in plt.gca().lines:
            l.set_linewidth(2)

        plt.grid(False)
        handles, labels = plt.gca().get_legend_handles_labels()
        unique_labels = list(set(labels))
        unique_handles = [handles[labels.index(label)] for label in unique_labels]
        plt.legend(unique_handles, unique_labels, loc='upper right', fontsize='large')
        plt.tight_layout()
        plt.savefig('wind_and_solar.png', dpi=500, bbox_inches='tight')



plot=PlottingEnbios(r'C:\Users\Administrator\PycharmProjects\SEEDS\runs\run_density_water\data\results_260.json',
                        flow_out_sum=r'C:\Users\Administrator\Downloads\flow_out_sum (1).csv',
                        starter=r'C:\Users\Administrator\PycharmProjects\SEEDS\runs\run_density_water\data\units_260.csv',
                        units=r'C:\Users\Administrator\PycharmProjects\SEEDS\runs\run_density_water\data\starter_260.csv')
plot.plot_wind_solar()