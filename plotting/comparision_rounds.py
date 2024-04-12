import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from typing_extensions import OrderedDict
import seaborn as sns


class CompRounds:

    def __init__(self, round_1: str, round_2: str):

        self._260 = self.load_data(round_1)
        pass
        self._270 = self.load_data(round_2)

        self.get_normalized_values()
        self.get_normalized_second(self_max_min=False)
        self.get_stats()

    def load_data(self, path: str) -> pd.DataFrame:
        """" load the results and return it as a dataframe"""
        out_results = OrderedDict()
        with open(path) as file:
            d = json.load(file, object_pairs_hook=OrderedDict)

        for scen in range(len(d)):
            scen_name = scen
            out_results[scen_name] = d[scen]['results']
            data = pd.DataFrame.from_dict(out_results)
            data = data.T
        return data

    def get_normalized_values(self):
        """
            Normalize the results between 0 and 1 for the first round of spores
            -------
        """
        res = self._260
        info_norm = {}
        normalized_df = pd.DataFrame(index=res.index)  # Nuevo DataFrame para almacenar las columnas normalizadas
        for col in res.columns:
            name = str(col)

            min_spore_0 = res[col][0]
            min_ = np.min(res[col])
            max_ = np.max(res[col])

            normalized_df[name] = (res[col] - min_) / (max_ - min_)
            #normalized_df[name] = res[col] / min_
            # normalized_df[name] = res[col] / min_spore_0
            info_norm[col] = {
                "max": max_,
                "min": min_,
                "scenario_max": normalized_df[name].idxmax().item(),
                "scenario_min": normalized_df[name].idxmin().item()
            }
        normalized_df.columns = [col.rstrip('_') for col in normalized_df.columns]
        normalized_df['Round'] = '1'
        self._260_normalized = normalized_df
        return normalized_df

    def get_normalized_second(self, self_max_min=False):

        if self_max_min:
            min_values=self._270.min()
            max_values=self._270.max()
            pass
        else:
            min_values = self._260.min()
            max_values = self._260.max()
        pass
        normalized_second_df = (self._270 - min_values) / (max_values - min_values)
        #normalized_second_df = self._270 / min_values
        normalized_second_df['Round'] = '2'
        self._270_normalized = normalized_second_df
        return normalized_second_df

    def get_stats(self):
        self.stats_270 = self._270.describe().T
        self.stats_260 = self._260.describe().T

        # Realizar la división de los valores de cada columna entre los valores correspondientes de 260
        stats=self.stats_270 / self.stats_260
        pass

        #stats = (1 - stats) * 100
        stats = stats[['mean', 'std', 'min', 'max', '50%']]
        # Transponer nuevamente el dataframe para que las columnas representen los valores estadísticos y las filas los diferentes conjuntos de datos
        self.stats = stats.T
        pass


    def filter_data(self):
        # There is an extreme value in _270
        data=self._270_normalized
        return data[data['surplus ore potential (SOP)']<1.4]

    def compare_boxplots(self, filter_data=False):

        if filter_data:
            self._270_normalized=self.filter_data()

        combined_data = pd.concat([self._260_normalized, self._270_normalized], axis=0)

        combined_data.columns = [x.split("(")[0] for x in combined_data.columns]

        melted_data = pd.melt(combined_data, id_vars=['Round'], var_name='Variable', value_name='Value')


        plt.figure(figsize=(12, 8))  # Figure size
        sns.boxenplot(data=melted_data, x='Value', y='Variable', hue='Round', orient='h',
                      palette=['skyblue', 'lightgreen'])

        plt.yticks(size=13)
        # Label and title configuration
        plt.xlabel('Normalized Value', fontsize=16)  # x-axis label
        plt.ylabel('Impact Category', fontsize=16)  # y-axis label
        #plt.title("Comparison of normalized results distributions relative to their min and max values",
          #        fontsize=18)  # Figure title

        plt.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        plt.axvline(x=1, color='black', linestyle='--', linewidth=0.5)
        # Legend adjustments
        plt.legend(title='Round', title_fontsize='11', fontsize='11', loc='upper right')

        # Save the figure as an image
        plt.savefig('plot_1.png', dpi=800, bbox_inches='tight')

    def subplots(self, filter_data=False):
        if filter_data:
            self._270_normalized = self.filter_data()

        combined_data = pd.concat([self._260_normalized, self._270_normalized], axis=0)
        combined_data.columns = [x.split("(")[0] for x in combined_data.columns]
        melted_data = pd.melt(combined_data, id_vars=['Round'], var_name='Variable', value_name='Value')

        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12, 18))
        sns.boxenplot(data=melted_data[melted_data['Variable'] == combined_data.columns[0]], x='Value', y='Variable',
                      hue='Round', palette=['skyblue', 'lightgreen'], ax=ax[0, 0])
        #ax[0, 0].set_xlabel('Normalized Value', fontsize=12)  # x-axis label
        ax[0, 0].set_title(combined_data.columns[0], fontsize=14)  # Title for the subplot
        plt.axvline(x=0, color='black', linestyle='--', linewidth=0.5, ax=ax[0,0])
        plt.axvline(x=1, color='black', linestyle='--', linewidth=0.5, ax= ax[0,0])

        # Save the figure as an image
        plt.savefig('fig_1_cut.png', dpi=800, bbox_inches='tight')

        plt.show()

        plt.show()







res = CompRounds(r'C:\Users\Administrator\PycharmProjects\SEEDS\results\results_260.json', r'C:\Users\Administrator\PycharmProjects\SEEDS\results\results_270.json')
res.compare_boxplots(filter_data=True)
#res.subplots()
#res.plot_stats()























