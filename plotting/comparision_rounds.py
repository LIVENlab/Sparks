import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from typing_extensions import OrderedDict
import seaborn as sns
from scipy.stats import mannwhitneyu

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

    def compare_boxplots(self, save_path : str,  filter_data=False):

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
        plt.savefig(save_path, dpi=800, bbox_inches='tight')



    def plot_stats(self, save_path):

        sns.set(style='whitegrid')

        # Convertir los datos a formato 'long' utilizando la función melt de pandas
        melted_stats = self.stats.reset_index().melt(id_vars='index', var_name='Métrica', value_name='Diferencia')

        plt.figure(figsize=(12, 8))  # Figure size
        ax = sns.barplot(data=melted_stats, x='Diferencia', y='Métrica', hue='index', orient='h', palette='Set3')

        # Añadir línea vertical en x=1
        plt.axvline(x=1, color='black', linestyle='--')

        # Añadir título y etiquetas
        plt.title('Statistical Differences', fontsize=16)
        plt.xlabel('Difference', fontsize=14)
        #plt.ylabel('Métrica', fontsize=14)

        # Cambiar el título de la leyenda
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, title='Statistics')
        plt.savefig(save_path, dpi=800, bbox_inches='tight')



    def mann_whitney_test(self):
        """ The Mann-Whitney U test is a nonparametric test of the null hypothesis
    that the distribution underlying sample `x` is the same as the
    distribution underlying sample `y`. It is often used as a test of
    difference in location between distributions"""
        df1=self._260_normalized.drop(columns=['Round'])
        df2=self._270_normalized.drop(columns=['Round'])
        pass

        for col in df1.columns:
            statistic, p_value = mannwhitneyu(df1[col], df2[col], alternative='two-sided')
            print(f'Comparación para la categoría {col}:')
            print(f'Estadística de prueba: {statistic}')
            print(f'Valor p: {p_value}')
            print('')







res = CompRounds(r'C:\Users\Administrator\PycharmProjects\SEEDS\runs\run_density_water\data\results_260.json', r'C:\Users\Administrator\PycharmProjects\SEEDS\runs\run_density_water\data\results_270.json')
res.compare_boxplots(filter_data=True, save_path='plots/boxplots.png')
#res.subplots()
#res.plot_stats(save_path='plots/stats.png')
res.mann_whitney_test()























