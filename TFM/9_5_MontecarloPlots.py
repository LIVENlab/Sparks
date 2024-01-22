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


class Montecarlo_viz:

    def __init__(self):
        self.general_results=self._get_data()
        pass

    def _get_data(self):

        out_results = {}
        with open('results_montecarlo.json') as file:
            d = json.load(file, object_pairs_hook=OrderedDict)
        pass
        d=d[0]['results']
        for k,v in d.items():
            results=v['multi_magnitude']
            out_results[k]=results

        self.results_general=out_results

        with open('results_TFM_update.json') as file:
            res=json.load(file,object_pairs_hook=OrderedDict)

        #spore 0
        res=res[0]['results']
        self.spore_0=res

        pass

        return out_results

    def viz(self):



        data_dict = self.general_results
        result_dict=self.spore_0
        keys = list(data_dict.keys())
        num_keys = len(keys)

        # Definir el número de filas y columnas en el subplot
        num_rows = num_keys // 2
        num_cols = 2 if num_keys % 2 == 0 else 1

        # Crear el subplot
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))


        # Convertir aplanar el array de ejes si es necesario
        axes = axes.flatten() if num_cols > 1 else [axes]


        axes = axes.flatten() if num_cols > 1 else [axes]

        for i, key in enumerate(keys):
            sns.histplot(data_dict[key], bins=60, kde=True, color='skyblue', ax=axes[i], label=key)
            axes[i].set_title(f'{key}')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')

            # Agregar línea vertical roja para el resultado específico
            result_value = result_dict[key]
            axes[i].axvline(result_value, color='red', linestyle='dashed', linewidth=2,
                            label='Spore Zero Deterministic')

            # Añadir leyenda compartida fuera de los subgráficos
            # Crear una línea roja ficticia para la leyenda
        legend_line = plt.Line2D([0], [0], color='red', linestyle='dashed', linewidth=2,
                                 label='Spore Zero Deterministic')

        # Añadir leyenda compartida fuera de los subgráficos
        fig.legend(handles=[legend_line], loc='upper right', bbox_to_anchor=(0.5, -0.1), frameon=False,
                   fontsize='medium')

        # Ajustes finales y mostrar el gráfico
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('plots/distribution_subplots.png', dpi=800)
        plt.show()


bby=Montecarlo_viz()
bby.viz()