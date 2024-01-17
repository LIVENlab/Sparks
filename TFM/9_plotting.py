
import pandas as pd
import shap
import numpy as np
import json

import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.outliers_influence  import variance_inflation_factor
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from collections import OrderedDict
def get_general_results():

    out_results=OrderedDict()
    with open('results_TFM.json') as file:
        d=json.load(file, object_pairs_hook=OrderedDict)
    pass
    for scen in range(len(d)):
        scen_name=scen
        out_results[scen_name]=d[scen]['results']
        a=pd.DataFrame.from_dict(out_results)
        pass
        a=a.T
    a=a.drop('global temperature change potential (GTP100)',axis=1)
    pass
    return a



def get_normalized_value():
    """
    Normalize the results between 0 and 1
    -------
    Return: modified DF
    """

    res = get_general_results()
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


def visualize1(spore: None):
    """
    General Violin plot of results
    """
    df = get_normalized_value()
    pass

    mi_paleta = sns.color_palette("coolwarm", n_colors=5)  # Cambia la paleta aquí
    sns.set(style="whitegrid", palette=mi_paleta)
    legend_labels = ['Spore', 'Cost-Optimized Spore']
    # Configurar el tamaño del gráfico
    plt.figure(figsize=(18, 12))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    # Crear un gráfico de violín horizontal
    sns.violinplot(data=df, orient='h', inner='quart', cut=0, palette='Set2',alpha=0.6)
    # Configurar etiquetas y título
    plt.xlabel("Valor Impact")
    plt.ylabel("Indicator")
    # sns.set_yticklabels(df.columns, fontsize=9)  # Cambia el tamaño de fuente aquí
    sns.stripplot(data=df, orient='h', color='black', alpha=0.4, jitter=0.2,label=legend_labels[0])

    # Add the selected spore. Spore 0 is the one that we would get with TIMES
    if spore is not None:
        df_selected = df.iloc[[spore]]
        sns.stripplot(data=df_selected, orient='h', color='red', jitter=1, size=13, marker='*',
                      label=legend_labels[1])
    pass
    plt.title("Normalized environmental impacts in Portugal", fontsize=18)

    for l in plt.gca().lines:
        l.set_linewidth(2)

    # Mostrar el gráfico
    plt.grid(True)
    # Añadir una leyenda única usando Matplotlib
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = list(set(labels))
    unique_handles = [handles[labels.index(label)] for label in unique_labels]
    plt.legend(unique_handles, unique_labels, loc='upper right', fontsize='medium')
    # Guardar el gráfico en alta resolución
    plt.savefig('plots/violin_plot.png', dpi=900, bbox_inches='tight')
    plt.legend(fontsize='medium')
    # Mostrar el gráfico


visualize1(spore=0)
pass

def join_impacts_input():
    """
    Join in the same df the energy inputs + results
    """
    df_energy = pd.read_csv(r'flow_out_clean.csv',delimiter=',')
    df_energy=df_energy.drop(df_energy.columns[0],axis=1)
    df_energy = df_energy.groupby(['scenarios', 'techs'])['flow_out_sum'].sum().reset_index()
    df_energy = df_energy.pivot(index='scenarios', columns='techs', values='flow_out_sum')
    pass

    res = get_normalized_value()
    pass

    result = pd.concat([res, df_energy], axis=1)
    pass
    return result


aa = join_impacts_input


import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor


def study_result():
    """
    Analyze the resutls of the different simulations:
        -Train a random forest regressor
        -Print accuracy
        -Print Feature importance


    """

    # Obtener los datos totales
    tot = join_impacts_input()
    X = tot.drop(columns=tot.columns[:11])
    Y = tot[tot.columns[:11]]
    pass

    # Crear un modelo de Random Forest
    model = RandomForestRegressor(n_estimators=400, max_depth=20, random_state=42)

    # Inicializar listas para almacenar los resultados
    test_scores = []
    cv_mean_scores = []
    cv_std_scores = []

    # Iterar a través de cada objetivo
    for goal in Y.columns:
        X_train, X_test, y_train, y_test = train_test_split(X, Y[goal], test_size=0.2, random_state=42)

        # Entrenar el modelo de Random Forest
        model.fit(X_train, y_train)

        # Calcular la precisión en datos de prueba (R-cuadrado)
        test_pred = model.predict(X_test)
        test_r2 = r2_score(y_test, test_pred)
        test_scores.append(test_r2)

        # Realizar validación cruzada para evaluar el rendimiento del modelo
        cv_scores = cross_val_score(model, X, Y[goal], cv=5, scoring='r2')
        cv_mean_scores.append(cv_scores.mean())
        cv_std_scores.append(cv_scores.std())

        # Crear un objeto explainer SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, features=X, feature_names=X.columns)
        shap.summary_plot(shap_values, features=X, feature_names=X.columns, show=False)
        plt.title(f"Shapley feature importance {goal}", fontsize=18)

        # Guardar la figura en un archivo
        name = 'plots/p_' + goal + '.png'
        plt.savefig(name, dpi=800, bbox_inches='tight')

        # Mostrar el SHAP plot en el notebook (opcional)



#study_result()


def corr_matrix_Y():
    """
    Study the correlation among the outputs (impacts)
    (Simple correlation)

    """
    df = join_impacts_input()
    df = df.drop(columns=df.columns[11:])
    pass

    correlation_matrix = df.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', cbar=True)
    plt.title('Correlación entre Objetivos')
    plt.savefig("plots/correlation_matrix.png", dpi=800, bbox_inches='tight')



corr_matrix_Y()


def multi_colinearity():
    """


    """
    # Calcular el VIF para las características y objetivos
    df = join_impacts_input()

    X = df.drop(columns=df.columns[:11], axis=1)  # Características predictoras

    X['Intercept'] = 1  # Agregar término constante para calcular el VIF
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print(vif)


multi_colinearity()


def Pearson_correlation():
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
    df = join_impacts_input()
    X = df.drop(columns=df.columns[:11], axis=1)  # Características predictoras
    Y = df[df.columns[:11]]
    correlation_matrix = pd.DataFrame(index=X.columns, columns=Y.columns)

    for col in Y.columns:
        var_y = Y[col]
        for cols in X.columns:
            var_x = X[cols]
            correlation, _ = pearsonr(var_x, var_y)

            # Almacenar el coeficiente en la matriz de correlación
            correlation_matrix.loc[cols, col] = correlation
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix.astype(float), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Pearson Correlation Matrix (Impacts-Technology)', fontsize=18)
    plt.savefig("plots/pearson.png", dpi=800, bbox_inches='tight')



    return correlation_matrix


pearrr = Pearson_correlation()

from scipy.stats import spearmanr


def Spearman_correlation():
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
    df = join_impacts_input()
    pass

    # Splitting the DataFrame into predictor variables (X) and output variables (Y)
    X = df.drop(columns=df.columns[:11], axis=1)  # Predictor features
    Y = df[df.columns[:11]]  # Output variables

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

    plt.title('Spearman Correlation Matrix (Impacts-Technology)', fontsize=18)
    plt.savefig('plots/sparman.png', dpi=800, bbox_inches='tight')


    return correlation_matrix


# Call the function to compute and visualize the Spearman correlation
correlation_matrix = Spearman_correlation()

import numpy as np
from scipy.stats import kendalltau


def Kendall():
    df = join_impacts_input()  # You'll need to define your 'join_impacts_input' function.
    # Splitting the DataFrame into predictor variables (X) and output variables (Y)
    X = df.drop(columns=df.columns[:11], axis=1)  # Predictor features
    Y = df[df.columns[:11]]  # Output variables
    correlation_matrix = pd.DataFrame(index=X.columns, columns=Y.columns)

    for col_y in Y.columns:
        for col_x in X.columns:
            kendall_corr, _ = kendalltau(X[col_x], Y[col_y])
            correlation_matrix.loc[col_x, col_y] = kendall_corr

    # Visualize the Kendall correlation matrix
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix.astype(float), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Kendall Correlation between Predictor Variables (X, Technologies) and Output Variables (Y, Impacts)')
    plt.xlabel('Output Variables (Y)')
    plt.ylabel('Predictor Variables (X)')
    plt.title('Kendall Correlation Matrix (Impacts-Technology)', fontsize=18)
    plt.savefig('plots/kendall.png', dpi=800, bbox_inches='tight')

# Example usage:
Kendall()


####### part 2: Second level of the dendrogram

def get_general_results():


    out_results=OrderedDict()
    with open('results_TFM.json') as file:
        d=json.load(file, object_pairs_hook=OrderedDict)

    for scen in range(len(d)):
        scen_name=scen

        out_results[scen_name]=d[scen]['results']
        a=pd.DataFrame.from_dict(out_results)
        a=a.T
    a=a.drop('global temperature change potential (GTP100)',axis=1)
    pass
    return a