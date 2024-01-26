
import pandas as pd
import shap
import numpy as np
import json

import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.outliers_influence  import variance_inflation_factor
from sklearn.metrics import r2_score, davies_bouldin_score, pairwise_distances_argmin_min, confusion_matrix
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from collections import OrderedDict
def get_general_results():

    out_results=OrderedDict()
    with open('results_TFM_update.json') as file:
        d=json.load(file, object_pairs_hook=OrderedDict)
    pass
    for scen in range(len(d)):
        scen_name=scen
        out_results[scen_name]=d[scen]['results']
        pass
        a=pd.DataFrame.from_dict(out_results)
        pass
        a=a.T
    a=a.drop(['exergy content',"global hectares","water pollutants","global temperature change potential (GTP100)"],axis=1)
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

    normalized_df = pd.DataFrame(index=res.index)  # Nuevo DataFrame para almacenar las columnas normalizadas

    for col in res.columns:
        name = str(col)
        name_modified = name + "_"

        min_ = np.min(res[col])
        max_ = np.max(res[col])
        normalized_df[name_modified] = (res[col] - min_) / (max_ - min_)
        info_norm[col] = {
            "max": max_,
            "min": min_,
            "scenario_max": normalized_df[name_modified].idxmax().item(),
            "scenario_min": normalized_df[name_modified].idxmin().item()
        }
    normalized_df.columns = [col.rstrip('_') for col in normalized_df.columns]

    return normalized_df



import seaborn as sns


def visualize1(spore=None):
    """
    General Violin plot of results
    """

    df = get_normalized_value()

    mi_paleta = sns.color_palette("coolwarm", n_colors=5)  # Cambia la paleta aquí
    sns.set(style="whitegrid", palette=mi_paleta)
    legend_labels = ['Spore', 'Cost-Optimized Spore']

    # Configurar el tamaño del gráfico
    plt.figure(figsize=(19, 13))

    # Ajustar el espacio alrededor del gráfico
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)

    # Crear un gráfico de violín horizontal
    ax = sns.violinplot(data=df, orient='h', inner='quart', cut=0, palette='Set2', alpha=0.6)

    # Configurar etiquetas y título con ajuste de fontsize
    ax.set_xlabel("Impact", fontsize=14)
    ax.set_ylabel("Environmental Indicators", fontsize=16)

    # Configurar el tamaño de fuente de los ticks en los ejes
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Configurar el tamaño de fuente de los ticks en el eje y
    ax.tick_params(axis='y', which='major', labelsize=17)

    # Configurar el título con un color más oscuro
    #ax.set_title("Normalized environmental impacts in Portugal", fontsize=18, color='black', fontweight='bold')

    sns.stripplot(data=df, orient='h', color='black', alpha=0.4, jitter=0.2, label=legend_labels[0])

    # Add the selected spore. Spore 0 is the one that we would get with TIMES
    if spore is not None:
        df_selected = df.iloc[[spore]]
        sns.stripplot(data=df_selected, orient='h', color='red', jitter=1, size=18, marker='*',
                      label=legend_labels[1])

    for l in ax.lines:
        l.set_linewidth(2)

    # Mostrar el gráfico
    plt.grid(True)

    # Añadir una leyenda única usando Matplotlib
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = list(set(labels))
    unique_handles = [handles[labels.index(label)] for label in unique_labels]

    # Ajustar la ubicación de la leyenda
    plt.legend(unique_handles, unique_labels, loc='upper right', fontsize='medium')
    plt.savefig('plots/violin_plot.png', dpi=1500, bbox_inches='tight')
    plt.legend(fontsize='medium')

visualize1(spore=0)

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import metrics
def clusters():
    df=get_normalized_value()
    df=df.drop('biotic resources', axis=1)
    X=df.values
    scaler=StandardScaler()
    X_scaled =scaler.fit_transform(X)

    n_clusters=2

    k_means=KMeans(n_clusters=n_clusters,random_state=42)
    df['cluster']=k_means.fit_predict(X_scaled)
    inertia = k_means.inertia_

    # Calcular la silueta
    silhouette_score = metrics.silhouette_score(X_scaled, df['cluster'])
    davies_bouldin=davies_bouldin_score(X_scaled, df['cluster'])



    # Visualizar la matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(df['cluster'], df['cluster']), annot=True, fmt='d', cmap='viridis', cbar=False)
    plt.xlabel('Predicho')
    plt.ylabel('Verdadero')
    plt.title('Matriz de Confusión')
    plt.savefig('plots/confusion.png')



    # Visualizar la matriz de c

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
    plt.show()

    conteo = df['cluster'].value_counts()
    print('##CONTEO###')
    print(conteo)
    # Verificar los resultados
    b=df.groupby('cluster').mean()

    return df,b



def visualize_single_column(column_name):
    """
    Violin plot and strip plot for a single column
    """

    # Supongamos que df es tu DataFrame después de la agrupación por kmeans
    # y que has añadido la columna "cluster"
    b1, df = clusters()
    df = b1
    pass

    # Ajusta la paleta de colores para tener dos colores, uno para cada valor de "cluster"
    mi_paleta = sns.color_palette("coolwarm", n_colors=2)

    sns.set(style="whitegrid", palette=mi_paleta)
    legend_labels = ['Spore', 'Cost-Optimized Spore']

    # Configurar el tamaño del gráfico
    plt.figure(figsize=(10, 6))

    # Ajustar el espacio alrededor del gráfico
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.2)

    # Crear un gráfico de violín horizontal para la columna específica
    ax = sns.violinplot(x=df[column_name], orient='h', inner='quart', cut=0, palette='Set2', alpha=0.1, linewidth=1)  # Ajusta el alpha y el linewidth aquí

    # Añadir los puntos del stripplot con diferente color según el valor de "cluster"
    sns.stripplot(x=df[column_name], orient='h', hue=df["cluster"], palette=mi_paleta, alpha=0.7, jitter=0.2, ax=ax, size=5, linewidth=0.5)  # Ajusta el alpha y el size aquí

    # Configurar etiquetas y título con ajuste de fontsize
    ax.set_xlabel("Relative Impact", fontsize=12)
    ax.set_ylabel(column_name, fontsize=12)

    # Configurar el tamaño de fuente de los ticks en los ejes
    ax.tick_params(axis='both', which='major', labelsize=8)  # Ajusta el labelsize aquí

    # Configurar el tamaño de fuente de los ticks en el eje y
    ax.tick_params(axis='y', which='major', labelsize=12)  # Ajusta el labelsize aquí

    # Configurar las líneas de la cuadrícula
    ax.grid(which='both', linestyle='-', linewidth='0.2', color='gray')  # Ajusta el linestyle, linewidth y color aquí

    # Añadir una leyenda única usando Matplotlib
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = list(set(labels))
    unique_handles = [handles[labels.index(label)] for label in unique_labels]

    # Cambiar el título de la leyenda

    legend = ax.legend(unique_handles, unique_labels, loc='upper right', fontsize='small', title='Cluster Group')
    legend.get_title().set_fontsize('10')  # Ajusta el tamaño del título de la leyenda
    plt.tight_layout(pad=2)


    # Guardar el gráfico con alta calidad
    plt.savefig(f'plots/violin_strip_plot_{column_name}.png', dpi=2000, bbox_inches='tight')  # Ajusta el dpi aquí
    plt.show()


# Llamada a la función para una columna específica
#visualize_single_column("water consumption potential (WCP)")










pass

def join_impacts_input():
    """
    Join in the same df the energy inputs + results
    """
    df_energy = pd.read_csv(r'flow_out_processed.csv',delimiter=',')
    pass
    #df_energy=df_energy.drop(df_energy.columns[0],axis=1)
    df_energy = df_energy.groupby(['scenarios', 'techs'])['flow_out_sum'].sum().reset_index()
    df_energy = df_energy.pivot(index='scenarios', columns='techs', values='flow_out_sum')
    pass

    res = get_normalized_value()
    pass

    result = pd.concat([res, df_energy], axis=1)
    pass
    return result


aa = join_impacts_input
#visualize1(spore=0)

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
        plt.savefig(name, dpi=1800, bbox_inches='tight')

        # Mostrar el SHAP plot en el notebook (opcional)



#study_result()


import seaborn as sns
import matplotlib.pyplot as plt

def correlation_matrix_analysis():
    """
    Analyze the correlation among the outputs (impacts).
    (Simple correlation)
    """
    # Assuming `join_impacts_input()` returns the required DataFrame
    df = join_impacts_input()

    # Drop unnecessary columns
    df = df.drop(columns=df.columns[8:])

    # Calculate correlation matrix
    correlation_matrix = df.corr()

    # Set up the matplotlib figure
    fig = plt.figure(figsize=(10, 8))

    # Customize the seaborn style for a clean and professional look
    sns.set(style="whitegrid")

    # Customize the heatmap with the "coolwarm" color palette and white borders
    heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', cbar=True, fmt=".2f", linewidths=.5,
                          linecolor='white', square=True)
    plt.title('n-Energy system', fontsize=16)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=35, ha="right", )
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)

    # Remove the colorbar created by Seaborn
    heatmap.collections[0].colorbar.remove()

    # Add a vertical title to the colorbar
    cbar = fig.colorbar(heatmap.collections[0], fraction=0.046, pad=0.04)
    cbar.set_label('Correlation Index r', rotation=270, labelpad=15)

    # Save the plot with high resolution and tight bounding box
    plt.savefig("plots/correlation_matrix.png", dpi=1900, bbox_inches='tight')

    # Show the plot
    plt.show()


correlation_matrix_analysis()
# Call the function to generate the heatmap



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


#multi_colinearity()


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
    X = df.drop(columns=df.columns[:11], axis=1)
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
    plt.savefig("plots/pearson.png", dpi=1800, bbox_inches='tight')



    return correlation_matrix


#pearrr = Pearson_correlation()

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
    X = df.drop(columns=df.columns[:8], axis=1)  # Predictor features
    Y = df[df.columns[:8]]  # Output variables

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
    plt.xticks(rotation=45, ha='right')

    plt.savefig('plots/sparman.png', dpi=1800, bbox_inches='tight')

    return correlation_matrix

correlation_matrix = Spearman_correlation()

import numpy as np
from scipy.stats import kendalltau


def Kendall():
    df = join_impacts_input()
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
    plt.savefig('plots/kendall.png', dpi=1800, bbox_inches='tight')

# Example usage:
#Kendall()


####### part 2: Second level of the dendrogram

