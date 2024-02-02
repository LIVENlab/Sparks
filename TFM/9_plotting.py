
import pandas as pd
import shap

import json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import metrics

from statsmodels.stats.outliers_influence  import variance_inflation_factor
from sklearn.metrics import r2_score, davies_bouldin_score, pairwise_distances_argmin_min, confusion_matrix
from scipy.stats import pearsonr
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

    plt.figure(figsize=(19, 13))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)

    ax = sns.violinplot(data=df, orient='h', inner='quart', cut=0, palette='Set2', alpha=0.6)
    ax.set_xlabel("Impact", fontsize=14)
    ax.set_ylabel("Environmental Indicators", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='y', which='major', labelsize=17)

    sns.stripplot(data=df, orient='h', color='black', alpha=0.4, jitter=0.2, label=legend_labels[0])

    # Add the selected spore. Spore 0 is the one that we would get with TIMES
    if spore is not None:
        df_selected = df.iloc[[spore]]
        sns.stripplot(data=df_selected, orient='h', color='red', jitter=1, size=18, marker='*',
                      label=legend_labels[1])

    for l in ax.lines:
        l.set_linewidth(2)

    plt.grid(True)
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = list(set(labels))
    unique_handles = [handles[labels.index(label)] for label in unique_labels]

    plt.legend(unique_handles, unique_labels, loc='upper right', fontsize='medium')
    plt.savefig('plots/violin_plot.png', dpi=1500, bbox_inches='tight')
    plt.legend(fontsize='medium')


visualize1(spore=0)




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

    plt.figure(figsize=(12, 6))

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


    b1, df = clusters()
    df = b1
    pass

    # Ajusta la paleta de colores para tener dos colores, uno para cada valor de "cluster"
    mi_paleta = sns.color_palette("coolwarm", n_colors=2)

    sns.set(style="whitegrid", palette=mi_paleta)
    legend_labels = ['Spore', 'Cost-Optimized Spore']

    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.2)

    ax = sns.violinplot(x=df[column_name], orient='h', inner='quart', cut=0, palette='Set2', alpha=0.1, linewidth=1)

    sns.stripplot(x=df[column_name], orient='h', hue=df["cluster"], palette=mi_paleta, alpha=0.7, jitter=0.2, ax=ax, size=5, linewidth=0.5)
    ax.set_xlabel("Relative Impact", fontsize=12)
    ax.set_ylabel(column_name, fontsize=12)


    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='y', which='major', labelsize=12)
    ax.grid(which='both', linestyle='-', linewidth='0.2', color='gray')
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = list(set(labels))
    unique_handles = [handles[labels.index(label)] for label in unique_labels]
    legend = ax.legend(unique_handles, unique_labels, loc='upper right', fontsize='small', title='Cluster Group')
    legend.get_title().set_fontsize('10')
    plt.tight_layout(pad=2)
    plt.savefig(f'plots/violin_strip_plot_{column_name}.png', dpi=2000, bbox_inches='tight')  # Ajusta el dpi aquí
    plt.show()


#visualize_single_column("water consumption potential (WCP)")

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
    tot = join_impacts_input()
    X = tot.drop(columns=tot.columns[:11])
    Y = tot[tot.columns[:11]]
    pass
    model = RandomForestRegressor(n_estimators=400, max_depth=20, random_state=42)


    test_scores = []
    cv_mean_scores = []
    cv_std_scores = []


    for goal in Y.columns:
        X_train, X_test, y_train, y_test = train_test_split(X, Y[goal], test_size=0.2, random_state=42)


        model.fit(X_train, y_train)
        test_pred = model.predict(X_test)
        test_r2 = r2_score(y_test, test_pred)
        test_scores.append(test_r2)

        cv_scores = cross_val_score(model, X, Y[goal], cv=5, scoring='r2')
        cv_mean_scores.append(cv_scores.mean())
        cv_std_scores.append(cv_scores.std())

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, features=X, feature_names=X.columns)
        shap.summary_plot(shap_values, features=X, feature_names=X.columns, show=False)
        plt.title(f"Shapley feature importance {goal}", fontsize=18)

        name = 'plots/p_' + goal + '.png'
        plt.savefig(name, dpi=1800, bbox_inches='tight')





#study_result()


import seaborn as sns
import matplotlib.pyplot as plt

def correlation_matrix_analysis():
    """
    Analyze the correlation among the outputs (impacts).
    (Simple correlation)
    """

    df = join_impacts_input()

    df = df.drop(columns=df.columns[8:])


    correlation_matrix = df.corr()

    fig = plt.figure(figsize=(10, 8))


    sns.set(style="whitegrid")


    heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', cbar=True, fmt=".2f", linewidths=.5,
                          linecolor='white', square=True)
    plt.title('n-Energy system', fontsize=16)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=35, ha="right", )
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    heatmap.collections[0].colorbar.remove()

    cbar = fig.colorbar(heatmap.collections[0], fraction=0.046, pad=0.04)
    cbar.set_label('Correlation Index r', rotation=270, labelpad=15)
    plt.savefig("plots/correlation_matrix.png", dpi=1900, bbox_inches='tight')
    plt.show()


correlation_matrix_analysis()




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


    X = df.drop(columns=df.columns[:8], axis=1)
    Y = df[df.columns[:8]]

    correlation_matrix = pd.DataFrame(index=X.columns, columns=Y.columns)

    for col in Y.columns:
        var_y = Y[col]
        for col_x in X.columns:
            var_x = X[col_x]
            correlation, _ = spearmanr(var_x, var_y)
            correlation_matrix.loc[col_x, col] = correlation

    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix.astype(float), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.xticks(rotation=45, ha='right')

    plt.savefig('plots/sparman.png', dpi=1800, bbox_inches='tight')

    return correlation_matrix

def Spearman_correlation_index():
    """
    Compute the Spearman correlation between inputs and outputs along with the Sensitivity Index.

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
    X = df.drop(columns=df.columns[:8], axis=1)
    Y = df[df.columns[:8]]

    correlation_matrix = pd.DataFrame(index=X.columns, columns=Y.columns)
    pass
    for col in Y.columns:
        var_y = Y[col]
        for col_x in X.columns:
            var_x = X[col_x]
            correlation, _ = spearmanr(var_x, var_y)
            sensitivity_index = (1 - correlation**2) / 2 #TODO: CHECK INDEX
            correlation_matrix.loc[col_x, f'{col}_Spearman_Correlation'] = correlation
            correlation_matrix.loc[col_x, f'{col}_Sensitivity_Index'] = sensitivity_index
    pass
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix.astype(float), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.xticks(rotation=45, ha='right')
    plt.savefig('plots/spearman_.png', dpi=1800, bbox_inches='tight')


correlation_matrix = Spearman_correlation()

correlation_matrix_2=Spearman_correlation_index()

import numpy as np
from scipy.stats import kendalltau

def Kendall():
    df = join_impacts_input()

    X = df.drop(columns=df.columns[:11], axis=1)  # Predictor features
    Y = df[df.columns[:11]]  # Output variables
    correlation_matrix = pd.DataFrame(index=X.columns, columns=Y.columns)

    for col_y in Y.columns:
        for col_x in X.columns:
            kendall_corr, _ = kendalltau(X[col_x], Y[col_y])
            correlation_matrix.loc[col_x, col_y] = kendall_corr

    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix.astype(float), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Kendall Correlation between Predictor Variables (X, Technologies) and Output Variables (Y, Impacts)')
    plt.xlabel('Output Variables (Y)')
    plt.ylabel('Predictor Variables (X)')
    plt.title('Kendall Correlation Matrix (Impacts-Technology)', fontsize=18)
    plt.savefig('plots/kendall.png', dpi=1800, bbox_inches='tight')

Kendall()

