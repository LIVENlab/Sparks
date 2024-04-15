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
        self.results_csv= None



        # Basic actions
        self.data=self._get_general_results(self._path)
        self.normal_data=self._normalize_values()
        self.starter=self._get_amounts_starter()

    def _get_general_results(self, *args):

        out_results = OrderedDict()
        with open(*args) as file:
            self._raw_data = json.load(file, object_pairs_hook=OrderedDict)
        for scen in range(len(self._raw_data)):
            scen_name = scen
            out_results[scen_name] = self._raw_data[scen]['results']
            data = pd.DataFrame.from_dict(out_results).T
        return data

    def _normalize_values(self,data=None) -> pd.DataFrame:
        """ Normalize values between 0 and 1"""
        pass
        if data is None:
            res = self.data
        else:
            res=data

        for col in res.columns:
            name = str(col)
            min_ = np.min(res[col])
            max_ = np.max(res[col])
            res[name] = (res[col] - min_) / (max_ - min_)

        return res

    def _join_impacts(self):
        """
        Join in the same df the energy inputs + results
        """
        #TODO: CHECK IF UNITS
        #df_energy = pd.read_csv(self._starter_path, delimiter=',')
        df_energy= pd.read_csv(self._units_path)
        pass
        #df_energy = df_energy.drop(df_energy.columns[0], axis=1)
        df_energy = df_energy.groupby(['spores', 'techs'])['flow_out_sum'].sum().reset_index()
        self.energy_straight=df_energy
        df_energy = df_energy.pivot(index='spores', columns='techs', values='flow_out_sum')


        return pd.concat([self._normalize_values(), df_energy], axis=1)



    def _load_csv(self, results_path:str):
        return pd.read_csv(results_path)



    def _handle_nan_columns(self, col_name: pd.Series ='scenario'):
        val_prev=np.nan
        pass
        filled_colum=[]
        for value in col_name:
            if pd.isnull(value):
                filled_colum.append(val_prev)
            else:
                filled_colum.append(int(value))
                val_prev=int(value)
        return pd.Series(filled_colum)

    def _get_amounts_starter(self):
        data=pd.read_csv(self._units_path)
        data['names_official']=data['techs']+'__'+data['carriers']+'___'+data['locs']

        return data

    def _handle_columns(self, df, level:int):
        #filter columns according to the desired level
        df=df.iloc[: , [0] + list(range(level+1, len(df.columns)))]
        df=df.drop(['unit'], axis=1)
        df=df.dropna(subset=['lvl_'+str(level)])
        df=df[df['scenario']==0]
        # load the unit data
        amounts=self.starter

        values=[]
        for scen,tech in zip(df['scenario'],df['lvl_' + str(level)]):
            amounts_temp=amounts[amounts['spores']==scen]
            for i, v in amounts_temp.iterrows():
                if tech==v['names_official']:
                    values.append(v['flow_out_sum'])

        df['flow_out']=values

        ind_amounts=df.columns.get_loc('amount')
        columns_to_divide=df.columns[ind_amounts : -1]
        #divide the impact

        df[columns_to_divide]=df[columns_to_divide].div(df['flow_out'], axis=0)

        # final modifications

        cols_to_filter=['lvl_'+str(level)] + list(columns_to_divide)
        df=df[cols_to_filter]
        df = df.drop(columns=['amount'])
        df=df.rename(columns={'lvl_'+str(level) : 'technologies'})
        df['technologies']=[x.split('___')[0] for x in df['technologies']]
        df.set_index('technologies', inplace=True)


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


    def _clean_csv(self, path_csv, level):
        #applies different changes
        data = self._load_csv(path_csv)
        data['scenario'] = self._handle_nan_columns(data['scenario'])
        data=self._handle_columns(data, level)

        datum=data.copy()
        self._impact_tech=datum
        data=self._normalize_values(data=data)
        self._impact_tech_normalized=data

        return data



    def intensive_plot(self,path_csv, level, path_save):
        data=self._clean_csv(path_csv, level)


        plt.figure(figsize=(10, 8))  # Ajusta el tamaño de la figura según tus necesidades
        #sns.heatmap(data, cmap='Greens',annot=self._impact_tech,
           #         linewidths=.0)  # Ajusta la paleta de colores y otras propiedades

        sns.heatmap(data, cmap='Greens', linewidths=.1)  # Ajusta la paleta de colores y otras propiedades
        # Personaliza el heatmap
        plt.title('Normalized impact values')  # Agrega un título al heatmap
        plt.xticks(rotation=45, horizontalalignment='right')
        plt.tight_layout()
        # Guarda la figura si es necesario
        plt.savefig(path_save, dpi=800)  # Ajusta la resolución y los márgenes según tus necesidades

        # Muestra el heatmap
        plt.show()

    def intensive_plot_absolute(self, path_csv, level, path_save):
        df = self._clean_csv(path_csv, level)
        fig, axs = plt.subplots(nrows=1, ncols=5,
                                figsize=(15, 4))  # Ajusta el tamaño de la figura según tus necesidades
        for i, columna in enumerate(df.columns):
            sns.heatmap(self._impact_tech_normalized[[columna]], cmap='Greens', annot=self._impact_tech[[columna]], fmt='.2f', cbar=False,
                        ax=axs[i])  # Utiliza la misma paleta de colores para todas las columnas
            axs[i].set_title(columna)  # Agrega un título a cada heatmap
            axs[i].set_ylabel('')  # Elimina la etiqueta del eje y para ahorrar espacio

        # Agrega una única leyenda adimensional para todos los heatmaps
        cbar_ax = fig.add_axes([0.95, 0.1, 0.03, 0.8])  # Ajusta la posición de la leyenda según tus necesidades
        sns.heatmap(df.iloc[0:1], cmap='Greens', cbar=True, cbar_ax=cbar_ax)
        cbar_ax.set_ylabel('Adimensional')  # Agrega una etiqueta a la leyenda
        plt.savefig(path_save, dpi=800)
        plt.show()



    def _gini(self,series):
        n = len(series)
        sorted_values = series.sort_values()  # Ordena los valores de la serie
        cumulative_demand = sorted_values.cumsum()  # Calcula la demanda acumulada
        total_demand = cumulative_demand.iloc[-1]  # Total de la demanda
        lorenz_curve = cumulative_demand / total_demand  # Curva de Lorenz
        area_lorenz_curve = lorenz_curve.sum() / n  # Área bajo la curva de Lorenz
        area_perfect_equality = 0.5  # Área debajo de la línea de igualdad
        gini_index = (area_perfect_equality - area_lorenz_curve) / area_perfect_equality  # Índice de Gini
        return gini_index


    def _compute_shanon(self):
        data=self._join_impacts()
        techs=data.iloc[:,5:]
        proportions=techs.div(techs.sum(axis=1),axis=0)
        shannon_indices = -(proportions * np.log2(proportions)).sum(axis=1)
        data['shanon']=shannon_indices
        return data




    def violin_and_shanon(self):
        data= self._compute_shanon()
        pass
        shanon = data.iloc[:, -1]
        impacts = data.iloc[:, 0:5]

        # Calcular la correlación entre cada una de las cinco primeras columnas y la última columna

        correlations = impacts.corrwith(shanon)
        print(correlations)
        pass







if __name__ =='__main__':

    plot=PlottingEnbios(r'C:\Users\Administrator\PycharmProjects\SEEDS\runs\run_density_water\data\results_260.json',
                        flow_out_sum=r'C:\Users\Administrator\Downloads\flow_out_sum (1).csv',
                        starter=r'C:\Users\Administrator\PycharmProjects\SEEDS\runs\run_density_water\data\units_260.csv',
                        units=r'C:\Users\Administrator\PycharmProjects\SEEDS\runs\run_density_water\data\starter_260.csv')

    #plot.violin_plot(spore=0, path_save='violin_260')
    #plot.correlation_matrix("plots/correlation_260.png")
    #plot.spearman_correlation("plots/spearman_260.png")
    #plot.correlations_calliope(r"plots/correlation_matrix_calliope_260.png")
    #plot.intensive_plot(r'C:\Users\Administrator\PycharmProjects\SEEDS\runs\run_density_water\data\results_260.csv', level=4, path_save='plots/intensive.png')
   # plot.intensive_plot_absolute(r'C:\Users\Administrator\PycharmProjects\SEEDS\runs\run_density_water\data\results_260.csv',
      #                  level=4, path_save='plots/intensive_absolute_260.png')
    plot.violin_and_shanon()
    ###### 270

    plot2 = PlottingEnbios(r'C:\Users\Administrator\PycharmProjects\SEEDS\runs\run_density_water\data\results_270.json',
                          flow_out_sum=r'C:\Users\Administrator\Downloads\flow_out_sum.csv',
                          starter=r'C:\Users\Administrator\PycharmProjects\SEEDS\runs\run_density_water\data\units_270.csv',
                          units=r'C:\Users\Administrator\PycharmProjects\SEEDS\runs\run_density_water\data\starter_270.csv')

    #plot2.violin_plot(spore=0, path_save='violin_270.png')
    #plot2.correlation_matrix("plots/correlation_270.png")
    #plot2.spearman_correlation("plots/spearman_270.png")
    #plot2.correlations_calliope(r"plots/correlation_matrix_calliope_270.png")
    #plot2.intensive_plot(r'C:\Users\Administrator\PycharmProjects\SEEDS\runs\run_density_water\data\results_270.csv',
     #                   level=4, path_save='plots/intensive_270.png')
    #plot2.intensive_plot_absolute(
     #   r'C:\Users\Administrator\PycharmProjects\SEEDS\runs\run_density_water\data\results_270.csv',
      #  level=4, path_save='plots/intensive_absolute_270.png')
    plot2.violin_and_shanon()






