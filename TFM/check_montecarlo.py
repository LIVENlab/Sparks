
from bw2data.backends.proxies import Activity
from bw2analyzer import ContributionAnalysis
import bw2data as bd
from bw2calc import LCA
import pandas as pd
from matplotlib import pyplot as plt

bd.projects.set_current("TFM_Lex")
db = bd.Database("ecoinvent")

method = ('ReCiPe 2016 v1.03, midpoint (H)', 'water use', 'water consumption potential (WCP)')

"""
check MonteCarlo
"""


class CheckMonteCarlo:
    """
    Run 40 times each activity and plot it
    """

    def __init__(self):
        self.data= self.get_data()
        self.results=[]
        self.iter_data()


    def get_data(self):
        return pd.read_excel(r'/home/lex/Downloads/basefile_montecarlo.xlsx', sheet_name='Processors')


    def iter_data(self):
        data=self.data
        for index,row in data.iterrows():
            try:
                activity=db.get_node(row['BW_DB_FILENAME'])# get activity code
                self.run_simulation(activity)
                pass
            except:
                print(f" error in getting {row['BW_DB_FILENAME']}, from {row['Processor']}")
                continue

            print('stuff done')
            pass

    def run_simulation(self, act: Activity):
        name=act['name']
        results = []
        # stochastic result
        for i in range(2):
            lca = LCA({act: 1}, method, use_distributions=True)
            print('doing something here')
            lca.lci()
            lca.lcia()
            scores = lca.score
            results.append(scores)

        lca=LCA({act:1},method, use_distributions=False)
        lca.lci()
        lca.lcia()
        score = lca.score
        pass

        result={name : {'stochastic':results, 'static': score}}
        CheckMonteCarlo.plot_results(result)
        self.results.append(result)
        return result


    @staticmethod

    def plot_results(result: dict):

        title=result.keys()[0]
        pass
        stochastic_results=result[title]['stochastic']
        static_results=result[title]['static']
        # Crear el histograma
        plt.hist(stochastic_results, bins=20, alpha=0.5, label='Resultados Estocásticos')

        # Agregar la línea vertical con el resultado estático
        plt.axvline(x=static_results, color='r', linestyle='--', linewidth=2, label='Resultado Estático')

        # Agregar título y leyenda
        plt.title(title)
        plt.legend()

        # Guardar el plot con el nombre del título
        path='plots_montecarlo' + title + '.png'
        plt.savefig(path)

        # Mostrar el plot
        plt.show()


a=CheckMonteCarlo()




