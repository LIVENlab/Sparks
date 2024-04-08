
import pandas as pd

import numpy as np
import json
import matplotlib.pyplot as plt
from typing_extensions import OrderedDict
import seaborn as sns
class CompRounds:

    def __init__(self):



        self._260=self.load_data(r'C:\Users\Administrator\PycharmProjects\SEEDS\results\results_260.json')
        self._270=self.load_data(r'C:\Users\Administrator\PycharmProjects\SEEDS\results\results.json')

        self.get_normalized_values()

        self.get_normalized_second()


    def load_data(self, path:str)->pd.DataFrame:
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


        res=self._260
        info_norm = {}
        normalized_df = pd.DataFrame(index=res.index)  # Nuevo DataFrame para almacenar las columnas normalizadas
        for col in res.columns:
            name = str(col)

            min_spore_0 = res[col][0]
            min_ = np.min(res[col])
            max_ = np.max(res[col])

            normalized_df[name] = (res[col] - min_) / (max_ - min_)
            # normalized_df[name_modified]= res[col] / min_
            #normalized_df[name_modified] = res[col] / min_spore_0
            info_norm[col] = {
                "max": max_,
                "min": min_,
                "scenario_max": normalized_df[name].idxmax().item(),
                "scenario_min": normalized_df[name].idxmin().item()
            }
        normalized_df.columns = [col.rstrip('_') for col in normalized_df.columns]
        normalized_df['clue']='Round 1'
        self._260_normalized = normalized_df
        return normalized_df

    def get_normalized_second(self):


        min_values = self._260.min()
        max_values = self._260.max()
        normalized_second_df = (self._270 - min_values) / (max_values - min_values)
        normalized_second_df['clue']='Round 2'
        self._270_normalized= normalized_second_df
        return normalized_second_df


    def compare_boxplots(self):

        data=pd.concat([self._260_normalized, self._270_normalized], axis=0).reset_index(drop=True)
        pass
        plt.figure(figsize=(10, 6))

        sns.boxplot(data, orient='h', palette=['skyblue', 'lightgreen'])
        plt.show()


res=CompRounds()
res.compare_boxplots()















