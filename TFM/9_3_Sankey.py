from collections import OrderedDict
import pandas as pd
import json
"""
This code produces the input file for: 
https://observablehq.com/d/ba60235b4494082f
"""

class Tree:
    def __init__(self,
                 motherfile:str, path:str):

        self.acts="to do"
        self.motherfile=motherfile
        self.results=self.read_results(path)




    def read_results(self,
                     pathlike:str):
        df=pd.read_csv(pathlike)
        pass


    def get_data(self,
                 indicator: str, scenario:int):

        element=self.results[scenario] # select the scenario

        for element in self.results:
            element=dict(element)
            self.inspect_dic(element)







    def hierarchy(self, *args) -> dict:
        """
        This function creates the hierarchy tree.
        It uses two complementary functions (generate_dict and tree_last_level).

        It reads the information contained in the mother file starting by the bottom (n-lowest) level
        :param data:
        :param args:
        :return:
        """

        pass
        df = pd.read_excel(self.motherfile, sheet_name='Dendrogram_top')
        df2 = pd.read_excel(self.motherfile, sheet_name='Processors')

        # Do some changes to match the regions and aliases

        df2['Processor'] = df2['Processor'] + '__' + df2['@SimulationCarrier']  # Mark, '__' for carrier split
        # Start by the last level of parents
        levels = df['Level'].unique().tolist()
        last_level_parent = int(levels[-1].split('-')[-1])
        last_level_processors = 'n-' + str(last_level_parent + 1)
        df2['Level'] = last_level_processors
        df = pd.concat([df, df2[['Processor', 'ParentProcessor', 'Level']]], ignore_index=True, axis=0)

        levels = df['Level'].unique().tolist()

        list_total = []
        for level in reversed(levels):
            df_level = df[df['Level'] == level]
            if level == levels[0]:
                break

            elif level == levels[-1]:
                last = self.tree_last_level(df_level, *args)
                global last_list
                last_list = last

            else:
                df_level = df[df['Level'] == level]
                list_2 = self.generate_dict(df_level, last_list)
                last_list = list_2
                list_total.append(list_2)

        dict_tree = list_total[-1]
        self.hierarchy_tree = dict_tree[-1]


class Leaf:

    def __init__(self,
                 name: str,
                 level,
                 parent=None):
        self.name = name
        self.level = level
        self.parent = parent


b=Tree(r'/home/lex/Downloads/basefile_off.xlsx', path='/home/lex/Downloads/results.csv')
c=b.load_results()
b.hierarchy('b')
b.get_data('a',0)



