import os

import pandas as pd
import numpy as np
import bw2data as bd
import bw2io as bi
from typing import Union
from bw2data.backends.proxies import Activity
"""

Base class to produce info for splitting LCA inventories

"""
bd.projects.set_current('TFM_Lex')
ei = bd.Database('ecoinvent')


class SPLIT:
    """
    add info
    """
    def __init__(self,
                 basefile: Union[str,os.PathLike]):

        self.basefile=self._load_basefile(basefile)
        self._iter_processors()

        pass



    @staticmethod
    def _load_bw(bw_project,bw_databse):


        bd.projects.set_current(bw_project)
        ei=bd.Database(bw_databse)
        return ei



    @staticmethod
    def _load_basefile(path: Union[str,os.PathLike]) -> pd.DataFrame:
        """
        load the Processors sheet from baseifle
        """
        return pd.read_excel(path, sheet_name='Processors')


    def _iter_processors(self):
        for index,row in self.basefile.iterrows():
            try:
                act=ei.get_node(row['BW_DB_FILENAME'])
                self.__extract_exchanges(act)
            except:
                pass

    @staticmethod
    def __extract_exchanges(act: Activity) ->dict:
        """
        Read a bw activitity and filter
        """
        for ex in act.technosphere():
            print(ex.input)
        exchanges={}
        pass


if __name__=='__main__':

    split=SPLIT('/home/lex/Downloads/basefile_montecarlo.xlsx')
