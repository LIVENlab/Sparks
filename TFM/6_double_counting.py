import json
from typing import Union,Optional
import pandas as pd
import bw2data as bd

from ProspectBackground.const.const import bw_project,bw_db
from typing import Dict,Union,Optional


bd.projects.set_current(bw_project)            # Select your project

database = bd.Database(bw_db)        # Select your db

""""
Delete the upstream of the foreground
"""

class DoubleAccounting():

    def __init__(self, enbios_data):
        self.enbios_data=self.load_data(enbios_data)
        pass

    @staticmethod
    def load_data(path: str) -> dict:
        """
        Load the different data used in previous sections
        """
        with open(path) as file:
            data = json.load(file)
        return data

    def iter_data(self):
        """
        access enbios' dictionary activities
        """
        activities=self.enbios_data['activities']
        for key,val in activities.items():
            code=val['id']['code']
            act=database.get_node(code)

            print('upstream_activity', key)

            for t in act.upstream():
                print(t)
            pass




if __name__=='__main__':
    double=DoubleAccounting(r'C:\Users\Administrator\PycharmProjects\TFM__Lex\ProspectBackground\Default\data_enbios.json')
    double.iter_data()
