import bw2data as bd
import pandas as pd
from ProspectBackground.const.const import bw_project,bw_db
import json
from collections import defaultdict
import os
bd.projects.set_current(bw_project)            # Select your project
ei = bd.Database(bw_db)        # Select your db
from typing import List,Dict,Union,Optional


class Matching:
    def __init__(self):
        self.exporter=None
        self.__TO_BE_INCLUDED=[]
        pass
    def export_usa_data(self):
        """
        modify the ecoinvent data to facilitate programming tasks
        {region : {act_name { code, share}}}

        If there's no activity: TO BE INCLUDED
        """
        with open('ecoinvent_map.json') as file:

            data = json.load(file)
            new = {}
            for k, v in data.items():
                internal = {}
                d = data[k]
                pass
                for key,val in d.items():
                    try:
                        activties=d[key]['activities'][0]
                        name=val['name']+'_'+activties['location']
                        internal[name] = {
                            "share": val['share'],
                            "code": activties['code']
                        }
                    except:
                        self.__TO_BE_INCLUDED.append(val['name'])
                        internal[val['name']] = {
                            "share": val['share'],
                            "code": "TO_BE_INCLUDED"
                        }
                new[k]=internal
            self.exporter=new
            with open('clean.json','w') as file:
                json.dump(new,file, indent=4)







match=Matching()
match.export_usa_data()