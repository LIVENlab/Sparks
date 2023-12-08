import bw2data as bd
import pandas as pd
from ProspectBackground.const.const import bw_project,bw_db
import json
from collections import defaultdict
import os
bd.projects.set_current(bw_project)            # Select your project
ei = bd.Database(bw_db)        # Select your db
from typing import List,Dict,Union,Optional




class GlobalMarkets:
    """
    This class:
        -Extracts info about the global electricity markets
        -Look for the ecoinvent acivities (if avaliable)
        -Creates a friendly data format to create the new markets
    """
    def __init__(self,path):

        self.__paths=self.get_paths(path)
        self.general={}
        self.region_mapping={}
        self.__not_found=[]

    @staticmethod
    def get_paths(path)->list:
        """
        This function iters over all the excel files in a folder and performs some transformations
        """
        files=[]
        pass
        for file in os.listdir(path):
            path_file=os.path.join(path,file)
            if path_file.endswith('.xlsx'):
                files.append(path_file)
        return files



    def get_data(self):
        """
        Open the excels in the folders and store some data
        Get general for general information
        Get region_mapping for continent-locations mapping
        """
        general=defaultdict(dict)  # Create a dictionary
        region_mapping={}
        for path in self.__paths:
            df=pd.read_excel(path)
            location=df['Loc'].unique().tolist()

            # Rename columns to avoid int problems
            cols=['Electricity market composition','stuff','share_15','share_20','share_30','share_40','share_50','Loc']
            df.columns=cols

            pass
            # get country name and store the mapping
            countries=df['stuff'].unique().tolist()
            countries=countries[0].split(',')
            countries=list(map(lambda x: x.strip(),countries))
            region_mapping[location[0]]=countries

            pass
            for index,row in df.iterrows():


                if row['share_50']==0:  #skip empty values
                    print(f"Debug: skipped {row['share_50']} as 0")
                    continue
                try:
                    total = row['Electricity market composition'].split('|')
                    name = total[0].strip()  # Elimina espacios al inicio y al final
                        # replace one inventory
                    pass
                    if str(name)==str("electricity production CSP mix, electricity high voltage, cut-off, U - GLO"):

                        name=str("electricity production, solar thermal parabolic trough, 50 MW")
                        carrier='electricity'
                        extra='RoW'
                        pass
                    else:
                        carrier = total[1].strip()
                        extra = total[2].split(',')[1].split('-')[1].strip()


                    general[location[0]][row['Electricity market composition']] = {
                        "name":name,
                        "share": row['share_50'],
                        "carrier": carrier,
                        "loc": extra
                    }

                except:
                    name = total[0]
                    if str(name)==str("electricity production CSP mix, electricity high voltage, cut-off, U - GLO"):
                        name=str("electricity production, solar thermal parabolic trough, 50 MW")
                    print(f'error in {total}, check it. Doc {path}')
                    general[location[0]][name] = {
                        "name":name.strip(),
                        "share": row['share_50'],
                        "carrier": 'electricity',
                        "loc": 'GLO'
                    }



        self.general=dict(general)
        self.region_mapping=dict(region_mapping)

        self.save_json_general()
        return dict(general)

    def save_json_general(self):
        """
        save general and region_maping as json
        """
        # combine the dictionaries
        general_merged={'general':self.general, "mapping": self.region_mapping}

        with open('general.json','w') as file:
            json.dump(general_merged,file,indent=4)


    @staticmethod
    def save_json_standard(name,dict):
        """
        general json save function
        """
        with open(name,'w') as file:
            json.dump(dict,file,indent=4)


    def check_ecoinvent_keys(self):

        gen=self.general
        ecoinvent_data={}
        possible_mismatch={}
        not_found={}
        pass
        for k in gen.keys():

            mismatch_region={}

            not_found_region={}
            pass
            for key,value in gen[k].items():
                print(key,value)
                pass

                activities=[{'name':a['name'],'unit':a['unit'],'location':a['location'],'code':a['code']}
                            for a in ei if
                            str(value['name']) in str(a['name']) and value['loc'] == str(a['location']) and str(a['unit'])=='kilowatt hour']

                if len(activities)>1:
                    activities = activities[0]

                if len(activities)<1:
                    pass
                    activities = [{'name': a['name'], 'unit': a['unit'], 'location': a['location'], 'code': a['code']}
                                  for a in ei if
                                  str(value['name']) in str(a['name']) and value['loc'] in str(a['location']) and str(
                                      a['unit']) == 'kilowatt hour']
                    if len(activities)>1:
                        # if there are multiple options, just get the first one
                        activities=activities[0]
                    elif len(activities)<1:
                        pass

                        # If still no activity, try RoW assignation

                        activities = [
                            {'name': a['name'], 'unit': a['unit'], 'location': a['location'], 'code': a['code']} for a
                            in ei if str(value['name']) in str(a['name']) and 'RoW' == str(a['location']) and str(
                                a['unit']) == 'kilowatt hour']
                        if len(activities)<1:
                            pass
                            not_found_region[key]={
                                'name':value['name'],
                                'location':value['loc'],
                                "full":key
                            }
                        else:
                            possible_mismatch[key]=[{"name":a['name'],"region":a['location']} for a in activities]



                gen[k][key]['activities']=activities
                #gen[k][key].update(activities)
                # TODO: Follow from here
                """
                ecoinvent_names[key]={
                    "acts":activities,
                    "carrier":value['carrier'],
                "location_theory":value['loc']}
                """
            #ecoinvent_data[k]=ecoinvent_names
            mismatch_region[k] = possible_mismatch
            not_found[k]=not_found_region

        self.__not_found=not_found
        self.ecoinvent_mapping=ecoinvent_data
        self.save_json_standard('ecoinvent_map.json',gen)
        self.save_json_standard('not_found.json',not_found)
        self.save_json_standard('mismatch.json',mismatch_region)
        pass



    def check_markets_electricity(self):
        data=self.region_mapping


        pass

    def check_region_markets(self):
        """
        Check if the included regions have an existing market in v3.9.1
        """
        data=self.region_mapping
        regions={}
        existing = []
        unmatched = []
        not_in_ecoinvent = []

        for key,value in data.items():
            for region in value:
                region=region.strip()
                print(region)
                try:
                    pass
                    act=ei.get_node(name='market for electricity, high voltage', location=region)
                    regions[region]={'name':act['name'],
                                     'region': act['location']
                                     }
                    existing.append(act['location'])
                except:
                    region=region.split('-')[0]
                    act=[a for a in ei if 'market for electricity, high voltage' in str(a['name']) and region in str(a['location'])]
                    if len(act)<1:
                        regions[region] = 'NaN'
                        not_in_ecoinvent.append(key)
                    else:
                        regions[region]=[{'name':a['name'],
                                     'region': a['location']
                                     } for a in act]
                        for a in act:
                            existing.append(a['location'])


                    pass

        markets = [a['location'] for a in ei if 'market for electricity, high voltage' == a['name'] and 'industry' not in str(
            a['name']) and 'renewable' not in a['name']]



        for market in markets:
            matched = False
            for existing_market in existing:
                if market in existing_market or existing_market in market:
                    matched = True
                    break
            if not matched:
                pass
                unmatched.append(market)

        print(unmatched)

        pass

        self.existing=list(set(existing))
        self.non_existing=list(set(unmatched))
        self.save_json_standard("non_market.json",regions)

        self.non_market=regions
        return regions

    def export_usa_data(self):
        """
        modify the ecoinvent data to facilitate programming tasks
        """
        with open('ecoinvent_map.json') as file:

            data=json.load(file)
            new={}
            for k,v in data.items():
                internal={}
                d=data[k]
                for val in d.values():
                    internal[val['name']]={
                        "share":val['share'],

                    }
                    pass

        pass










    # df=pd.read_excel(path)


data=GlobalMarkets(r'/home/lex/Documents/ICTA')
#data.get_data2()
data.get_data()

data.check_ecoinvent_keys()
#regs=data.check_region_markets()
data.export_usa_data()
pass

