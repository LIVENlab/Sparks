""""
@LexPascal
"""
import warnings

import bw2data as bd
import bw2data.backends
import pandas as pd
from ProspectBackground.const.const import bw_project,bw_db
import json
from collections import defaultdict
import os
bd.projects.set_current(bw_project)            # Select your project
ei = bd.Database(bw_db)        # Select your db
from typing import List,Dict,Union,Optional

"""
try:
    ei.copy('copy_test') #fes una copia de la db sencera
    ei_copy=bd.Database('copy_test')
except AssertionError:
    ei_copy=bd.Database('copy_test')
"""

class MarketBuilder():
    """

    This class contains a set of functions aiming to modify the ecoinvent database
    by updating the markets for electricity for the year 2050.

    The data is extracted from  Junne et al 2020, Environmental Sustainability Assessment of Multi-Sectoral Energy Transformation Pathways: Methodological Approach and Case Study for Germany
    https://doi.org/10.3390/su12198225.
    --2 DEGREES SCENARIO CONSIDERED--
    The data used in this article is also extracted from:
    Teske, S. Achieving the Paris Climate Agreement Goals; Springer Science and Business Media:
    Luxemburg, 2019. [Google Scholar]

    For the treatment of the data, check other classes and functions {}
    """
    def __init__(self):
        self.market_mapping=self.load_data("written.json") #JSON data with the grouped markets
        self.data=self.load_data("clean.json") #markets + ecoinvent info
        self.errors={} # summary of missing information and errors

        pass

    @staticmethod
    def check_loop(func):
        """
        Check that a modification happens
        """
        def wrapper(*args, **kwargs):
            print(f'Checking input from {func.__name__}, market {args[1]["name"]}')
            market = args[1]
            tech_start=[e for e in market.technosphere()]

            print('checking inputs...')
            func(*args, **kwargs)
            market1=args[1]
            tech_final=[e for e in market1.technosphere()]
            assert tech_start != tech_final
            return

        return wrapper


    def iter(self):
        """
        Iter through the data (regions --> markets) and:
            -Check that the activity exists
            -pop the inputs
            -create new data
        """

        for key,value in self.market_mapping.items():
            print(f"Changing markets in region {key}...")
            for v in value:
                try:
                    market = ei.get_node(name='market for electricity, high voltage', location=v)   # get the market activity to change
                except:
                    print(f" No activity for {v}")
                    continue
                self.pop_inputs(market)  # pop inputs
                self.input_changer(market, key)
    @staticmethod
    def pop_inputs(activity: bw2data.backends.proxies.Activity):
        """
        Delete all the inputs of a bw activity except transmission
        """
        pass
        to_delete = [e for e in activity.technosphere()
                     if 'transmission' not in str(e.input["name"])]
        for e in to_delete:
            e.delete()


    @check_loop
    def input_changer(self, market: bw2data.backends.proxies.Activity, region:str):

        """
        Extract the data from one specific region and use it as an input
        """

        data_region=self.data[region] # select the data from one region
        for key,value in data_region.items():
            try:
                new_input=ei.get_node(value['code'])    # get the activity from dict
                exchange=market.new_exchange(input=new_input,type='technosphere',amount=value["share"])
                exchange.save()
                if exchange in market.technosphere():
                    pass
                    print(f" new exchange included in market for electricity {market['location']} from group {region}: \n {exchange}")
            except bw2data.errors.UnknownObject:
                warnings.warn(f'Acivity {key}, with code {value["code"]} not in the db', Warning)
                self.__storerror__(key,value)
                continue




    def market_checker(self):
        """
        Check if the input classificatiopn (written.json) or self.market_mapping is ok
        """

        markets=[act['location'] for act in ei if act['name']=="market for electricity, high voltage"]
        pass
        all_regions=[]
        for value in self.market_mapping.values():
            all_regions=all_regions+value
        print(len(all_regions), len(set(markets)))
        print(set(markets).difference(set(all_regions)))
        duplicates = set(x for x in all_regions if all_regions.count(x) > 1)
        print("Duplicates:", list(duplicates))



    @staticmethod
    def load_data(path:str)->dict:
        """
        Load the different data used in previous sections
        """
        with open(path) as file:
            data=json.load(file)
        return data


    def __storerror__(self,name: str,code:str):
        """
        get error info and store it in self.errors
        """
        self.errors[name]=code
        pass

    def simple_test(self):
        """
        logic behind the substitution of activities
        """
        act1=ei.get_node('31850df83d34f335f1b2f58dc42d7fcc')
        add=ei.get_node('337e70e3ec52515067d02f0e169da57e') # market for steam

        to_delete=[e for e in act1.exchanges() if 'distribution' not in e['name']]
        for e in to_delete:
            e.delete()
        ex=act1.new_exchange(input=add, amount=10,type='technosphere')
        ex.save()

        for a in act1.technosphere():
            print(a)
        pass



        pass


        pass
markets=MarketBuilder()
markets.iter()
#markets.market_checker()
#markets.simple_test()
pass