from warnings import warn
from logging import getLogger
import bw2data as bd
import pandas as pd
from bw2data.errors import UnknownObject
from ProspectBackground.const.const import bw_project,bw_db
from ProspectBackground.const.caliope_ecoinvent_regions import ecoinvent_caliope_regions
from .activity_creator import InventoryFromExcel
import warnings
from typing import Dict,Union
getLogger("peewee").setLevel("ERROR")
bd.projects.set_current(bw_project)            # Select your project
ei = bd.Database(bw_db)

class Market_for_electricity():

    """
    This class creates a template for the future market for electricity.

    """

    def __init__(self,enbios_data,dendrogram_key,regions,units):

        self.enbios_data= enbios_data
        self.regions=regions
        self.dendrogram_key=dendrogram_key
        self.electricity_list =[]
        self.units=units



    @staticmethod
    def create_template_df():
        """
        :return: a template df of the excel like inventory
        """
        column_names = ["Amount",
                        "Location",
                        "Activity name",
                        "Activity_code",
                        "Reference_product",
                        "Unit",
                        "Act_to",
                        "Technosphere",
                        "Database",
                        "Unit_check"]
        # Update --> add a unit check to verify the units of the inventory
        df = pd.DataFrame(columns=column_names)
        return df




    def get_elec_acts(self,region,present_dict=None):
        """
                :param
                    -args: We're looking for the activities under that define a "future market for electricity". Hence, we specify
                            the keys that point something like "Electricity generation" in the hierarchy
                :return:  list of the activities failing into the electricity_market
                """
        present_dict = self.enbios_data
        keys_to_check = [present_dict]
        while keys_to_check:
            current_dict = keys_to_check.pop()
            for key, value in current_dict.items():
                if key == self.dendrogram_key and isinstance(value, list):
                    value=[element for element in value if element.split('___')[-1] == region]
                    self.electricity_list=value
                    return value
                if isinstance(value, dict):
                    keys_to_check.append(value)
        return None




    def template_market_4_electricity(self,*args):
        """
        This function returns a template of the "default_market_for electricity" in order to create an inventory.

        Check the template_market.csv to do some changes

        :param market_el_list:
        :param Location:
        :param Activity_name:
        :param Activity_code:
        :param Reference_product:
        :param Unit:
        :param Database:
        :return:
        """
        Location=args[0]
        Activity_name=args[1]
        Activity_code=args[2]
        Reference_product=args[3]
        Units=args[4]



        # Call create_template to create the df of the inventory
        df = self.create_template_df()
        map_activities=self.enbios_data['activities']

        # Add first row to the df
        first_row = {"Amount": 1,
                     "Location": Location,
                     "Activity name": Activity_name,
                     "Activity_code": Activity_code,
                     "Reference_product": Reference_product,
                     "Unit": Units,
                     "Act_to": "marker",
                     "Technosphere": "Yes",
                     "Database": bw_db,
                     "Unit_check": None
                     }
        df.loc[len(df.index)] = first_row

        for element in self.electricity_list:
            for key in map_activities.keys():
                if element == key:
                    code = map_activities[key]['id']['code']  # This might change

                    try:
                        act = ei.get_node(code)
                    except UnknownObject:
                        warn(f'You are trying to include the activity with code {code} which is not registered in your database. This activity will be skipped')
                        continue

                    location = act['location']
                    ref_prod = act['reference product']
                    unit_check = act['unit']
                    row = {"Amount": 1,
                           "Location": str(location),
                           "Activity name": str(key),
                           "Activity_code": code,
                           "Reference_product": ref_prod,  # TODO: change to "activity_to" or similar
                           "Unit": Units,  # All should be converted previously
                           "Act_to": Activity_name,
                           "Technosphere": "Yes",  # All should be
                           "Database": act['database'],
                           "Unit_check": unit_check
                           }

                    df.loc[len(df.index)] = row
        #TODO: Check if needed
        # df_gruoped = df.groupby('Activity_code')['Amount'].transform('sum')
        # df['Amount'] = df_gruoped
        # df = df.drop_duplicates(subset='Activity_code')
        print(f'Template for future market for electicity in {Location} has been created')
        InventoryFromExcel(df)

        return df

    def get_market_ecoinvent(self,region):
        ecoinvent_region=self.get_location_key(region)
        code=None
        for act in ei:
            if 'market for electricity, high voltage' in act['name'] and 'aluminium industry' not in str(act['name']) and act['location'] == ecoinvent_region:
                code=act['code']
                if code is None:
                    message=f'''The defined calliope region {region} has no market for electricity in the database
                        \n This market for electricity won't be used'''
                    warnings.warn(message,Warning)
        return code

    @staticmethod
    def get_location_key(key):
        """
        pass the calliope location and return the econvent location
        """
        ecoinvent_location=ecoinvent_caliope_regions[key]
        return ecoinvent_location

    def build_templates(self)-> Dict[str,Union[pd.DataFrame,str]]:
        templates={}
        for region in self.regions:
            activities=self.get_elec_acts(region)
            old_code = self.get_market_ecoinvent(region)
            if old_code is None:
                continue
            else:
                # Modify the activity code
                Location=region
                Activity_name='Future_market_for_electricity'+'_'+region
                Activity_code = 'FM4E' +'_'+ region
                Reference_product = f'Future electricity production in {region}, {self.units}'
                template=self.template_market_4_electricity(Location,Activity_name,Activity_code,Reference_product,self.units)

                templates[region]=[template,Activity_code,old_code]
        return templates

