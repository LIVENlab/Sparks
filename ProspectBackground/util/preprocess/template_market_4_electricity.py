from warnings import warn
from logging import getLogger
import bw2data as bd
import pandas as pd
from bw2data.errors import UnknownObject
from ProspectBackground.const.const import bw_project,bw_db
from ProspectBackground.const.caliope_ecoinvent_regions import ecoinvent_caliope_regions
from .activity_creator import InventoryFromExcel
import warnings
from typing import Dict,Union,List
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
        self.electricity_list ={}
        self.units=units



    @staticmethod
    def create_template_df()-> pd.DataFrame:
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




    def get_elec_acts(self,region,present_dict=None)->Dict[str,List[str]]:

        """
        Update self.electricity_list with the activities that form the market for electricity
        """
        present_dict = self.enbios_data
        keys_to_check = [present_dict]
        while keys_to_check:
            current_dict = keys_to_check.pop()
            for key, value in current_dict.items():
                if key == self.dendrogram_key and isinstance(value, list):
                    value=[element for element in value if element.split('___')[-1] == region]
                    self.electricity_list[region]=value
                    return value
                if isinstance(value, dict):
                    keys_to_check.append(value)
        return None




    def template_market_4_electricity(self,region,*args: str) -> pd.DataFrame:
        """
        This function returns a template of the "default_market_for electricity" in order to create an inventory.
        It fills all the information required to create a new activity in ecoinvent + the activities from that market
        args:
        -Location
        -Activity_name
        -Activity_code
        -Units
        """
        Location=region
        Activity_name=args[0]
        Activity_code=args[1]
        Reference_product=args[2]
        Units=args[3]
        pass


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
        self.get_elec_acts(region)
        pass


        for activity in self.electricity_list[region]:
            pass
            for key in map_activities.keys():
                if activity == key:
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
    def get_location_key(key)->str:
        """
        pass the calliope location and return the econvent location
        """
        ecoinvent_location=ecoinvent_caliope_regions[key]
        return ecoinvent_location

    def build_templates(self)-> Dict[str,Union[pd.DataFrame,str]]:
        """
        Call functions defined in this file in order to create the template of the market for electricity for one regions
        This function adds the market in the template dictionary
        {region : template(pd.DataFrame)}
        """
        templates={}
        for region in self.regions:
            old_code = self.get_market_ecoinvent(region)
            if old_code is None:
                continue
            else:
                # Modify the activity code
                Location=region
                Activity_name='Future_market_for_electricity'+'_'+region
                Activity_code = 'FM4E' +'_'+ region
                Reference_product = f'Future electricity production in {region}, {self.units}'
                template=self.template_market_4_electricity(Location,Activity_code,Activity_name,Reference_product,self.units)
                templates[region]=[template,Activity_code,old_code]
        pass
        return templates

