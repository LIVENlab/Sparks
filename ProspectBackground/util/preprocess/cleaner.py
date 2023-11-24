"""
@author: Alexander de Tomás (ICTA-UAB)
        -LexPascal
"""

import bw2data.errors
import pandas as pd
pd.options.mode.chained_assignment = None
import bw2data as bd
import warnings
from ProspectBackground.const.const import bw_project,bw_db
from typing import Dict,Union,Optional
bd.projects.set_current(bw_project)            # Select your project
database = bd.Database(bw_db)        # Select your db
import time
import os


class Cleaner():
    """
    The main objectives of this class are:
        *Clean the input data
        *Modify the units of the input data
    """
    __aliases=[]

    def __init__(self, caliope, motherfile, subregions : [Optional,bool]= False):
        """
        @param caliope: Calliope data, str path
        @param motherfile: Basefile, str path
        @param subregions: BOOL:
            -If true, the cleaning and preprocess of the data, will consider the different subregions
            *ESP_1,ESP_2..
            Default value set as false:
            - It will group the different subregions per country
        """
        self.subregions=subregions
        self.data=caliope # flow_out_sum
        self.mother_file=motherfile # basefile
        self.clean=None # Final output of the Cleaning functions
        self.mother_ghost=None
        # Unit changer
        self.activity_conversion_dictionary : Dict[str,Dict[str,Union[str,int]]]=dict() #Dictionary containing the activity-conversion factor information
        self.clean_modified=None # Final output of the Unit functions
        self.techs_region_not_included=[] #List of the Processors and regions (together) not included in the basefile
        self.locations=[] # List of the regions included in the study



    @staticmethod
    def create_df() -> pd.DataFrame:
        """
        Basic template for clean data
        """
        columns = [
            "spores",
            "techs",
            "locs",
            "carriers",
            "unit",
            "flow_out_sum"
        ]
        df = pd.DataFrame(columns=columns)
        return df


    @staticmethod
    def input_checker(data):
        """
        Check whether the input from Calliope follows the expected structure
        data: pd.Dataframe
        """
        expected_cols = set(['spores', 'techs', 'locs', 'carriers', 'unit', 'flow_out_sum'])
        cols = set(data.columns)

        # Search for possible differences. Older versions of calliope produce "spore"
        if "spore" in cols:
            data.rename(columns={'spore': 'spores'}, inplace=True)
            cols = set(data.columns)

        if expected_cols == cols:
            print('Input checked. Columns look ok')
        else:
            raise KeyError(f"Columns {cols} do not match the expected columns: {expected_cols}")

    def modify_mother_file(self):
        """
        Crete a new column "aliases" and store the file as a ghost file
        """
        ex_data = pd.read_excel(self.mother_file, sheet_name=None)
        # Create the alias
        sheet_name = 'Processors'
        mother_file = ex_data[sheet_name].copy()

        # 1. Generate the same aliases on the mother file
        mother_file['aliases'] = mother_file['Processor'] + '__' + mother_file['@SimulationCarrier'] + '___' +mother_file['Region']
        ex_data[sheet_name] = mother_file.copy()

        current = os.path.dirname(os.path.abspath(__file__))
        folder_path = os.path.join(os.path.dirname(current), 'Default')
        print(current)
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, 'base_ghost.xlsx')
        with pd.ExcelWriter(file_path) as writer:
            for sheet, df in ex_data.items():
                cols=df.columns
                df.to_excel(writer, sheet, index=False, columns=cols)

        self.mother_ghost = file_path

    def get_mother_data(self):
        """
        Get and store some data from the mother file: - regions - aliases
        """
        ex_data = pd.read_excel(self.mother_ghost, sheet_name='Processors')
        pass
        regions = ex_data['Region'].unique().tolist()
        aliases = ex_data['aliases'].unique().tolist()
        self.regions = regions
        self.__aliases = aliases

    pass

    def apply_filters(self,df):
        """
        Filter the calliope data based on the filters from get_mother_data
        -regions
        -aliases
        """
        caliope=df
        calliope = caliope[caliope['locs'].isin(self.regions)]
        calliope = calliope[calliope['aliases'].isin(self.__aliases)]
        return calliope


    def changer(self):
        """
        *Assume that the csv is comma separated*
        Group subregions in regions and sum the value for each technology /carrier
        :param df:
        :return:

        """

        print('Adapting input data...')
        try:
            df = pd.read_csv(self.data, delimiter=',')
            self.data=df
        except FileNotFoundError:
            raise FileNotFoundError(f'File {self.data} does not exist. Please check it')

        else:

            self.input_checker(df)
            df = df.dropna()
            # Create an empty df with the expected data
            gen_df = self.create_df()
            scenarios = list(df.spores.unique())
            for scenario in scenarios:
                df_sub = df.loc[df['spores'] == scenario]   # Create a new df with the data from 1 scenario
                df_sub['locs'] = df['locs'].apply(self.manage_regions)
                df_sub = df_sub.groupby(['techs', 'locs']).agg({
                    "spores": "first",
                    "carriers": "first",
                    "unit": "first",
                    "flow_out_sum": "sum"
                }).reset_index()
                gen_df = pd.concat([gen_df, df_sub])
        gen_df['aliases']=gen_df['techs']+'__'+gen_df['carriers']+'___'+gen_df['locs']
        pass
        gen_df=self.apply_filters(gen_df)
        self.clean=gen_df
        return gen_df




    def manage_regions(self,arg):


        # Special issue for the Portugal Analysis
        if arg == 'ESP-sink':
            region = 'ESP'

        if self.subregions is True:
            # If the used is not interested in having subregions, the location will be the same
            region=str(arg)
        else:
            # If the user is interested in having subregions, get the first part
            # CZE_1 = CZE
            region = arg.split('_')[0]

        return region




    def get_regions(self,df)->list:
        """
        This function returns a list of the existing regions in the study
        """
        regions = df['locs'].unique().tolist()
        if self.subregions is not True:
            return regions

        else:
            regions=list(set([e.split('_')[0] for e in regions]))
            return regions

    def preprocess_data(self):
        """
        Run different functions of the class under one call
        @return: final_df: calliope data cleaned. Check definitions of the class for more information
        """
        self.modify_mother_file()
        self.get_mother_data()

        dat = self.changer()
        pass
        self.clean = dat
        return dat




    ##############################################################################

    # This second part focuses on the modification of the units
    def data_merge(self):
        """
        This function reads the Excel "mother file" and generates a dictionary following the structure:

        {technology name :
            { bw_code : [str],
            conversion factor: [int]}}
        """
        df = pd.read_excel(self.mother_ghost)
        general_dict = {}
        for index, row in df.iterrows():

            alias=row['aliases']
            code = row['BW_DB_FILENAME']
            factor = row['@SimulationToEcoinventFactor']
            general_dict[alias] = {
                'factor': factor,
                'code': code
            }
        self.activity_conversion_dictionary=general_dict
        return general_dict




    def modify_data(self):
        """
        This function reads the dictionary generated by data_merge, and the flow_out_sum file.
        Applies some transformations:
            *Multiply the flow_ou_sum by the characterization factor
            *Change the unit according to the conversion
        :return: Two files.
            *calliope_function.csv : Intermediate file to check units and techs
            * flow_out_sum_modified.csv : Final csv ready for enbios

            Returns a list of techs to apply the following function "check elements"
        """
        df=self.clean

        # Create a modified column name to match  the names
        print('Preparing to change and adapt the units...')


        df=self.ecoinvent_units_factors(df)

        # Prepare an enbios-like file    cols = ['spores', 'locs', 'techs', 'carriers', 'units', 'new_vals']
        cols = ['spores', 'locs', 'techs', 'carriers', 'units', 'new_vals','aliases']
        df = df[cols]
        df.rename(columns={'spores': 'scenarios', 'new_vals': 'flow_out_sum'}, inplace=True)
        df.dropna(axis=0, inplace=True)
        print('Units adapted and ready to go')
        self.clean_modified=df

        return df







    def ecoinvent_units_factors(self,df):
        """
        Read the calliope data and extend the information based on self.actvity_conversion dictionary
        *add new columns and apply the conversion factor to the value

        *delete the activities with non existing codes in the db
        """
        # Create new columns
        #df=df.copy() # avoid modifications during the loop
        df['new_vals'] = None
        df['Units_new'] = None
        df['codes'] = None
        df['names_db'] = None

        delete=[]
        # df['flow_out_sum']=[x.replace(',','.') for x in df['flow_out_sum']]
        for key in self.activity_conversion_dictionary.keys():
            code = self.activity_conversion_dictionary[key]['code']
            try:
                activity = database.get_node(code)
                unit = activity['unit']
                act_name = activity['name']
            except bw2data.errors.UnknownObject:
                message=f" \n{code} from activity, {key} not found in the database. Please check your database. This activitiy will be deleted"
                warnings.warn(message,Warning)
                delete.append(key)

                continue  # If activity doesn't exists, do nothing
            pass
            for index, row in df.iterrows():
                if str(key) == str(row['aliases']):
                    factor = (self.activity_conversion_dictionary[key]['factor'])
                    value = float(row['flow_out_sum'])
                    new_val = value * factor
                    df.at[index, 'codes'] = code
                    df.at[index, 'units'] = unit
                    df.at[index, 'new_vals'] = new_val
                    df.at[index, 'names_db'] = act_name
                else:
                    pass
        df=df.loc[~df['aliases'].isin(delete)]

        return df


    def clean_included_activities(self):
        """
        This code filters the technologies only if they are avaliable on the db
        Delete the activities from the ghost mother file
        """
        ex_data = pd.read_excel(self.mother_file, sheet_name=None)
        # Create the alias
        sheet_name = 'Processors'
        df_mother = ex_data[sheet_name].copy()
        #df_mother = pd.read_excel(self.mother_ghost, sheet_name='Processors')
        df_cal = self.clean_modified

        delete = []
        for index, row in df_mother.iterrows():
            processor = row['Processor']
            code = row['BW_DB_FILENAME']

            try:
                database.get_node(code)
            except:
                delete.append(processor)

                pass

        # Save the modified mother file
        df_mother = df_mother.loc[~df_mother['Processor'].isin(delete)]



        # Save the calliope data
        self.clean_modified = df_cal
        caliope_techs=df_cal['aliases'].unique().tolist()
        df_mother=df_mother.loc[df_mother['Processor'].isin(caliope_techs)]


        ex_data[sheet_name] = df_mother
        with pd.ExcelWriter(self.mother_ghost) as writer:
            for sheet, df in ex_data.items():
                cols = df.columns
                df.to_excel(writer, sheet, index=False, columns=cols)


        return df_cal
        pass

    def clean_base_ghost(self):
        """
        This function filters the other way arround;
        If the base file has activities which do not exist on the calliope data, delete them
        """

    def adapt_units(self)-> pd.DataFrame:
        "combines some functions under one call"
        self.data_merge()
        pass
        modified_units_df=self.modify_data()
        return modified_units_df


































