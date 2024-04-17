"""
@author: Alexander de Tomás (ICTA-UAB)
        -LexPascal
"""
import bw2data.errors
import pandas as pd
pd.options.mode.chained_assignment = None
import bw2data as bd
import warnings
from Sparks.const.const import bw_project,bw_db
from typing import Dict,Union,Optional
bd.projects.set_current(bw_project)            # Select your project
database = bd.Database(bw_db)        # Select your db
pass

class Cleaner():
    """

        *Clean the input data
        *Modify the units of the input data
    """
    def __init__(self,
                 caliope,
                 motherfile,
                 subregions : [Optional,bool]= False):

        """
        @param caliope: Calliope data, str path
        @param motherfile: Basefile, str path
        @param subregions: BOOL:
            -If true, the cleaning and preprocess of the data, will consider the different subregions
            *ESP_1,ESP_2..
            Default value set as false:
            - It will group the different subregions per country
        """
        self.subregions = subregions
        self._raw_data = caliope
        self.mother_file = motherfile
        self.final_df = None
        self.activity_conversion_dictionary = {}
        self.techs_region_not_included = []

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
        expected_cols = set(['spores',
                             'techs',
                             'locs',
                             'carriers',
                             'unit',
                             'flow_out_sum'])
        cols = set(data.columns)
        # Search for possible differences. Older versions of calliope produce "spore"
        if "spore" in cols:
            data.rename(columns={'spore': 'spores'}, inplace=True)
            cols = set(data.columns)
        if expected_cols != cols:
            raise KeyError(f"Columns {cols} do not match the expected columns: {expected_cols}")


    def _load_data(self) -> pd.DataFrame:
        """ Assuming comma separated input"""
        return pd.read_csv(self._raw_data, delimiter=',').dropna()


    def _group_df(self, df: pd.DataFrame)-> pd.DataFrame:
        """
        group input data by specified criteria
        """
        gen_df = self.create_df()
        scenarios = df['spores'].unique()

        for scenario in scenarios:
            df_sub = df[df['spores'] == scenario]
            if not self.subregions:
                df_sub['locs'] = df['locs'].apply(self._manage_regions)
            df_sub = df_sub.groupby(['techs', 'locs', 'carriers']).agg({
                "spores": "first",
                "unit": "first",
                "flow_out_sum": "sum"
            }).reset_index()
            gen_df = pd.concat([gen_df, df_sub])

        return gen_df


    def _adapt_data(self)->pd.DataFrame:

        print('Adapting input data...')
        try:
            df=self._load_data()
            self.input_checker(data=df)
        except FileNotFoundError:
            raise FileNotFoundError(f'File {self._raw_data} does not exist. Please check it')

        return self._group_df(df)


    def filter_techs(self,df: pd.DataFrame)-> pd.DataFrame:
        """
        Filter the input data based on technologies defined in the basefile
        """

        df_names=df.copy()
        self.basefile=pd.read_excel(self.mother_file, sheet_name='Processors')
        # Filter Processors
        df_names['alias_carrier']=df_names['techs'] + '_' +df_names['carriers']
        self.basefile['alias_carrier']=self.basefile['Processor']+ '_' + self.basefile['@SimulationCarrier']
        excluded_techs = set(df_names['techs']) - set(self.basefile['Processor'])
        self.techs_region_not_included=excluded_techs

        df_names = df_names[~df_names['techs'].isin(excluded_techs)] # exclude the technologies
        if len(excluded_techs)>0:
            message=f'''\nThe following technologies, are present in the energy data but not in the Basefile: 
            \n{excluded_techs}
            \n Please,check the following items in order to avoid missing information'''
            warnings.warn(message,Warning)

        return df_names

    def preprocess_data(self)->pd.DataFrame:
        """
        Run data preprocessing steps
        """

        self.final_df = self.filter_techs(self._adapt_data())
        return self.final_df

    def _manage_regions(self, *arg)->str:
        """
        Manage region names
        """
        pass
        if isinstance(arg, tuple):
            arg = arg[0]
            region = arg.split('-')[0]
            region = region.split('_')[0]
        # Special issue for the Portugal Analysis
        if arg == 'ESP-sink':
            region = 'ESP'
        else:
            region = arg.split('-')[0]
            region = region.split('_')[0]

        return region


    ##############################################################################
    # This second part focuses on the modification of the units

    def _data_merge(self):
        """
        This function reads the Excel "mother file" and generates a dictionary following the structure:

        {technology name :
            { bw_code : [str],
            conversion factor: [int]}}
        """
        df= self.basefile
        pass
        general_dict = {}
        for index, row in df.iterrows():
            name = row['Processor'] + '_' + row['@SimulationCarrier']
            code = row['Ecoinvent_key_code']
            factor = row['@SimulationToEcoinventFactor']
            general_dict[name] = {
                'factor': factor,
                'code': code
            }

        self.activity_conversion_dictionary=general_dict
        return general_dict



    def _modify_data(self)->pd.DataFrame:
        """
        This function reads the dictionary generated by data_merge, and the flow_out_sum file.
        Applies some transformations:
            *Multiply the flow_ou_sum by the characterization factor
            *Change the unit according to the conversion
        """
        df=self.final_df
        pass
        # Create a modified column name to match  the names
        print('Chaning and adapting units')
        df=self._ecoinvent_units_factors(df)
        # Prepare an enbios-like file    cols = ['spores', 'locs', 'techs', 'carriers', 'units', 'new_vals']
        df=self._clean_final_df(df)
        self.clean_modified = df
        print('Units adapted')
        return df


    def adapt_units(self):
        self._data_merge()
        modified_units_df=self._modify_data()
        return modified_units_df


    def _ecoinvent_units_factors(self,df):
        """
        Read the calliope data and extend the information based on self.actvity_conversion dictionary
        *add new columns and apply the conversion factor to the value
        """
        # Create new columns
        df['new_vals'] = None
        df['units_new'] = None
        df['codes'] = None
        df['names_db'] = None

        # df['flow_out_sum']=[x.replace(',','.') for x in df['flow_out_sum']]
        for key in self.activity_conversion_dictionary.keys():
            code = self.activity_conversion_dictionary[key]['code']
            pass
            try:
                activity = database.get_node(code=code)
                unit = activity['unit']
                act_name = activity['name']
            except (bw2data.errors.UnknownObject, KeyError) as e:
                pass
                message=f" \n{code} from activity, {key} not found in the database. Please check your database"
                warnings.warn(message,Warning)
                continue  # If activity doesn't exist, don't do anything

            df=df.reset_index(drop=True)
            for index, row in df.iterrows():
                pass
                if str(key) == str(row['alias_carrier']):
                    factor = (self.activity_conversion_dictionary[key]['factor'])
                    value = float(row['flow_out_sum'])
                    new_val = value * factor
                    df.at[index, 'codes'] = code
                    df.at[index, 'units_new'] = unit
                    df.at[index, 'new_vals'] = new_val
                    df.at[index, 'names_db'] = act_name
                    pass
                else:
                    pass

        return df


    def _clean_final_df(self,df)-> pd.DataFrame:
        """ """
        df.dropna(axis=0, inplace=True)
        self.clean_total=df
        pass
        cols = ['spores',
                'locs',
                'techs',
                'carriers',
                'units_new',
                'new_vals']

        df = df[cols]
        df.rename(columns={'spores': 'scenarios', 'new_vals': 'flow_out_sum'}, inplace=True)
        return df
































