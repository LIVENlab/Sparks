"""
@author: Alexander de Tomás (ICTA-UAB)
        -LexPascal
"""
import bw2data
import pandas as pd
from typing import Dict, Union, Optional, Any, List
import warnings
import bw2data as bd
from bw2data.errors import UnknownObject
from Sparks.const.const import bw_project, bw_db
from bw2data.backends import Activity, ActivityDataset
from dataclasses import dataclass
from Sparks.generic.generic import *

bd.projects.set_current(bw_project)
database = bd.Database(bw_db)


class Cleaner:
    """Clean the input data and modify the units"""

    def __init__(self,
                 caliope: str,
                 motherfile: str,
                 subregions : [Optional,bool]= False):

        self.subregions = subregions
        self._raw_data = caliope
        self.mother_file = motherfile
        self.final_df = None
        self.techs_region_not_included = []


    @staticmethod
    def create_template_df() -> pd.DataFrame:
        """
        Basic template for clean data
        """
        columns = ["spores","techs","locs","carriers","unit","flow_out_sum"]
        return pd.DataFrame(columns=columns)


    def _load_data(self) -> pd.DataFrame:
        """ Assuming comma separated input"""
        return pd.read_csv(self._raw_data, delimiter=',').dropna()


    def _input_checker(self):
        """
        Check whether the input from Calliope follows the expected structure
        """
        data = self._load_data()
        expected_cols = {'spores',
                         'techs',
                         'locs',
                         'carriers',
                         'unit',
                         'flow_out_sum'}
        cols = set(data.columns) # Search for possible differences. Older versions of calliope produce "spore"

        if "spore" in cols:
            data.rename(columns={'spore': 'spores'}, inplace=True)
            cols = set(data.columns)

        if expected_cols != cols:
            raise KeyError(f"Columns {cols} do not match the expected columns: {expected_cols}")


    def _get_region_from_location(self, loc: str) -> str:
        """ Extract the region from the location string"""
        if loc == 'ESP-sink':
            return 'ESP'
        return loc.split('-')[0].split('_')[0]


    def _group_data(self, df: pd.DataFrame)-> pd.DataFrame:
        """ Group input data by specified criteria"""

        grouped_df = self.create_template_df()
        scenarios = df['spores'].unique()

        if not self.subregions:
            df['locs'] = df['locs'].apply(self._get_region_from_location)

        for scenario in scenarios:
            df_sub = df[df['spores'] == scenario]
            df_sub = df_sub.groupby(['techs', 'locs', 'carriers']).agg({
                "spores": "first",
                "unit": "first",
                "flow_out_sum": "sum"
            }).reset_index()
            grouped_df = pd.concat([grouped_df, df_sub])

        return grouped_df


    def _adapt_data(self)->pd.DataFrame:
        print('Adapting input data...')
        try:
            df=self._load_data()
            self._input_checker()
        except FileNotFoundError:
            raise FileNotFoundError(f'File {self._raw_data} does not exist. Please check it')
        return self._group_data(df)



    def _filter_techs(self,df: pd.DataFrame)-> pd.DataFrame:
        """
        Filter the input data based on technologies defined in the basefile
        """

        df_names=df.copy()
        self.basefile=pd.read_excel(self.mother_file, sheet_name='Processors')
        # Filter Processors
        df_names['alias_carrier']=df_names['techs'] + '_' +df_names['carriers']
        df_names['alias_region']=df_names['alias_carrier'] + '_' +df_names['locs']
        self.basefile['alias_carrier']=self.basefile['Processor']+ '_' + self.basefile['@SimulationCarrier']
        self.basefile['alias_region']=self.basefile['alias_carrier']+'_'+self.basefile['Region']
        excluded_techs = set(df_names['techs']) - set(self.basefile['Processor'])
        self.techs_region_not_included=excluded_techs
        df_names = df_names[~df_names['techs'].isin(excluded_techs)] # exclude the technologies

        if excluded_techs:
            message=f'''\nThe following technologies, are present in the energy data but not in the Basefile: 
            \n{excluded_techs}
            \n Please,check the following items in order to avoid missing information'''
            warnings.warn(message,Warning)
        return df_names


    def preprocess_data(self)->pd.DataFrame:
        """Run data preprocessing steps"""
        self.final_df = self._filter_techs(self._adapt_data())
        return self.final_df


    def _extract_data(self)->List['BaseFileActivity']:
        base_activities=[]
        for _,r in self.basefile.iterrows():
            base_activities.append(
                BaseFileActivity(
                    name=r['Processor'],
                    carrier=r['@SimulationCarrier'],
                    parent=r['ParentProcessor'],
                    region=r['Region'],
                    code=r['Ecoinvent_key_code'],
                    factor=r['@SimulationToEcoinventFactor']
                )
            )

        return  [activity for activity in base_activities if activity.unit is not None]


    def _adapt_units(self):
        """adapt the units (flow_out_sum * conversion factor)"""
        self.base_activities=self._extract_data()

        alias_to_factor = {x.alias_carrier: x.factor for x in self.base_activities}
        unit_to_factor = {x.alias_carrier: x.unit for x in self.base_activities}

        self.final_df['new_vals'] = self.final_df['alias_carrier'].map(alias_to_factor) * self.final_df['flow_out_sum']
        self.final_df['new_units'] =self.final_df['alias_carrier'].map(unit_to_factor)

        return self._final_dataframe(self.final_df)



    def _final_dataframe(self,df):
        cols = ['spores',
                'locs',
                'techs',
                'carriers',
                'new_units',
                'new_vals']
        df.dropna(axis=0, inplace=True)
        df = df[cols]
        df.rename(columns={'spores': 'scenarios', 'new_vals': 'flow_out_sum'}, inplace=True)
        df['aliases'] = df['techs'] + '__' + df['carriers'] + '___' + df['locs']
        self._techs_sublocations=df['aliases'].unique().tolist() # save sublocation aliases for hierarchy
        return df


    def adapt_units(self):
        """Public method to adapt the units"""
        return self._adapt_units()









































