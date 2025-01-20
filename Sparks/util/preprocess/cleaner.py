"""
@author: Alexander de Tomás (ICTA-UAB)
        -LexPascal
"""
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from Sparks.generic.generic_dataclass import *

bd.projects.set_current(bw_project)
database = bd.Database(bw_db)


class Cleaner:
    """Clean the input data and modify the units"""

    def __init__(self,
                 caliope: str,
                 motherfile: str,
                 file_handler: dict,
                 subregions : [Optional,bool]= False,
                 ):

        self.subregions = subregions
        self._raw_data = caliope
        self.mother_file = motherfile
        self.final_df = None
        self.techs_region_not_included = []
        self.file_handler = file_handler

        self._edited = False



    @staticmethod
    def create_template_df() -> pd.DataFrame:
        """
        Basic template for clean data
        """
        columns = ['spores',"techs","carriers","unit","energy_value"]
        return pd.DataFrame(columns=columns)


    def _load_data(self, source:str) -> pd.DataFrame:
        """ Assuming comma separated input, load the data from a specific file"""
        try:
            return pd.read_csv(self.file_handler[source], delimiter=',').dropna()
        except:
            FileNotFoundError(f"File {source} does not exist")


    def _input_checker(self, data: pd.DataFrame, filename: str):
        """
        Check whether the input follows the expected strucutre.
        Assume that last column corresponds to filename (flow_out_sum etc)
        """
        filename=filename.split('.')[0]
        expected_cols = {'spores',
                         'techs',
                         'locs',
                         'carriers',
                         'unit',
                         filename}
        # Add 'spores' column if it does not exist
        if 'spores' not in data.columns:
            data['spores'] = 0
        try:
            data[list(expected_cols)]
            data['country'] = data['locs'].str.split('_').str[0]

            return data.rename(columns={filename: 'energy_value'})
        except KeyError:
            raise KeyError(f"Columns {data.columns} do not match the expected columns: {expected_cols}")


    def _filter_techs(self,df: pd.DataFrame, filter: str)-> pd.DataFrame:
        """
        Filter the input data based on technologies defined in the basefile
        """
        df_names=df.copy()
        # Filter Processors
        df_names['alias_carrier']=df_names['techs'] + '_' +df_names['carriers']
        df_names['alias_region']=df_names['alias_carrier'] + '_' +df_names['locs']


        if self._edited is False:
            self.basefile = pd.read_excel(self.mother_file, sheet_name='Processors').dropna(subset=['Ecoinvent_key_code'])
            self.basefile['alias_carrier']=self.basefile['Processor'] + '_' + self.basefile['@SimulationCarrier']
            self.basefile['alias_region']=self.basefile['alias_carrier']+'_'+self.basefile['Region']
            self._edited=True

        basefile=self.basefile.loc[self.basefile['File_source']==filter]


        excluded_techs = set(df_names['alias_carrier']) - set(basefile['alias_carrier'])
        self.techs_region_not_included=excluded_techs
        df_names = df_names[~df_names['alias_carrier'].isin(excluded_techs)] # exclude the technologies
        pass

        if excluded_techs:
            message=f'''\nThe following technologies, are present in the energy data but not in the Basefile: 
            \n{excluded_techs}
            \n Please,check the following items in order to avoid missing information'''
            warnings.warn(message,Warning)
        return df_names


    def _group_data(self,df: pd.DataFrame)-> pd.DataFrame:
        grouped_df = df.groupby(['spores', 'techs', 'carriers', 'unit', 'locs', 'alias_carrier'], as_index=False).agg({
            'energy_value': 'sum'
        })
        return grouped_df


    def preprocess_data(self)->pd.DataFrame:
        """Run data preprocessing steps"""

        self.basefile = pd.read_excel(self.mother_file, sheet_name='Processors')
        all_data=self.create_template_df()
        grouped= self.basefile.groupby('File_source')

        for data_source, group in grouped:
            if pd.isna(data_source):
                warnings.warn(f"DataSource is missing for some entries. Skipping these entries.", Warning)
                continue
            try:
                # Load data once for the data source
                raw_data = self._load_data(data_source)
                checked_data = self._input_checker(data=raw_data, filename=data_source)
                filtered_data = self._filter_techs(checked_data,data_source)
                all_data = pd.concat([all_data, filtered_data], ignore_index=True)
            except Exception as e:
                warnings.warn(f"Error processing {data_source}: {e}", Warning)
        pass
        self.final_df = self._group_data(all_data)
        pass
        return self.final_df



    def _extract_data(self) -> List['BaseFileActivity']:
        base_activities = []
        pass
        for _, r in self.basefile.iterrows():
            try:
                base_activities.append(
                    BaseFileActivity(
                        name=r['Processor'],
                        carrier=r['@SimulationCarrier'],
                        parent=r['ParentProcessor'],
                        region = r['Region'],
                        code=r['Ecoinvent_key_code'],
                        factor=r['@SimulationToEcoinventFactor']
                    )
                )
            except: #TODO: better handle this
                continue

        return [activity for activity in base_activities if activity.unit is not None]

    def _adapt_units(self):
        """adapt the units (flow_out_sum * conversion factor)"""
        pass
        self.base_activities = self._extract_data()
        pass
        alias_to_factor = {x.alias_carrier: x.factor for x in self.base_activities}
        unit_to_factor = {x.alias_carrier: x.unit for x in self.base_activities}

        self.final_df['new_vals'] = self.final_df['alias_carrier'].map(alias_to_factor) * self.final_df['energy_value']
        self.final_df['new_units'] = self.final_df['alias_carrier'].map(unit_to_factor)
        return self._final_dataframe(self.final_df)


    def _final_dataframe(self, df):
        cols = ['spores',
                'locs',
                'techs',
                'carriers',
                'new_units',
                'new_vals']
        pass
        df.dropna(axis=0, inplace=True)
        df = df[cols]
        df.rename(columns={'spores': 'scenarios', 'new_vals': 'energy_value'}, inplace=True)
        df['aliases'] = df['techs'] + '__' + df['carriers'] + '___' + df['locs']
        self._techs_sublocations = df['aliases'].unique().tolist()  # save sublocation aliases for hierarchy
        return df


    def adapt_units(self):
        """Public method to adapt the units"""
        return self._adapt_units()









































