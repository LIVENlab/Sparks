"""
@author: Alexander de Tomás (ICTA-UAB)
        -LexPascal
"""
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from Sparks.generic.generic_dataclass import *
from tqdm import tqdm
tqdm.pandas()
bd.projects.set_current(bw_project)



class Cleaner:
    """Clean the input data and modify the units"""

    def __init__(self,
                 motherfile: str,
                 file_handler: dict,
                 national: Optional[bool]= False,
                 ):

        self.national = national
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
        #todo: remove from, here
        columns = ['spores',"techs","carriers","energy_value"]
        return pd.DataFrame(columns=columns)


    def _load_data(self, source:str) -> pd.DataFrame:
        """ Assuming comma separated input, load the data from a specific file"""
        try:
            return pd.read_csv(self.file_handler[source], delimiter=',').dropna()

        except FileNotFoundError as e:
            raise FileNotFoundError(f"File {source} does not exist") from e


    def _input_checker(self, data: pd.DataFrame, filename: str):
        """
        Check whether the input follows the expected strucutre.
        Assume that last column corresponds to filename (flow_out_sum etc)
        """



        filename = filename.split('.')[0]
        expected_cols = {'spores',
                         'techs',
                         'locs',
                         'carriers',
                         filename}


        # Add 'spores' column if it does not exist
        if 'spores' not in data.columns:
            data['spores'] = 0
        if 'Unnamed: 0' in data.columns:
            data = data.drop('Unnamed: 0', axis=1)

        if 'carriers' not in data.columns:
            data['carriers'] = 'default_carrier'


        try:
            tr = data[list(expected_cols)]
            data.rename(columns={filename: 'energy_value'}, inplace=True)
            data['filename'] = filename

            return data


        except KeyError:
            raise KeyError(f"Columns {data.columns} do not match the expected columns: {expected_cols}")


    def _filter_techs(self,df: pd.DataFrame, filter: str)-> pd.DataFrame:
        """
        Filter the input data based on technologies defined in the basefile
        """
        df_names=df.copy()
        # Filter Processors from calliope data
        df_names['alias_carrier'] = df_names['techs'] + '_' + df_names['carriers']
        df_names['alias_filename'] = df_names['alias_carrier']+ '__' + df_names['filename']
        df_names['full_name'] = df_names['alias_filename'] + '___' + df_names['locs']
        df_names = self._manage_regions(df_names)
        df_names['alias_filename']=df_names['alias_filename']+'___' + df_names['countries']


        if self._edited is False:
            # working with basefile data
            self.basefile = pd.read_excel(self.mother_file, sheet_name='Processors').dropna(subset=['Ecoinvent_key_code'])
            self.basefile['alias_carrier'] = self.basefile['Processor'] + '_' + self.basefile['@SimulationCarrier']
            #self.basefile['alias_region'] = self.basefile['alias_carrier']+'_'+self.basefile['Region']
            self.basefile['alias_filename'] = self.basefile['alias_carrier']+'__'+self.basefile['File_source']
            self.basefile['alias_filename'] = self.basefile['alias_filename'].str.split('.').str[0]
            #TODO: redundant to get it trhough
            self.basefile['alias_filename_loc'] = self.basefile['alias_filename'] + '___' + self.basefile['Region']
            self._edited = True


        basefile = self.basefile.loc[self.basefile['File_source'] == filter]
        excluded_techs = set(df_names['alias_carrier']) - set(basefile['alias_carrier'])
        self.techs_region_not_included=excluded_techs

        df_names = df_names[~df_names['alias_carrier'].isin(excluded_techs)] # exclude the technologies

        return df_names


    def _group_data(self,df: pd.DataFrame)-> pd.DataFrame:
        """
        Group the input data based on technologies defined in the basefile
        If national= True, it aggregates by country
        """

        if self.national:
            pass
            df['locs'] = df['locs'].str.split('_').str[0].str.split('-').str[0]
            grouped_df = df.groupby(['alias_filename', "locs"], as_index=False).agg({ #todo: remove units form here
                'energy_value': 'sum',
                'spores': 'first',
                'techs': 'first',
                'carriers': 'first',
            })

        else:

            grouped_df = df.groupby(['spores',
                                     'techs',
                                     'carriers',
                                     'locs',
                                     'alias_carrier', #remove unit from here
                                     'alias_filename',
                                     'full_name'
                                     ], as_index=False).agg({
                'energy_value': 'sum'
            })
        return grouped_df


    @staticmethod
    def _manage_regions(df:pd.DataFrame)-> pd.DataFrame:
        """
        Edit the regions format strings in order to get general country names
        """
        df['countries'] = [x.split('_')[0].split('-')[0]
                     for x in df['locs']]
        return df


    def preprocess_data(self)->pd.DataFrame:
        """Run data preprocessing steps"""

        self.basefile = pd.read_excel(self.mother_file, sheet_name='Processors')
        all_data = self.create_template_df()
        grouped = self.basefile.groupby('File_source')

        for data_source, group in grouped:
            
            if pd.isna(data_source):
                warnings.warn(f"DataSource is missing for some entries. Skipping these entries.", Warning)
                continue
            try:

                raw_data = self._load_data(data_source) # Calliope data
                checked_data = self._input_checker(data=raw_data, filename = data_source) # calliope data
                filtered_data = self._filter_techs(checked_data, data_source)
                all_data = pd.concat([all_data, filtered_data], ignore_index=True)

            except Exception as e:
                warnings.warn(f"Error processing {data_source}: {e}", Warning)

        if len(self.techs_region_not_included) > 1:
            formatted_items = "\n".join(f"    - {item}" for item in self.techs_region_not_included)
            message = f"""\nThe following technologies are present in the energy data but not in the Basefile: 
            {formatted_items}

            Please, check the following items to avoid missing information."""
            #warnings.warn(message, Warning)

        self.final_df = self._group_data(all_data)
        self.final_df = self._manage_regions(self.final_df)

        return self.final_df

    # noinspection PyArgumentList
    def _extract_data(self) -> List['BaseFileActivity']:
        pass
        def _create_activity(row):
            try:

                return BaseFileActivity(
                    name=row['Processor'],
                    carrier=row['@SimulationCarrier'],
                    parent=row['ParentProcessor'],
                    region=row['Region'],
                    code=row['Ecoinvent_key_code'],
                    factor=row['@SimulationToEcoinventFactor'],
                    full_alias=row['alias_filename_loc']
                )

            except KeyError:
                return None

        base_activities = self.basefile.progress_apply(_create_activity, axis=1).dropna().tolist()
        return [activity for activity in base_activities if activity.unit is not None]
        pass





    def _adapt_units(self):
        """adapt the units (flow_out_sum * conversion factor)"""

        self.base_activities = self._extract_data()
        pass
        alias_to_factor = {x.full_alias: x.factor for x in self.base_activities}
        unit_to_factor = {x.full_alias: x.unit for x in self.base_activities}

        self.final_df['new_vals'] = self.final_df['alias_filename'].map(alias_to_factor) * self.final_df['energy_value']
        self.final_df['new_units'] = self.final_df['alias_filename'].map(unit_to_factor)
        return self._final_dataframe(self.final_df)


    def _final_dataframe(self, df):

        cols = ['spores',
                'locs',
                'techs',
                'full_name',
                'carriers',
                'new_units',
                'new_vals']

        df.dropna(axis=0, inplace=True)
        pass
        if self.national:
            df.rename(columns={'alias_filename' : 'full_name'}, inplace=True)
            
        df = df[cols]
        df.rename(columns={'spores': 'scenarios', 'new_vals': 'energy_value'}, inplace=True)
        df['aliases'] = df['techs'] + '__' + df['carriers'] + '___' + df['locs']
        self._techs_sublocations = df['full_name'].unique().tolist()  # save sublocation aliases for hierarchy

        return df


    def adapt_units(self):
        """Public method to adapt the units"""
        return self._adapt_units()












































