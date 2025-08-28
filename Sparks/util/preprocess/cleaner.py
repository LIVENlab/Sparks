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
from dataclasses import asdict
from pandas.api.types import is_numeric_dtype


class Cleaner:
    """Clean the input data and modify the units"""

    def __init__(self,
                 motherfile: str,
                 file_handler: dict,
                 national: Optional[bool]= False,
                 specify_database: Optional[bool] =False,
                 additional_columns: Optional [List[str]] = None
                 ):

        self.national = national
        self.specify_database = specify_database
        self.mother_file = motherfile
        self.final_df = None
        self.techs_region_not_included = []
        self.file_handler = file_handler
        self.additional_columns = additional_columns or []
        self._edited = False


    @staticmethod
    def create_template_df() -> pd.DataFrame:
        """
        Basic template for clean data
        """
        #todo: remove from, here
        columns = ['spores',"techs","carriers","energy_value"]
        return pd.DataFrame(columns=columns)


    def verify_csv(self, source:str)-> None:
        """
        Verify that the value passed has a csv extension.
        """
        if not source.endswith('.csv'):
            raise ValueError(f"File {source} is not a csv file")
        
        
    def _load_data(self, source:str) -> pd.DataFrame:
        """ Assuming comma separated input, load the data from a specific file"""
        self.verify_csv(source)
        try:
            return pd.read_csv(self.file_handler[source], sep = None, engine='python').dropna()
        
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File {source} does not exist") from e
        

    def _input_checker(self, data: pd.DataFrame, filename: str) -> pd.DataFrame:
        """
        Validate and standardize the input DataFrame structure.

        This function ensures that required columns are present, renames specific columns for consistency,
        and injects default values where necessary. It assumes that the column corresponding to the `filename`
        parameter contains the main energy value to be analyzed.

        Scenarios / spores are now treated as synonyms
        Locs / nodes are now consiered synonyms


        ----------
        data : pd.DataFrame
            The input data to validate.
        filename : str
            Name of the file, used to identify the corresponding data column.

        Returns
        -------
        pd.DataFrame
            The validated and reformatted DataFrame.
        """


        filename_base = filename.split('.')[0]

        # Standardize scenario/spores column
        if 'spores' not in data.columns:
            if 'scenario' in data.columns:
                data = data.rename(columns={'scenario': 'spores'})
            else:
                data['spores'] = 0  # Default spores value

        if 'locs' not in data.columns:
            if 'nodes' in data.columns:
                data = data.rename(columns={'nodes': 'locs'})


        if 'Unnamed: 0' in data.columns:
            data = data.drop(columns='Unnamed: 0')


        if 'carriers' not in data.columns:
            data['carriers'] = 'default_carrier'


        expected_columns = {'spores', 'techs', 'locs', 'carriers', filename_base}


        missing_columns = expected_columns - set(data.columns)
        if missing_columns:
            raise KeyError(
                f"Missing required column(s): {missing_columns}. "
                f"Expected columns include: {expected_columns}. "
                f"Available columns: {list(data.columns)}"
            )


        data = data.rename(columns={filename_base: 'energy_value'})
        data['filename'] = filename_base


        if data.empty:
            raise ValueError(
                f"Error processing '{filename}'. No rows found after processing. "
                f"Verify that the input file contains valid data."
            )

        return data


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


        if self._edited is False:  # working with basefile data

            if self.specify_database: # check that the database column is there and contains no empty values
                self._validate_databases()

            self.basefile = pd.read_excel(self.mother_file, sheet_name='Processors').dropna(subset=['Ecoinvent_key_code'])
            self.basefile['alias_carrier'] = self.basefile['Processor'] + '_' + self.basefile['@SimulationCarrier']
            #self.basefile['alias_region'] = self.basefile['alias_carrier']+'_'+self.basefile['Region']
            self.basefile['alias_filename'] = self.basefile['alias_carrier']+'__'+self.basefile['File_source']
            self.basefile['alias_filename'] = self.basefile['alias_filename'].str.split('.').str[0]

            self.basefile['alias_filename_loc'] = self.basefile['alias_filename'] + '___' + self.basefile['Region']
            self.basefile['full_alias'] = self.basefile['alias_filename_loc'] + '-' + self.basefile['geo_loc'] # TODO: make optional
            self._edited = True


        basefile = self.basefile.loc[self.basefile['File_source'] == filter]
        excluded_techs = set(df_names['alias_carrier']) - set(basefile['alias_carrier'])
        self.techs_region_not_included=excluded_techs

        df_names = df_names[~df_names['alias_carrier'].isin(excluded_techs)] # exclude the technologies

        return df_names


    def _validate_databases(self):
        """
        if specify database is activated, check that the basefile has non NaN values
        """
        if 'database' not in self.basefile.columns:
            raise KeyError(f"Please, specify the database column in the basefile")

        if self.basefile['database'].isnull().any():
            raise ValueError(f"The column database from the basefile containse missing values")



    def _group_data(self,df: pd.DataFrame)-> pd.DataFrame:
        """
        Group the input data based on technologies defined in the basefile
        If national= True, it aggregates by country
        """

        if self.additional_columns:
            # Safely combine spores + additional columns into a single string key
            df['spores'] = df['spores'].astype(str)
            for col in self.additional_columns:
                if col in df.columns:
                    df['spores'] += '_' + df[col].astype(str)
        else:
            df['spores'] = df['spores'].astype(str)

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

            group_cols = ['spores',
                          'techs',
                          'carriers',
                          'locs',
                          'alias_carrier',
                          'alias_filename',
                          'full_name']

            grouped_df = df.groupby(group_cols, as_index=False).agg({
                'energy_value': 'sum'
            })
        pass
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

                filtered_data = self._filter_techs(checked_data, data_source) # calliope data

                all_data = pd.concat([all_data, filtered_data], ignore_index=True)

            except ValueError as e:
                # Propagate validation errors (e.g., from verify_csv) as real errors
                raise
            except Exception as e:
                warnings.warn(f"Error processing {data_source}: {e}", Warning)

        if len(self.techs_region_not_included) > 1:
            formatted_items = "\n".join(f"    - {item}" for item in self.techs_region_not_included)
            message = f"""\nThe following technologies are present in the energy data but not in the Basefile: 
            {formatted_items}

            Please, check the following items to avoid missing information."""
            warnings.warn(message, Warning)
        pass
        self.final_df = self._group_data(all_data) # calliope data
        self.final_df = self._manage_regions(self.final_df)

        return self.final_df



    # noinspection PyArgumentList
    def _extract_data(self) -> List['BaseFileActivity']:
        """
        extract activities from the basefile and create a list BasFileActivity instances
        """
        pass
        def _create_activity(row):
            # move the activities from the basefile into a DataBase dataclass

            try:
                kwargs= {
                    'name': row['Processor'],
                    'carrier': row['@SimulationCarrier'],
                    'parent': row['ParentProcessor'],
                    'region': row['Region'],
                    'code': row['Ecoinvent_key_code'],
                    'factor': row['@SimulationToEcoinventFactor'],
                    'full_alias': row['full_alias'],
                    'alias_filename_loc': row['alias_filename_loc']
                }

                if self.specify_database:
                    kwargs['database']=row['database']

                return BaseFileActivity(**kwargs)


            except KeyError:
                return None

        base_activities = self.basefile.progress_apply(_create_activity, axis=1).dropna().tolist()
        return [activity for activity in base_activities if activity.unit is not None]






    def _adapt_units(self):
        """adapt the units (flow_out_sum * conversion factor)"""
        self.base_activities = self._extract_data()

        df = pd.DataFrame([asdict(activity) for activity in self.base_activities])

        df = pd.merge(
            self.final_df,
            df,
            left_on="full_name",
            right_on="alias_filename_loc",
            how="right"
        )

        df=df.dropna()
        df=self._check_str_values(df, 'energy_value')
        df['new_vals'] = df['factor'] * df['energy_value']
        df['new_units'] =df['unit']

        return self._final_dataframe(df)

    @staticmethod
    def _check_str_values(df: pd.DataFrame, column: str, cast_to_int: bool = False):
        """
        if ';' passed, issues may rise. This function checks a particular column that could potentially be a string
        instead of float
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")

        if not is_numeric_dtype(df[column]):
            df[column] = (
                df[column]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .str.replace(" ", "", regex=False)
                .str.strip()
            )
            df[column] = pd.to_numeric(df[column], errors="coerce")

        if cast_to_int:
            df[column] = df[column].astype("Int64")  # Nullable integer type for missing values

        return df

    def _final_dataframe(self, df):


        cols = ['spores',
                'locs',
                'techs',
                'full_alias',
                'carriers',
                'new_vals',
                'new_units']

        df.dropna(axis=0, inplace=True)

        if self.national:
            df.rename(columns={'alias_filename' : 'full_name'}, inplace=True)
            
        df = df[cols]

        df.rename(columns={'full_alias':'full_name', 'spores': 'scenarios', 'new_vals': 'energy_value'}, inplace=True)
        df['aliases'] = df['techs'] + '__' + df['carriers'] + '___' + df['locs'] # TODO: remove in clean versions

        self._techs_sublocations = df['full_name'].unique().tolist()  # save sublocation aliases for hierarchy

        return df


    def adapt_units(self):
        """Public method to adapt the units"""
        return self._adapt_units()












































