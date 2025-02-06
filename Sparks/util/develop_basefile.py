import pandas as pd
import bw2data as bd
import bw2io as bi

class Support():
    def __init__(self,
                 file: [str],
                 project: [str],
                 ):
        """
        @file_path: str
        Path to the folder with the files to be used
        @project: str
        Name of the bw project
        @multiple_codes: bool
        Boolean: when True, it allows the possibility of using two bw codes for one single activity
        This is necessary in cases where you want to consider operation and construction separately
        """
        self.file = file
        self.project = project
        bd.projects.set_current(self.project)
        pass

    def _read_excel(self, file):  # Added self as the first parameter
        self.om = pd.read_excel(file, sheet_name='o&m')  # Changed o&m to om
        self.infrastructure = pd.read_excel(file, sheet_name='infrastructure')
        pass


