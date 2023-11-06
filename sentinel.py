import pandas as pd
from ProspectBackground.util.update_experiment import Prospect

enbios_mod = Prospect(r'C:\Users\altz7\PycharmProjects\enbios__git\projects\data_sentinel\flow_out_sum.csv',
                       r'C:\Users\altz7\PycharmProjects\enbios__git\projects\seed\MixUpdater\data\base_file_simplified_v2.xlsx',
                       'Seeds_exp4', 'db_experiments')

enbios_mod.preprocess(subregions=True)
enbios_mod.data_for_ENBIOS(smaller_vers=1)
enbios_mod.template_electricity('Electricity_generation', Units='kWh')

