import pandas as pd
from ProspectBackground.util.update_experiment import Prospect

enbios_mod = Prospect('/home/lex/Documents/data_sentinel/flow_out_sum.csv',
                       '/home/lex/PycharmProjects/SparkBox/ProspectBackground/data/base_file_simplified_v2.xlsx',
                       'Seeds_exp4', 'db_experiments')


enbios_mod.preprocess(subregions=False)
enbios_mod.data_for_ENBIOS(smaller_vers=1)
enbios_mod.template_electricity('Electricity_generation', Units='kWh')

enbios_mod.updater_run('/home/lex/PycharmProjects/SparkBox/result_test')

