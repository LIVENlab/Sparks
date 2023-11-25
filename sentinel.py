
"""
EXAMPLE
"""
from ProspectBackground.util.update_experiment import Prospect

enbios_mod = Prospect('/home/lex/Downloads/flow_out_sum.csv',
                       '/home/lex/PycharmProjects/SparkBox/ProspectBackground/data/base_file_simplified_v2.xlsx',
                       'Seeds_exp4', 'db_experiments_2')
enbios_mod.preprocess(small_vers=None,subregions=False)
enbios_mod.data_for_ENBIOS()
enbios_mod.template_electricity('Electricity_generation', Units='kWh')
enbios_mod.updater_run('/home/lex/PycharmProjects/SparkBox/result_test')

