from Sparks.util.base import SoftLink


enbios_mod = SoftLink(r'C:\Users\altz7\PycharmProjects\ENBIOS4TIMES_\testing\data_test\flow_out_sum.csv',
                      r'C:\Users\altz7\PycharmProjects\ENBIOS4TIMES_\testing\data_test\basefile.xlsx',
                       'Seeds_exp4',
                      'db_experiments')

enbios_mod.preprocess(subregions=False)
enbios_mod.data_for_ENBIOS(smaller_vers=True, path_save=r'data/test.json')