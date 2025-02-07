from Sparks.util.base import SoftLink

enbios_mod = SoftLink(r'C:\Users\altz7\PycharmProjects\ENBIOS4TIMES_\testing\data_test',
                    'Hydrogen_SEEDS')
pass
enbios_mod.sup_basefile()
enbios_mod.preprocess()
enbios_mod.data_for_ENBIOS(smaller_vers=False, path_save=r'test.json')