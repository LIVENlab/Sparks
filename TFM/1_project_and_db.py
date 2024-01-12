
import bw2io as bi
import bw2data as bd

#create project
bd.projects.set_current('TFM_Lex')
bi.bw2setup()
# spolds
spolds= r'C:\Users\Administrator\Documents\Alex\ecoinvent\ecoinvent 3.9.1_cutoff_ecoSpold02\datasets'
database='ecoinvent'
#Import Ecoinvent v 3.9.1
ei=bi.SingleOutputEcospold2Importer(spolds,database,use_mp=False)
ei.apply_strategies()
ei.write_database()
