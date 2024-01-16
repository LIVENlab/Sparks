
import bw2data as bd
from bw2calc import LCA
from bw2calc.monte_carlo import MonteCarloLCA, MultiMonteCarlo
import json

class MonteCarlo:
    pass




bd.projects.set_current("TFM_Lex")  # Select your project
ei = bd.Database("ecoinvent")

act = ei.get_node("fd6356e68720aec505c69cdadb3046fc")
pass
#act2 = ei.get_node('413bc4617794c6e8b07038dbeca64adb')

method = ('CML v4.8 2016', 'climate change', 'global warming potential (GWP100)')

# Fer un LCA simple

for i in range(2):
    lca = LCA({act: 1}, method, use_distributions=True)
    lca.lci()
    lca.lcia()
    print(lca.score)

demands1 = [{act.key: 1}]
print(act.key)

a=MonteCarloLCA()

#demands = [{act: 1,
 #           act2: 10}]
