
import bw2data as bd
from bw2calc import LCA
from bw2calc.monte_carlo import MonteCarloLCA, MultiMonteCarlo

bd.projects.set_current("ecoinvent_391")  # Select your project
ei = bd.Database("ecoinvent_391_cutoff")

act = ei.get_node('a8fe0b37705fe611fac8004ca6cb1afd')
act2 = ei.get_node('413bc4617794c6e8b07038dbeca64adb')

method = ('CML v4.8 2016', 'climate change', 'global warming potential (GWP100)')

# Fer un LCA simple

for i in range(1):
    lca = LCA({act: 1, act2: 3}, method, use_distributions=True)
    lca.lci()
    lca.lcia()
    print(lca.score)

demands1 = [{act.key: 1}]
print(act.key)
demands = [{act: 1,
            act2: 10}]


# mc = MonteCarloLCA(demands[0], data_objs=[])
# import pickle
# pickle.dumps(mc)

mc = MultiMonteCarlo(demands1, method=method)
#
mc.calculate()
