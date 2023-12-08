import pathlib

import bw2data as bd
import pandas as pd
import time
from tqdm import tqdm
from ProspectBackground.const.const import bw_project,bw_db


bd.projects.set_current(bw_project)            # Select your project
ei = bd.Database(bw_db)        # Select your db
print(bd.projects.dir)




a=ei.get_node('81174ec2c20931c1a36f65c654bbd11e')
for input in a.technosphere():
    print(input)

for ex in a.biosphere():
    print(a)
pass

act=[a for a in ei if "electricity production, hydro, run-of-river" in str(a['name']) and 'IN' in str(a['location'])]

market=[a for a in ei if 'market for electricity' in str(a['name']) and 'CN' in str(a['location'])]


#aaaa=ei.get_node(name='market for electricity, high voltage', location='CN')
pass












stuff=[
('ReCiPe 2016 v1.03, midpoint (H)', 'climate change', 'global warming potential (GWP1000)') ,
('ReCiPe 2016 v1.03, midpoint (H)', 'water use', 'water consumption potential (WCP)') ,
('ReCiPe 2016 v1.03, midpoint (H)', 'material resources: metals/minerals', 'surplus ore potential (SOP)') ,
    ('ReCiPe 2016 v1.03, midpoint (H)', 'eutrophication: freshwater', 'freshwater eutrophication potential (FEP)'),
    ('ReCiPe 2016 v1.03, midpoint (H)', 'land use', 'agricultural land occupation (LOP)')

]

from bw2analyzer import ContributionAnalysis
import bw2data as bd


# activity = db.random()
# 'electricity production, oil' (kilowatt hour, SE, None)
activity = ei.get_node('fd6356e68720aec505c69cdadb3046fc')

method = ('EDIP 2003 no LT', 'global warming no LT', 'GWP 100a no LT')
lca = activity.lca(method, amount=1)

contribution_ana = ContributionAnalysis()

# results a list of tuples (lca score, supply, activity)
contributions = contribution_ana.annotated_top_processes(lca, limit=25)
print(contributions)






