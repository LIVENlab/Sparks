"""
@LexPascal

This code aims to add the different missing inventories for the global market modification (chp hydrogen)

"""


import bw2data as bd
from bw2data.errors import DuplicateNode

bd.projects.set_current('TFM_Lex')
ei=bd.Database('ecoinvent')
pass

"""
We are going to create the following activity:
- heat and power co-generation, hydrogen, combined cycle power plant, 400MW electrical | electricity, high voltage | cut-off, U - RoW"
"""


# Take the following activity: heat and power co-generation, natural gas, combined cycle power plant, 400MW electrical, e21e89d2a1cd9d137a27321a08636231
gas=ei.get_node('e21e89d2a1cd9d137a27321a08636231')

# Create a copy

try:
    chp_copy=gas.copy(name='heat and power co-generation, hydrogen, combined cycle power plant, 400MW electrical',code='CHP_hydrogen_2050', location='Row')
    chp_copy.save()
except:
    chp_copy=ei.get_node(code='CHP_hydrogen_2050')
    chp_copy.delete()
    chp_copy=gas.copy(name='heat and power co-generation, hydrogen, combined cycle power plant, 400MW electrical',code='CHP_hydrogen_2050', location='Row')
    chp_copy.save()


# Explore the original technosphere:
for ex in gas.technosphere():
    print(ex)

pass

#load the second db
h2=bd.Database('h2_pem')
# load the 3 hydrogen production technologies
PEM=h2.get_node('c864ccf9cb50864488753874b5d69057')
AWE=h2.get_node('ef87f15226dbd2b2ed93e38bb205e69e')
SOEC=h2.get_node('5056237d93a8bbe12e15a00eaaefd585')


for ex in chp_copy.technosphere():
    if ex.input['name'] =='market for natural gas, high pressure':
        ex.delete()
        print(ex.input['name'], 'has been deleted')
# PEM
exchange=chp_copy.new_exchange(input=PEM, amount=0.0006, type='technosphere')
exchange.save()

# AWE
exchange=chp_copy.new_exchange(input=AWE, amount=0.0009, type='technosphere')
exchange.save()

# SOEC
exchange=chp_copy.new_exchange(input=AWE, amount=0.018, type='technosphere')
exchange.save()




# Let's focus on the biosphere:
# We'll keep the NOx emisisons and modify the water vapor


for ex in chp_copy.biosphere():

    name=str(ex.input)
    if "Water" in name  and "air" in name:
        ex['amount']=0.81 # Check calculation file. Stoichiometry based
        ex.save()
        print('water modified', name, ex)
        continue
    if "Nitrogen oxides" in name:
        print('keeping NOX', name, ex)
        continue
    else:
        print("Deleting...", name, ex)
        ex.delete()
        continue


# Check that it worked:
for ex in chp_copy.biosphere():
    print(ex)


# Create a group market for hydrogen production groupping the 3 main technologies in only one.
try:
    h2_mark=ei.new_activity(name='Group_market_for_hydrogen',code='market_hydrogen_2050',location='GLO')
    h2_mark.save()
except:
    h2_mark=ei.get_node(code='market_hydrogen_2050')
    h2_mark.delete()
    h2_mark = ei.new_activity(name='Group_market_for_hydrogen', code='market_hydrogen_2050', location='GLO')
    h2_mark.save()


# Add the 3 hydrogen production technologies. Assume a 0.33kg per tech
exchange=h2_mark.new_exchange(input=PEM, amount=0.33, type='technosphere')
exchange.save()
# AWE
exchange=h2_mark.new_exchange(input=AWE, amount=0.33, type='technosphere')
exchange.save()
# SOEC
exchange=h2_mark.new_exchange(input=AWE, amount=0.33, type='technosphere')
exchange.save()

h2_mark['unit']='kilogram'

h2_mark.save()
