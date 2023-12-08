import json
from pathlib import Path

base = Path("###FOLDER WITH ALL THE RESULTS JSONS")
# grab all the result files, and get the top level results
total_res = []
for index in range(len(list(base.glob("*.json")))):
    print(index)
    file = base / f"{index}.json"
    total_res.append(json.load(file.open())[index]["results"])


# store to results.json file
with open("results_blaaaa.json", "w") as f:
    json.dump(total_res, f)


# cols = ["Scenario"] + list#(total_res[0].keys())
