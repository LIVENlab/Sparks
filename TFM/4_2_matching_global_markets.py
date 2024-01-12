import bw2data as bd
import json


bd.projects.set_current('Lex_TFM')
ei=bd.Database('ecoinvent')
class Matching:
    def __init__(self):
        self.exporter=None
        self.__TO_BE_INCLUDED=[]
        pass

    def export_usa_data(self):
        """
        Modify the ecoinvent data to facilitate programming tasks
        {region: {act_name {code, share}}}

        If there's no activity: TO BE INCLUDED
        """
        with open('ecoinvent_map.json') as file:
            data = json.load(file)
            new = {}

            for region, activities in data.items():
                internal = {}
                pass
                for activity_name, activity_data in activities.items():
                    try:
                        pass

                        name=activity_data['name']+"_"+ activity_data['loc']
                        internal[name]={
                            "share": activity_data['share'],
                            "code":activity_data['activities']['code']
                        }

                    except:
                        print(f"Exception {activity_data['activities']}")
                        pass
                        self.__TO_BE_INCLUDED.append(activity_data)
                        internal[activity_name] = {
                            "share": activity_data.get('share', None),
                            "code": "TO_BE_INCLUDED"
                        }

                new[region] = internal

            self.exporter = new

            with open('clean.json', 'w') as file:
                json.dump(new, file, indent=4)


match=Matching()
match.export_usa_data()