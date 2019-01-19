import json
import string
import pickle

# decoder = json.JSONDecoder()
defs = {}
for char in string.ascii_lowercase:
    print(char)
    with open(f'OPTED-to-JSON/json/{char}.json', 'rb') as handle:
        defs.update(json.loads(handle.read()))

with open('all.json', 'wb') as pickle_file:
     pickle.dump(defs, pickle_file)
