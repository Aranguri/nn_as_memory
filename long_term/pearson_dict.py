import urllib.request
import json

def pearson_meaning(word):
    try:
        with urllib.request.urlopen(f'http://api.pearson.com/v2/dictionaries/wordwise,lasde/entries?headword={word}') as response:
           html = response.read()
           values = json.loads(html)
           for results in values['results']:
               dict_ = results['datasets'][0]
               defs = results['senses'][0]['definition']
               return defs if type(defs) == str else defs[0]
           return None
    except:
        return None
