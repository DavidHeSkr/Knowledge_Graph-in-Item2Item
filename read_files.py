import json
import pandas as pd
with open('rel2movies.json') as jsonfile:
    rel2movie = json.load(jsonfile)
with open('rel2ids.json') as jsonfile:
    rel2ids = json.load(jsonfile)
movie2rels = pd.read_csv("movie2rels.csv")

print(rel2movie)
print(rel2ids)
print(movie2rels)
