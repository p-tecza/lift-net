import pandas as pd

def override_incident(type):
    if type == 'Incident':
        return 'Problem'
    return type

data = pd.read_csv("scripts/en-tickets.csv")

data['type'] = data['type'].apply(lambda x: override_incident(x))

data.to_csv('en-3-classes.csv')