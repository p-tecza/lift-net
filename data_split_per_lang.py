import pandas as pd
from langdetect import detect, DetectorFactory
from tqdm import tqdm


data = pd.read_csv("dataset-tickets-multi-lang-4-20k.csv")

data['subject'] = data['subject'].fillna('')
data['body'] = data['body'].fillna('')

en_data = pd.DataFrame(data[data['language']=='en'])
de_data = pd.DataFrame(data[data['language']=='de'])

print("en-data size before filter:", len(en_data))
print("de-data size before filter:", len(de_data))

tqdm.pandas()

# Wykryj język
en_data['lang'] = en_data['body'].progress_apply(lambda x: detect(x) if x.strip() else 'unknown')
de_data['lang'] = de_data['body'].progress_apply(lambda x: detect(x) if x.strip() else 'unknown')

# Filtrowanie tylko angielskich
en_data = en_data[en_data['lang'] == 'en']
de_data = de_data[de_data['lang'] == 'de']

print("en-data size after filter:", len(en_data))
print("de-data size after filter:", len(de_data))

en_data.to_csv("en-tickets.csv")
de_data.to_csv("de-tickets.csv")
