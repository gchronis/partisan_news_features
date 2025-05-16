# NOTE: we are not using this script anymore, but keeping it for reference
# This script generates features for the Pavlick sample words using a trained model from features_in_context
# now, we are generating features for the Pavlick sample words using a trained model from jwalanthi's repo semantic_features


# Change directory
from pathlib import Path
import os
os.chdir(Path("/home/gsc685/partisan_features_3_23/features_in_context/notebooks"))
print("Current Directory:", Path.cwd())

import sys
import torch
sys.path.append("..")
from src.bert import *
import pandas as pd
import io
from tqdm import tqdm
import pyarrow


def get_features(row, model, bert ):
  feature_map = model.feature_norms.feature_map
  feature_labels = [str(feature_map.get_object(i)) for i in range(0, len(feature_map))]
  predictions = model.predict_in_context(row["word"], row["sentence"], bert)

  #create dictionary where key is feature name and value is feature value
  myDict = {feature_labels[i]: pred for i, pred in enumerate(predictions)}
  #sort
  sortedDict = {key: value for key, value in sorted(myDict.items())}

  return sortedDict


if __name__ == "__main__":



    bert = BERTBase()
    #buchanan = torch.load('../trained_models/model.plsr.buchanan.allbuthomonyms.5k.300components.500max_iters')
    binder = torch.load('../trained_models/model.ffnn.binder.5k.50epochs.0.5dropout.lr1e-4.hsize300')


    # Reading a CSV file

    df = pd.read_csv('/home/gsc685/partisan_features_3_23/data/sampled_sentences.csv')
    df = df.iloc[0:1000, :]
    print(len(df))

    #print(binder)

    print(get_features(df.iloc[0], binder, bert))

    # separate by words
    

    data = {'word': [],
#            'sentence': [],
            'cluster': [],
            'feature': [],
            'value': [],
            'token_id': []}
    

    for i in tqdm(range(len(df))):
        try:
            feature_dict = get_features(df.iloc[i], binder, bert)
            for feature, value in feature_dict.items():
                data['word'].append(df.iloc[i]["word"])
#                data['sentence'].append(df.iloc[i]["sentence"])
                data['cluster'].append(df.iloc[i]["cluster"])
                data['feature'].append(feature)
                data['value'].append(value)
                data['token_id'].append(i)
        except Exception as e:
            print(f"Error processing row {i}: {e}")

    #print(data)

    df_with_features = pd.DataFrame.from_dict(data)
    #print(df_with_features)

    #df.to_csv('sampled_sentences_with_ids.csv', index = True, index_label="id")
    df.to_csv('/home/gsc685/partisan_features_3_23/data/sampled_sentences_with_ids.csv', index = True, index_label="id")
    df_with_features.to_parquet("/home/gsc685/partisan_features_3_23/data/sampled_words_with_features.csv", engine='pyarrow')