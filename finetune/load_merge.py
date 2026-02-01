import pandas as pd
import os
from pathlib import Path
from datasets import Dataset, DatasetDict

def load_lyrics(song, quad, base_path="finetune/MERGE_Lyrics_Balanced"):
    
    lyrics_path = os.path.join(base_path, quad, f"{song}.txt")
    if os.path.exists(lyrics_path):
        with open(lyrics_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return None

def create_dataset(split_df, av_df, base_path="finetune/MERGE_Lyrics_Balanced"):

    merged = split_df.merge(av_df, on='Song', how='left')
    
    lyrics_list = []
    missing_songs = []
    
    for idx, row in merged.iterrows():
        song_id = row['Song']
        lyrics = load_lyrics(song_id, row['Quadrant'], base_path)
        
        if lyrics is None:
            missing_songs.append(song_id)
            lyrics = ""
        
        lyrics_list.append(lyrics)
    
    merged['Lyrics'] = lyrics_list
    
    
    if missing_songs:
        print(f"NO LYRICS FOR {len(missing_songs)} SONGS")
        print(missing_songs[:10])  # Show first 10
    
    merged = merged[merged['Lyrics'] != ""]
    
    return merged


av = pd.read_csv("finetune/MERGE_Lyrics_Balanced/merge_lyrics_balanced_av_values.csv")
test = pd.read_csv("finetune/MERGE_Lyrics_Balanced/tvt_dataframes/tvt_70_15_15/tvt_70_15_15_test_lyrics_balanced.csv")
train = pd.read_csv("finetune/MERGE_Lyrics_Balanced/tvt_dataframes/tvt_70_15_15/tvt_70_15_15_train_lyrics_balanced.csv")
validate = pd.read_csv("finetune/MERGE_Lyrics_Balanced/tvt_dataframes/tvt_70_15_15/tvt_70_15_15_validate_lyrics_balanced.csv")

ted = create_dataset(test, av)
trd = create_dataset(train, av)
vad = create_dataset(validate, av)



train_hf = Dataset.from_pandas(trd[['Lyrics', 'Arousal', 'Valence']])
test_hf = Dataset.from_pandas(ted[['Lyrics', 'Arousal', 'Valence']])
validate_hf = Dataset.from_pandas(vad[['Lyrics', 'Arousal', 'Valence']])

dataset_dict = DatasetDict({
    'train': train_hf,
    'test': test_hf,
    'validation': validate_hf
})

print(dataset_dict)

dataset_dict.save_to_disk("finetune/MERGE_PREPPED")