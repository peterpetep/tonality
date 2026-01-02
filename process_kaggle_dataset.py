import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = "spotify_songs.csv"

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "imuhammad/audio-features-and-lyrics-of-spotify-songs",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

print("First 5 records:", df.head())

# Save the dataset locally
df.to_csv('spotify_songs_local.csv', index=False)
print("Dataset saved locally as 'spotify_songs_local.csv'")

#track id, title, language, date, key, lyrics
# english, 21st cent, has key, has lyrics


df_final = df.loc[
    (df['language'] == 'en') & 
    (df['track_album_release_date'].str[0] == '2') &
    (df['key'] > -1) & 
    (df['mode'] >= 0) &
    (df['lyrics'] != "NA") &
    (df['lyrics'] != "Lyrics for this song have yet to be released. Please check back once the song has been released."), 
    ['track_id','track_name','track_album_release_date','lyrics', 'key', 'mode', 'language']
]

df_final.to_csv('spotify_songs_filtered.csv', index=False)