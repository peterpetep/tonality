import pandas as pd

songs = pd.read_csv("./spotify_songs_predicted.csv")
print(songs.sort_values("arousal"))
