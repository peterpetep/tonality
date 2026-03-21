import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set_theme()


songs = pd.read_csv("./spotify_songs_predicted.csv")

sns.displot(
    data=songs,
    x="valence", col="mode", kde=True
)

plt.show()