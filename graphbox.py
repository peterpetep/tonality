import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set_theme()

songs = pd.read_csv("./spotify_songs_predicted.csv")


from statsmodels.stats.multicomp import pairwise_tukeyhsd

key_names = {0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E", 5: "F",
             6: "F#", 7: "G", 8: "G#", 9: "A", 10: "A#", 11: "B"}
mode_names = {0: "minor", 1: "major"}

songs['key_mode'] = (songs['key'].map(key_names) + " " + 
                     songs['mode'].map(mode_names))
tukey = pairwise_tukeyhsd(
    endog=songs['arousal'],
    groups=songs['key_mode'],
    alpha=0.05
)

tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], 
                         columns=tukey._results_table.data[0])
significant = tukey_df[tukey_df['reject'] == True]
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(significant)

wins = significant.groupby('group1')['reject'].count().rename('wins')
losses = significant.groupby('group2')['reject'].count().rename('losses')
leaderboard = pd.concat([wins, losses], axis=1).fillna(0)
leaderboard['net'] = leaderboard['wins'] - leaderboard['losses']
leaderboard.sort_values('net', ascending=False).T.to_csv('arousaltukey.csv')


fig, ax = plt.subplots(figsize=(12, 5))

sns.violinplot(data=songs, x="key", y="arousal", ax=ax,
    hue="mode",
    
    split=True,     
    inner="quart",   
    palette={0: "steelblue", 1: "coral"},
    )

ax.set_xlabel("Key")
ax.set_ylabel("Arousal")
ax.set_title("Arousal Distribution by Key")
plt.show()

