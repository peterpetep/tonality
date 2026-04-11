import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set_theme()

songs = pd.read_csv("./spotify_songs_local.csv")

fig, ax = plt.subplots()

key_counts = songs["key"].value_counts().sort_index()

key_labels = ["C(0)", "C#(1)", "D(2)", "D#(3)", "E(4)", "F(5)", "F#(6)", "G(7)", "G#(8)", "A(9)", "A#(10)", "B(11)"]

bars = sns.barplot(x=key_labels, y=key_counts.values, color="steelblue", ax=ax)

ax.bar_label(ax.containers[0], fontsize=9)

ax.set_xlabel("Key")
ax.set_ylabel("Count")
ax.set_title("Distribution of Keys")

plt.show()