import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set_theme()

songs = pd.read_csv("./spotify_songs_local.csv")

key_counts = songs["key"].value_counts().sort_index()

key_labels = ["C(0)", "C#(1)", "D(2)", "D#(3)", "E(4)", "F(5)", "F#(6)", "G(7)", "G#(8)", "A(9)", "A#(10)", "B(11)"]


key_mode_counts = (
    songs.groupby(["key", "mode"])
    .size()
    .reset_index(name="count")
)
key_mode_counts["key"] = key_mode_counts["key"].map(dict(enumerate(key_labels)))
key_mode_counts["mode"] = key_mode_counts["mode"].map({0: "Minor", 1: "Major"})

fig, ax = plt.subplots()

sns.barplot(
    data=key_mode_counts,
    x="key", y="count",
    hue="mode",
    palette={"Minor": "steelblue", "Major": "coral"},
    ax=ax
)

for container in ax.containers:
    ax.bar_label(container, fontsize=7)

ax.set_xlabel("Key")
ax.set_ylabel("Count")
ax.set_title("Distribution of Keys by Mode")

plt.show()