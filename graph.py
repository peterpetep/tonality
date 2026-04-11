import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set_theme()

songs = pd.read_csv("./spotify_songs_predicted.csv")
print(songs.groupby("mode")["arousal"].describe())

from scipy import stats

minor = songs[songs["mode"] == 0]["arousal"]
major = songs[songs["mode"] == 1]["arousal"]

t_stat, p_value = stats.ttest_ind(minor, major)

print(f"t-statistic: {t_stat:.4f}")
print(f"p-value:     {p_value:.4f}")

import pingouin as pg

d = pg.compute_effsize(minor, major, eftype='cohen')
print(f"Cohen's d: {d:.4f}")

fig, ax = plt.subplots()

sns.kdeplot(
    data=songs,
    x="arousal",
    hue="mode",
    common_norm=False,
    fill=True,
    alpha=0.4,
    palette={0: "steelblue", 1: "coral"},
    ax=ax
)

ax.set_xlabel("Arousal")
ax.set_ylabel("Density")
ax.set_title("Arousal Distribution by Mode")

legend = ax.get_legend()
legend.set_title("Mode")
for t, label in zip(legend.get_texts(), ["Minor (0)", "Major (1)"]):
    t.set_text(label)

plt.show()