import statsmodels.formula.api as smf
import pandas as pd

songs = pd.read_csv("./spotify_songs_predicted.csv")
model = smf.ols("valence ~ C(key) + C(mode) + C(key):C(mode)", data=songs).fit()

import statsmodels.api as sm
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)