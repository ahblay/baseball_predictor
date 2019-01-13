import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from models.model import Model
import os

pd.set_option('display.max_columns', 500)
to_drop = ["ht", "at", "Unnamed: 0"]
results = "home_team_won"

clf = RandomForestClassifier(max_depth=2, random_state=0)
data = os.path.abspath("./data/master_data.csv")
clf = Model(clf, data)

clf.drop_columns(to_drop)
clf.split_data(results, 0.20)
clf.standard_scale()
clf.lda(1)
clf.fit_clf()
clf.pred_clf()
cm, cr = clf.eval_clf()

print(cm)
print(cr)
