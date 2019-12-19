import pandas as pd
from sklearn.neural_network import MLPClassifier
import os
from models.model import Model
from models.test import Test
from sklearn.feature_selection import chi2, f_classif

pd.set_option('display.max_columns', 500)
to_drop = ["ht", "at", "Unnamed: 0"]
results = "home_team_won"

clf = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
data = os.path.abspath("../data/big_data.csv")
print(data)
clf = Model(clf, data)

clf.drop_columns(to_drop)
clf.get_X_y(results)
clf.to_numerical()

#tester = Test(clf)
#tester.test_k_best()
#clf.feature_corr()
clf.k_best(f_classif, 8)

clf.split_data(0.2)
clf.standard_scale()
clf.lda(1)
clf.fit_clf()
clf.pred_clf()
ac, cm, cr = clf.eval_clf()

print(cm)
print(cr)
print(ac)
