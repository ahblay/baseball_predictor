import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import os
from models.model import Model
from models.test import Test
from sklearn.feature_selection import chi2

pd.set_option('display.max_columns', 500)
to_drop = ["ht", "at", "Unnamed: 0"]
results = "home_team_won"

clf = DecisionTreeClassifier(criterion="gini")
data = os.path.abspath("./data/big_data.csv")
clf = Model(clf, data)

clf.drop_columns(to_drop)
clf.get_X_y(results)

#tester = Test(clf)
#tester.test_k_best()
#clf.feature_corr()
clf.k_best(chi2, 33)

clf.split_data(0.2, 42)
#clf.standard_scale()
#clf.lda(1)

#clf.rfe(10)

clf.fit_clf()
clf.pred_clf()
ac, cm, cr = clf.eval_clf()

fi = clf.feature_importances()

print(fi)
print(cm)
print(cr)
print(ac)