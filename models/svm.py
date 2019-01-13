import pandas as pd
from sklearn import svm
import os
from models.model import Model

pd.set_option('display.max_columns', 500)
to_drop = ["ht", "at", "Unnamed: 0"]
results = "home_team_won"

clf = svm.SVC(kernel='rbf')
data = os.path.abspath("./data/master_data.csv")
svm_clf = Model(clf, data)

svm_clf.drop_columns(to_drop)
svm_clf.split_data(results, 0.20)
svm_clf.standard_scale()
svm_clf.lda(1)
svm_clf.fit_clf()
svm_clf.pred_clf()
cm, cr = svm_clf.eval_clf()

print(cm)
print(cr)


