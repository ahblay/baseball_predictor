import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from setup_logger import logger
from sklearn.feature_selection import SelectKBest
import seaborn as sns
import matplotlib.pyplot as plt


class Model:
    def __init__(self, clf, csv):
        self.clf = clf
        self.csv = csv
        self.data = self.data = pd.read_csv(self.csv)
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None

    def drop_columns(self, columns):
        self.data = self.data.drop(columns, axis=1)
        logger.info("Dropped columns: " + str(columns))

    def get_X_y(self, results):
        self.X = self.data.drop(results, axis=1)
        self.y = self.data[results]
        logger.info("Split data into X and y")

    def split_data(self, pct, seed=None):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=pct, random_state=seed)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        logger.info("Split data into testing and training sets: " + str(pct) + " testing")

    def standard_scale(self):
        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.X_train)
        self.X_test = sc.transform(self.X_test)
        logger.info("Normalized feature values with sklearn StandardScaler")

    def lda(self, n_components):
        lda = LDA(n_components=n_components)
        self.X_train = lda.fit_transform(self.X_train, self.y_train)
        self.X_test = lda.transform(self.X_test)
        logger.info("Performed feature selection with LDA")

    def k_best(self, func, k, test=False):
        k_best = SelectKBest(score_func=func, k=k)
        fit = k_best.fit(self.X, self.y)

        # Code for displaying selected features
        df_scores = pd.DataFrame(fit.scores_)
        df_columns = pd.DataFrame(self.X.columns)
        feature_scores = pd.concat([df_columns, df_scores], axis=1)
        feature_scores.columns = ['Specs', 'Score']
        logger.info(f"Choosing {str(k)} best features with function {str(func)}...")
        logger.info(feature_scores.nlargest(k, 'Score'))

        if test:
            return feature_scores.nlargest(k, "Score")
        else:
            self.X = k_best.transform(self.X)

    def feature_corr(self):
        corr = self.data.corr()
        top_corr_features = corr.index
        plt.figure(figsize=(10, 7))
        # plot heat map
        sns.heatmap(self.data[top_corr_features].corr(), annot=False, cmap="RdYlGn")
        plt.show()

    def fit_clf(self):
        logger.info("Fitting classifier...")
        self.clf.fit(self.X_train, self.y_train)
        logger.info("Classifier fitted")

    def pred_clf(self):
        self.y_pred = self.clf.predict(self.X_test)
        logger.info("Ran predictions on test data")

    def eval_clf(self):
        ac = accuracy_score(self.y_test, self.y_pred)
        cm = confusion_matrix(self.y_test, self.y_pred)
        cr = classification_report(self.y_test, self.y_pred)
        logger.info("Evaluated classifier performance.")
        return ac, cm, cr