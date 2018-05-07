import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.feature_selection import SelectKBest
from sklearn.neural_network import MLPClassifier
import warnings
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.feature_selection import chi2,SelectFromModel
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import argparse

warnings.filterwarnings("ignore")

train_data_path = ""
test_data_path = ""

class LabelBinarizer2:

    def __init__(self):
        self.lb = LabelBinarizer()

    def fit(self, X):
        # Convert X to array
        X = np.array(X)
        # Fit X using the LabelBinarizer object
        self.lb.fit(X)
        # Save the classes
        self.classes_ = self.lb.classes_

    def fit_transform(self, X):
        # Convert X to array
        X = np.array(X)
        # Fit + transform X using the LabelBinarizer object
        Xlb = self.lb.fit_transform(X)
        # Save the classes
        self.classes_ = self.lb.classes_
        if len(self.classes_) == 2:
            Xlb = np.hstack((Xlb, 1 - Xlb))
        return Xlb

    def transform(self, X):
        # Convert X to array
        X = np.array(X)
        # Transform X using the LabelBinarizer object
        Xlb = self.lb.transform(X)
        if len(self.classes_) == 2:
            Xlb = np.hstack((Xlb, 1 - Xlb))
        return Xlb

    def inverse_transform(self, Xlb):
        # Convert Xlb to array
        Xlb = np.array(Xlb)
        if len(self.classes_) == 2:
            X = self.lb.inverse_transform(Xlb[:, 0])
        else:
            X = self.lb.inverse_transform(Xlb)
        return X


def perform_test(model):
    test_df = pd.read_csv(test_data_path)
    test_df2 = test_df[test_df.columns[test_df.dtypes != "object"]].apply(lambda x: round(x.fillna(x.mean())), axis=0)
    test_df3 = test_df[test_df.columns[test_df.dtypes == "object"]].apply(lambda x: (x.fillna(x.mode()[0])), axis=0)
    dfs = [test_df2, test_df3]
    test_df = pd.concat(dfs, axis=1)
    for i in test_df.columns[test_df.dtypes == "object"]:

        lb_style = LabelBinarizer2()
        lb_results = lb_style.fit_transform(test_df[i].astype(str))
        column = lb_style.classes_.tolist()
        df = pd.DataFrame(data=lb_results, columns=column)
        test_df = pd.merge(test_df, df, left_index=True, right_index=True)
        del test_df[test_df[i].name]

    predictions = model.predict(test_df)

    result_df = pd.DataFrame()
    result_df["Spending On Healthy Eating"] = predictions
    result_df.to_csv('predictions.csv', encoding='utf-8', index=False)
    print("Predictions are generated with the file name predictions.csv")

def train():

    young = pd.read_csv(train_data_path)
    young.head()
    nulls = young.isnull().sum().sort_values(ascending=False)
    young2=young[young.columns[young.dtypes!="object"]].apply(lambda x: round(x.fillna(x.mean())),axis=0)
    young3=young[young.columns[young.dtypes=="object"]].apply(lambda x: (x.fillna(x.mode()[0])),axis=0)
    dfs = [young2,young3]
    result = pd.concat( dfs,axis=1)
    result.head(25)
    young = result
    for i in young.columns[young.dtypes == "object"]:
        lb_style = LabelBinarizer2()
        lb_results = lb_style.fit_transform(young[i].astype(str))
        column = lb_style.classes_.tolist()
        df = pd.DataFrame(data=lb_results, columns=column)
        young = pd.merge(young, df, left_index=True, right_index=True)
        del young[young[i].name]

    young_Y=young["Spending on healthy eating"]
    young_X=young.drop("Spending on healthy eating",axis=1)

    scaler = StandardScaler()

    nf = 100

    MLP=MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
    svm=LinearSVC()

    maximum = 0.0
    features = []
    features.append(('select_best', SelectKBest(chi2,k=10)))
    feature_union = FeatureUnion(features)

    estimators = []
    estimators.append(('feature_union', feature_union))

    estimators.append(('rf_classifier' , RandomForestClassifier(criterion='gini', max_depth = 11, random_state=0)))
    model = Pipeline(estimators)
    model = model.fit(young_X,young_Y)

    seed = 7
    kfold = KFold(n_splits=10, random_state=seed)
    from sklearn.model_selection import cross_val_predict
    predicted = cross_val_predict(model, young_X, young_Y, cv=kfold)
    print("Accuracy: {0}%".format(metrics.accuracy_score(young_Y, predicted)*100))
    results = cross_val_score(model, young_X, young_Y, cv=kfold)
    #print(results)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-trn', '--Train', help='Provide the path to the training data set on which the model will be trained',required=True)
    parser.add_argument('-tst', '--Test', help='Provide the path to the Test data set on which the model will perform predictions',required=True)

    args = vars(parser.parse_args())

    train_data_path = args['Train']
    test_data_path = args['Test']

    model = train()
    perform_test(model)