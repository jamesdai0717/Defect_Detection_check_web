import numpy as np
import pandas as pd
import pymongo
import lightgbm as lgb
from self_paced_ensemble import SelfPacedEnsembleClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold


def tune_param_brf(X_train, Y_train):
    parameters = {'max_features': ['auto',0.05, 0.1, 0.2, 0.3, 0.4],
                  'max_depth': [5, 10, 15, 20, 25, 30],
                  'n_estimators': [100, 200, 300, 400, 500]
                  }
    gsearch = GridSearchCV(estimator=BalancedRandomForestClassifier(
        n_estimators=200, max_depth=20, max_features='auto', random_state=12, n_jobs=-1),
        param_grid=parameters, scoring='roc_auc', cv=5, n_jobs=-1)
    gsearch.fit(X_train, Y_train)

    best_parameters = gsearch.best_estimator_.get_params()
    param_brf = {param_name: best_parameters[param_name] for param_name in sorted(parameters.keys())}
    return param_brf
def tune_param_lgbm(X_train, Y_train):
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'nthread': 4,
        'learning_rate': 0.01,
        'num_leaves': 20,
        'max_depth': 20,
        'subsample': 0.8,
        'colsample_bytree': 1
    }
    data_1 = lgb.Dataset(X_train, Y_train)
    cv_results = lgb.cv(
        params, data_1, num_boost_round=1000, nfold=5, stratified=False,
        shuffle=True, metrics='auc', early_stopping_rounds=50, seed=0)
    param_lgbm = dict()
    param_lgbm['n_estimators'] = len(cv_results['auc-mean'])

    parameters = {'max_depth': [10, 20, 30],
                  'num_leaves': [20, 30, 40],
                  'min_data_in_leaf': [1, 5, 10, 20, 30]
                  }
    gsearch = GridSearchCV(estimator=lgb.LGBMClassifier(
        boosting_type='gbdt',
        objective='binary',
        metrics='auc',
        learning_rate=0.01,
        n_estimators=param_lgbm['n_estimators'],
        num_leaves=30,
        max_depth=10,
        bagging_fraction=0.8,
        feature_fraction=0.8
    ),
        param_grid=parameters, scoring='roc_auc', cv=5, n_jobs=-1)
    gsearch.fit(X_train, Y_train)

    best_parameters = gsearch.best_estimator_.get_params()

    for param_name in sorted(parameters.keys()):
        param_lgbm[param_name] = best_parameters[param_name]

    parameters = {'max_depth': [param_lgbm['max_depth']],
                  'num_leaves': [param_lgbm['num_leaves']],
                  'min_data_in_leaf': [param_lgbm['min_data_in_leaf']],
                  'feature_fraction': [0.7, 0.8, 0.9],
                  'bagging_fraction': [0.7, 0.8, 0.9],
                  'lambda_l1': [1e-3, 1e-2, 0.1, 0.0],
                  'lambda_l2': [1e-3, 1e-2, 0.1, 0.0],
                  'learning_rate': [0.001, 0.005, 0.007, 0.01]
                  }
    gsearch = GridSearchCV(estimator=lgb.LGBMClassifier(
        boosting_type='gbdt',
        objective='binary',
        metrics='auc',
        learning_rate=0.01,
        n_estimators=param_lgbm['n_estimators'],
        num_leaves=30,
        max_depth=10,
        bagging_fraction=0.8,
        feature_fraction=0.8
    ),
        param_grid=parameters, scoring='roc_auc', cv=5, n_jobs=-1)
    gsearch.fit(X_train, Y_train)

    best_parameters = gsearch.best_estimator_.get_params()

    for param_name in sorted(parameters.keys()):
        param_lgbm[param_name] = best_parameters[param_name]
    return param_lgbm

def BRF1(X_train, Y_train, X_test, Y_test, BRF_MF, BRF_MD, BRF_NE):
    def same_col(X_train, X_test):
        col_1 = X_train.columns.values.tolist()
        col_2 = X_test.columns.values.tolist()
        if col_1 != col_2:
            n = X_test.shape[0]
            for i in [a for a in col_1 if a not in col_2]:
                X_test[i] = [0] * n
            for i in [a for a in col_2 if a not in col_1]:
                del X_test[i]
        X_test = X_test[col_1]
        return X_train, X_test

    data_index = X_test.index
    X_train, X_test = same_col(X_train, X_test)

    X_train = X_train.values
    Y_train = Y_train.values.ravel()
    X_test = X_test.values
    Y_test = Y_test.values.ravel()

    X = X_train
    Y = Y_train
    proba_all = [0] * np.empty(X.shape[0])
    kfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=12)
    for i, (train_index, test_index) in enumerate(kfold.split(X, Y)):
        x_train = X[train_index]
        y_train = Y[train_index]
        x_test = X[test_index]
        y_test = Y[test_index]

        model = SelfPacedEnsembleClassifier(
            base_estimator=BalancedRandomForestClassifier(
                random_state=12,n_estimators=BRF_NE, max_depth=BRF_MD, max_features=BRF_MF, n_jobs=-1),
            n_estimators=20,
            random_state=12,
            n_jobs=-1
        ).fit(x_train, y_train)

        proba = model.predict_proba(x_test)[:, 1]
        proba_all[test_index] = proba

    def tune_threshold(proba, x):
        y_adjust = [1 if float(y) >= x else 0 for y in proba]
        cof_mat = confusion_matrix(Y, y_adjust)

        def true_negative_rate(mat):
            return mat[1][1] / (mat[1][1] + mat[1][0])

        def true_positive_rate(mat):
            return mat[0][0] / (mat[0][1] + mat[0][0])

        def acc(mat):
            return (mat[0][0] + mat[1][1]) / (mat[0][1] + mat[0][0] + mat[1][0] + mat[1][1])

        TNR = true_negative_rate(cof_mat)
        TPR = true_positive_rate(cof_mat)
        return {'TNR': TNR, 'TPR': TPR}

    threshold = np.arange(0, 0.5, 0.001)
    TNR = [tune_threshold(proba_all, a)['TNR'] for a in threshold]
    TPR = [tune_threshold(proba_all, a)['TPR'] for a in threshold]
    a = pd.DataFrame({'TNR': TNR, 'TPR': TPR})
    b = a.drop_duplicates(subset=['TNR'], keep='last')
    THRESHOLD = b.iloc[1, :].name * 0.001
    BEST_TPR = b.iloc[1, 1]
    return BEST_TPR, THRESHOLD

def BRF2(BEST_TPR,THRESHOLD,X_train, Y_train, X_test, Y_test, BRF_MF, BRF_MD, BRF_NE):
    def same_col(X_train, X_test):
        col_1 = X_train.columns.values.tolist()
        col_2 = X_test.columns.values.tolist()
        if col_1 != col_2:
            n = X_test.shape[0]
            for i in [a for a in col_1 if a not in col_2]:
                X_test[i] = [0] * n
            for i in [a for a in col_2 if a not in col_1]:
                del X_test[i]
        X_test = X_test[col_1]
        return X_train, X_test

    data_index = X_test.index
    X_train, X_test = same_col(X_train, X_test)

    X_train = X_train.values
    Y_train = Y_train.values.ravel()
    X_test = X_test.values
    Y_test = Y_test.values.ravel()

    model = SelfPacedEnsembleClassifier(
        base_estimator=BalancedRandomForestClassifier(
            random_state=12,n_estimators=BRF_NE, max_depth=BRF_MD, max_features=BRF_MF, n_jobs=-1),
        n_estimators=20,
        random_state=12,
        n_jobs=-1
    ).fit(X_train, Y_train)

    proba = model.predict_proba(X_test)[:, 1]

    def adjusted_classes(y_scores, t):
        return [1 if y >= t else 0 for y in y_scores]

    y_adjust_label = adjusted_classes(proba, THRESHOLD)
    cof_mat = confusion_matrix(Y_test, y_adjust_label)

    def true_negative_rate(mat):
        return mat[1][1] / (mat[1][1] + mat[1][0])

    def true_positive_rate(mat):
        return mat[0][0] / (mat[0][1] + mat[0][0])

    def acc(mat):
        return (mat[0][0] + mat[1][1]) / (mat[0][1] + mat[0][0] + mat[1][0] + mat[1][1])

    TNR = true_negative_rate(cof_mat)
    TPR = true_positive_rate(cof_mat)
    ACC = acc(cof_mat)
    AUC = roc_auc_score(Y_test, proba)
    summary = [TNR, TPR, ACC, (TNR + TPR) / 2, AUC]
    proba = pd.DataFrame(proba)
    proba.index = data_index
    return BEST_TPR, THRESHOLD, proba, [round(a, 3) for a in summary]


def LGBM1(X_train, Y_train, X_test, Y_test, LGBM_NE, LGBM_MD, LGBM_NL, LGBM_ML, LGBM_FF, LGBM_BF, LGBM_L1, LGBM_L2,
         LGBM_LR):
    def same_col(X_train, X_test):
        col_1 = X_train.columns.values.tolist()
        col_2 = X_test.columns.values.tolist()
        if col_1 != col_2:
            n = X_test.shape[0]
            for i in [a for a in col_1 if a not in col_2]:
                X_test[i] = [0] * n
            for i in [a for a in col_2 if a not in col_1]:
                del X_test[i]
        X_test = X_test[col_1]
        return X_train, X_test

    data_index = X_test.index
    X_train, X_test = same_col(X_train, X_test)

    X_train = X_train.values
    Y_train = Y_train.values.ravel()
    X_test = X_test.values
    Y_test = Y_test.values.ravel()

    X = X_train
    Y = Y_train
    proba_all = [0] * np.empty(X.shape[0])
    kfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=12)
    for i, (train_index, test_index) in enumerate(kfold.split(X, Y)):
        x_train = X[train_index]
        y_train = Y[train_index]
        x_test = X[test_index]
        y_test = Y[test_index]

        model = SelfPacedEnsembleClassifier(
            base_estimator=lgb.LGBMClassifier(
                boosting_type='gbdt',
                objective='binary',
                metrics='auc',
                learning_rate=LGBM_LR,
                n_estimators=LGBM_NE,
                max_depth=LGBM_MD,
                num_leaves=LGBM_NL,
                bagging_fraction=LGBM_BF,
                feature_fraction=LGBM_FF,
                min_data_in_leaf=LGBM_ML,
                lambda_l1=LGBM_L1,
                lambda_l2=LGBM_L2,
                n_jobs=-1),
            n_estimators=20).fit(x_train, y_train)

        proba = model.predict_proba(x_test)[:, 1]
        proba_all[test_index] = proba

    def tune_threshold(proba, x):
        y_adjust = [1 if float(y) >= x else 0 for y in proba]
        cof_mat = confusion_matrix(Y, y_adjust)

        def true_negative_rate(mat):
            return mat[1][1] / (mat[1][1] + mat[1][0])

        def true_positive_rate(mat):
            return mat[0][0] / (mat[0][1] + mat[0][0])

        TNR = true_negative_rate(cof_mat)
        TPR = true_positive_rate(cof_mat)
        return {'TNR': TNR, 'TPR': TPR}

    threshold = np.arange(0, 0.5, 0.001)
    TNR = [tune_threshold(proba_all, a)['TNR'] for a in threshold]
    TPR = [tune_threshold(proba_all, a)['TPR'] for a in threshold]
    a = pd.DataFrame({'TNR': TNR, 'TPR': TPR})
    b = a.drop_duplicates(subset=['TNR'], keep='last')
    THRESHOLD = b.iloc[1, :].name * 0.001
    BEST_TPR = b.iloc[1, 1]
    return BEST_TPR, THRESHOLD

def LGBM2(BEST_TPR,THRESHOLD,X_train, Y_train, X_test, Y_test, LGBM_NE, LGBM_MD, LGBM_NL, LGBM_ML, LGBM_FF, LGBM_BF, LGBM_L1, LGBM_L2,
          LGBM_LR):
    def same_col(X_train, X_test):
        col_1 = X_train.columns.values.tolist()
        col_2 = X_test.columns.values.tolist()
        if col_1 != col_2:
            n = X_test.shape[0]
            for i in [a for a in col_1 if a not in col_2]:
                X_test[i] = [0] * n
            for i in [a for a in col_2 if a not in col_1]:
                del X_test[i]
        X_test = X_test[col_1]
        return X_train, X_test

    data_index = X_test.index
    X_train, X_test = same_col(X_train, X_test)

    X_train = X_train.values
    Y_train = Y_train.values.ravel()
    X_test = X_test.values
    Y_test = Y_test.values.ravel()

    model = SelfPacedEnsembleClassifier(
        base_estimator=lgb.LGBMClassifier(
            boosting_type='gbdt',
            objective='binary',
            metrics='auc',
            learning_rate=LGBM_LR,
            n_estimators=LGBM_NE,
            max_depth=LGBM_MD,
            num_leaves=LGBM_NL,
            bagging_fraction=LGBM_BF,
            feature_fraction=LGBM_FF,
            min_data_in_leaf=LGBM_ML,
            lambda_l1=LGBM_L1,
            lambda_l2=LGBM_L2,
            n_jobs=-1),
        n_estimators=20).fit(X_train, Y_train)

    proba = model.predict_proba(X_test)[:, 1]

    def adjusted_classes(y_scores, t):
        return [1 if y >= t else 0 for y in y_scores]

    y_adjust_label = adjusted_classes(proba, THRESHOLD)
    cof_mat = confusion_matrix(Y_test, y_adjust_label)

    def true_negative_rate(mat):
        return mat[1][1] / (mat[1][1] + mat[1][0])

    def true_positive_rate(mat):
        return mat[0][0] / (mat[0][1] + mat[0][0])

    def acc(mat):
        return (mat[0][0] + mat[1][1]) / (mat[0][1] + mat[0][0] + mat[1][0] + mat[1][1])

    TNR = true_negative_rate(cof_mat)
    TPR = true_positive_rate(cof_mat)
    ACC = acc(cof_mat)
    AUC = roc_auc_score(Y_test, proba)
    summary = [TNR, TPR, ACC, (TNR + TPR) / 2, AUC]
    proba = pd.DataFrame(proba)
    proba.index = data_index
    return BEST_TPR, THRESHOLD, proba, [round(a, 3) for a in summary]

def IMPORT_MONGO(data, data_name):
    before_list = []
    after_list = []
    CP_key = data.columns.values.tolist()
    for a in range(len(CP_key)):
        before_list.append(CP_key[a])
        if CP_key[a] == 0:
            b = '0'
        else:
            b = CP_key[a].replace('.', ' ')
        after_list.append(b)
    rename_dic = dict(zip(before_list, after_list))
    data.rename(rename_dic, axis=1, inplace=True)
    client = pymongo.MongoClient("mongodb://localhost:37017/", replicaset='wtckhbd',username='administrator',password='P2ssw0rd')
    db = client["james"]
    collection = db[data_name]
    data.reset_index(inplace=True)
    data_dict = data.to_dict("records")
    collection.insert_many(data_dict)
