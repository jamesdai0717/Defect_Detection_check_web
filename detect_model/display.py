import pandas as pd
import clean_function
import model_function

#prepare data

##train

dir = "/root/QMSAS_dir/detect_model2/"

train_1 = clean_function.read_data(dir=dir,name='df_train_no_makeup0_20201215_20210530_mes_1420_14_dup.csv',drop_dup=False)
train_2 = clean_function.read_data2(dir=dir,name='df_train_no_makeup0_20210602_20210620_mes_1420_14_dup.csv',drop_dup=False)
train_data = pd.concat([train_1,train_2],axis = 0)

X_TRAIN_A, Y_TRAIN_A = clean_function.clean(train_data,'A',True)#(5107, 812)
X_TRAIN_A, Y_TRAIN_A = clean_function.select_imp(X_TRAIN_A,Y_TRAIN_A,True)

pd.DataFrame(X_TRAIN_A).to_csv(dir + "result/X_TRAIN_A.csv",encoding='utf_8_sig')
pd.DataFrame(Y_TRAIN_A).to_csv(dir + "result/Y_TRAIN_A.csv",encoding='utf_8_sig')

X_TRAIN_B, Y_TRAIN_B = clean_function.clean(train_data,'B',True)
X_TRAIN_B, Y_TRAIN_B = clean_function.select_imp(X_TRAIN_B,Y_TRAIN_B,True)

pd.DataFrame(X_TRAIN_B).to_csv(dir + "result/X_TRAIN_B.csv",encoding='utf_8_sig')
pd.DataFrame(Y_TRAIN_B).to_csv(dir + "result/Y_TRAIN_B.csv",encoding='utf_8_sig')

##test

test_data = clean_function.read_data2(dir=dir,name='df_train_no_makeup0_20210621_20210703_mes_1420.csv',drop_dup=True)

X_TEST_A, Y_TEST_A = clean_function.clean(test_data,'A',False)
X_TEST_A, Y_TEST_A = clean_function.select_imp(X_TEST_A,Y_TEST_A,False)

pd.DataFrame(X_TEST_A).to_csv(dir + "result/X_TEST_A.csv",encoding='utf_8_sig')
pd.DataFrame(Y_TEST_A).to_csv(dir + "result/Y_TEST_A.csv",encoding='utf_8_sig')

X_TEST_B, Y_TEST_B = clean_function.clean(test_data,'B',False)
X_TEST_B, Y_TEST_B = clean_function.select_imp(X_TEST_B,Y_TEST_B,False)

pd.DataFrame(X_TEST_B).to_csv(dir + "result/X_TEST_B.csv",encoding='utf_8_sig')
pd.DataFrame(Y_TEST_B).to_csv(dir + "result/Y_TEST_B.csv",encoding='utf_8_sig')
#
# tune parameter
#
X_TRAIN_A = pd.read_csv(dir + "result/X_TRAIN_A.csv", index_col='ZORDNO_1320')
Y_TRAIN_A = pd.read_csv(dir + "result/Y_TRAIN_A.csv", index_col='ZORDNO_1320')

X_TRAIN_B = pd.read_csv(dir + "result/X_TRAIN_B.csv", index_col='ZORDNO_1320')
Y_TRAIN_B = pd.read_csv(dir + "result/Y_TRAIN_B.csv", index_col='ZORDNO_1320')

param_brf,param_lgbm = model_function.tune_param(X_TRAIN_A.values, Y_TRAIN_A.values.ravel())

param_brf_a = pd.DataFrame(param_brf,index=[0])
param_lgbm_a = pd.DataFrame(param_lgbm,index=[0])

pd.DataFrame(param_brf_a).to_csv(dir + "result/PARAM_BRF_A.csv",encoding='utf_8_sig')
pd.DataFrame(param_lgbm_a).to_csv(dir + "result/PARAM_LGBM_A.csv",encoding='utf_8_sig')

param_brf,param_lgbm = model_function.tune_param(X_TRAIN_B.values, Y_TRAIN_B.values.ravel())

param_brf_b = pd.DataFrame(param_brf,index=[0])
param_lgbm_b = pd.DataFrame(param_lgbm,index=[0])

pd.DataFrame(param_brf_b).to_csv(dir + "result/PARAM_BRF_B.csv",encoding='utf_8_sig')
pd.DataFrame(param_lgbm_b).to_csv(dir + "result/PARAM_LGBM_B.csv",encoding='utf_8_sig')

#choose the best method & data and predict

X_TEST_A = pd.read_csv(dir + "result/X_TEST_A.csv", index_col='ZORDNO_1320')
Y_TEST_A = pd.read_csv(dir + "result/Y_TEST_A.csv", index_col='ZORDNO_1320')

X_TEST_B = pd.read_csv(dir + "result/X_TEST_B.csv", index_col='ZORDNO_1320')
Y_TEST_B = pd.read_csv(dir + "result/Y_TEST_B.csv", index_col='ZORDNO_1320')

param_brf = pd.read_csv(dir + "result/PARAM_BRF_A.csv")
param_lgbm = pd.read_csv(dir + "result/PARAM_LGBM_A.csv")

RESULT_A_BRF_TPR, RESULT_A_BRF_THRESHOLD = model_function.BRF1(X_TRAIN_A, Y_TRAIN_A, X_TEST_A, Y_TEST_A, BRF_RS = 12, BRF_MF = param_brf['max_features'][0], BRF_MD = param_brf['max_depth'][0], BRF_NE = param_brf['n_estimators'][0])
RESULT_A_LGBM_TPR, RESULT_A_LGBM_THRESHOLD = model_function.LGBM1(X_TRAIN_A, Y_TRAIN_A, X_TEST_A, Y_TEST_A, LGBM_NE = param_lgbm['n_estimators'][0], LGBM_MD = param_lgbm['max_depth'][0], LGBM_NL = param_lgbm['num_leaves'][0], LGBM_ML = param_lgbm['min_data_in_leaf'][0], LGBM_FF = param_lgbm['feature_fraction'][0], LGBM_BF = param_lgbm['bagging_fraction'][0], LGBM_L1 = param_lgbm['lambda_l1'][0], LGBM_L2 = param_lgbm['lambda_l2'][0], LGBM_LR = param_lgbm['learning_rate'][0])

param_brf = pd.read_csv(dir + "result/PARAM_BRF_B.csv")
param_lgbm = pd.read_csv(dir + "result/PARAM_LGBM_B.csv")

RESULT_B_BRF_TPR, RESULT_B_BRF_THRESHOLD = model_function.BRF1(X_TRAIN_B, Y_TRAIN_B, X_TEST_B, Y_TEST_B, BRF_RS = 12, BRF_MF = param_brf['max_features'][0], BRF_MD = param_brf['max_depth'][0], BRF_NE = param_brf['n_estimators'][0])
RESULT_B_LGBM_TPR, RESULT_B_LGBM_THRESHOLD = model_function.LGBM1(X_TRAIN_B, Y_TRAIN_B, X_TEST_B, Y_TEST_B, LGBM_NE = param_lgbm['n_estimators'][0], LGBM_MD = param_lgbm['max_depth'][0], LGBM_NL = param_lgbm['num_leaves'][0], LGBM_ML = param_lgbm['min_data_in_leaf'][0], LGBM_FF = param_lgbm['feature_fraction'][0], LGBM_BF = param_lgbm['bagging_fraction'][0], LGBM_L1 = param_lgbm['lambda_l1'][0], LGBM_L2 = param_lgbm['lambda_l2'][0], LGBM_LR = param_lgbm['learning_rate'][0])

if max(RESULT_B_BRF_TPR,RESULT_B_LGBM_TPR) > max(RESULT_A_BRF_TPR,RESULT_A_LGBM_TPR):
    x_train = X_TRAIN_B
    y_train = Y_TRAIN_B
    x_test = X_TEST_B
    y_test = Y_TEST_B
    DATA = 'B'
    if RESULT_B_BRF_TPR > RESULT_B_LGBM_TPR:
        METHOD = 'BRF'
        param_brf = pd.read_csv(dir + "result/PARAM_BRF_B.csv")
        model_function.IMPORT_MONGO(param_brf, 'param_brf')
        RESULT_TPR, RESULT_THRESHOLD, RESULT_PROBA, RESULT = model_function.BRF2(RESULT_B_BRF_TPR, RESULT_B_BRF_THRESHOLD, x_train, y_train, x_test, y_test, BRF_RS = 12, BRF_MF=param_brf['max_features'][0], BRF_MD=param_brf['max_depth'][0], BRF_NE=param_brf['n_estimators'][0])
    else:
        METHOD = 'LGBM'
        param_lgbm = pd.read_csv(dir + "result/PARAM_LGBM_B.csv")
        model_function.IMPORT_MONGO(param_lgbm, 'param_lgbm')
        RESULT_TPR, RESULT_THRESHOLD, RESULT_PROBA, RESULT = model_function.LGBM2(RESULT_B_LGBM_TPR, RESULT_B_LGBM_THRESHOLD, x_train, y_train, x_test, y_test, LGBM_NE=param_lgbm['n_estimators'][0], LGBM_MD=param_lgbm['max_depth'][0], LGBM_NL=param_lgbm['num_leaves'][0], LGBM_ML=param_lgbm['min_data_in_leaf'][0], LGBM_FF=param_lgbm['feature_fraction'][0], LGBM_BF=param_lgbm['bagging_fraction'][0], LGBM_L1=param_lgbm['lambda_l1'][0], LGBM_L2=param_lgbm['lambda_l2'][0], LGBM_LR=param_lgbm['learning_rate'][0])
else:
    x_train = X_TRAIN_A
    y_train = Y_TRAIN_A
    x_test = X_TEST_A
    y_test = Y_TEST_A
    DATA = 'A'
    if RESULT_A_BRF_TPR > RESULT_A_LGBM_TPR:
        METHOD = 'BRF'
        param_brf = pd.read_csv(dir + "result/PARAM_BRF_A.csv")
        model_function.IMPORT_MONGO(param_brf, 'param_brf')
        RESULT_TPR, RESULT_THRESHOLD, RESULT_PROBA, RESULT = model_function.BRF2(RESULT_A_BRF_TPR, RESULT_A_BRF_THRESHOLD, x_train, y_train, x_test, y_test, BRF_RS = 12, BRF_MF=param_brf['max_features'][0], BRF_MD=param_brf['max_depth'][0], BRF_NE=param_brf['n_estimators'][0])
    else:
        METHOD = 'LGBM'
        param_lgbm = pd.read_csv(dir + "result/PARAM_LGBM_A.csv")
        model_function.IMPORT_MONGO(param_lgbm, 'param_lgbm')
        RESULT_TPR, RESULT_THRESHOLD, RESULT_PROBA, RESULT = model_function.LGBM2(RESULT_A_LGBM_TPR, RESULT_A_LGBM_THRESHOLD, x_train, y_train, x_test, y_test, LGBM_NE=param_lgbm['n_estimators'][0], LGBM_MD=param_lgbm['max_depth'][0], LGBM_NL=param_lgbm['num_leaves'][0], LGBM_ML=param_lgbm['min_data_in_leaf'][0], LGBM_FF=param_lgbm['feature_fraction'][0], LGBM_BF=param_lgbm['bagging_fraction'][0], LGBM_L1=param_lgbm['lambda_l1'][0], LGBM_L2=param_lgbm['lambda_l2'][0], LGBM_LR=param_lgbm['learning_rate'][0])

RESULT1 = pd.DataFrame(
    {
     'PROB' : [float(a) for a in RESULT_PROBA.values],
     'RESULT' : [1 if float(x) > RESULT_THRESHOLD else 0 for x in RESULT_PROBA.values],
	'REAL_RESULT':['']*len(RESULT_PROBA.values)
     }
    )
RESULT1.index = RESULT_PROBA.index

RESULT2 = pd.DataFrame(
    {'METHOD' : [METHOD],
     'DATA' : [DATA],
     'THRESHOLD' : [RESULT_THRESHOLD],
     'BEST_TPR' : [RESULT_TPR],
     'TNR' : [RESULT[0]],
     'TPR' : [RESULT[1]],
     'ACC' : [RESULT[2]],
     'WACC' : [RESULT[3]],
     'AUC' : [RESULT[4]]
     }
    )

pd.DataFrame(RESULT1).to_csv(dir + "result/RESULT1.csv",encoding='utf_8_sig')
pd.DataFrame(RESULT2).to_csv(dir + "result/RESULT2.csv",encoding='utf_8_sig')

#mongo

model_function.IMPORT_MONGO(x_train,'x_train')
y_train = y_train.rename(columns={'0':'label'})
model_function.IMPORT_MONGO(y_train,'y_train')

model_function.IMPORT_MONGO(x_test,'x_test')
y_test = y_test.rename(columns={'0':'label'})
model_function.IMPORT_MONGO(y_test,'y_test')

RESULT1 = pd.read_csv(dir + "result/RESULT1.csv", index_col='ZORDNO_1320')
RESULT2 = pd.read_csv(dir + "result/RESULT2.csv")
model_function.IMPORT_MONGO(RESULT1,'result1')
model_function.IMPORT_MONGO(RESULT2,'result2')
