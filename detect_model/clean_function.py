import pandas as pd
import numpy as np
import datetime
import lightgbm as lgb
from functools import partial
from sklearn.impute import KNNImputer
from sklearn.model_selection import cross_validate
from imblearn.ensemble import BalancedRandomForestClassifier


def clean(data,inter_type,train):
    
    def split_zordpn(data):
        data1 = data.copy()
        for i in range(data.shape[0]):
            if len(list(data['ZORDPN'])[i]) < 12:
                data1.drop(data.index[i], inplace = True)
        data1['ZORDPN_s'] = data1.ZORDPN.apply(lambda x : x[:4])
        data1['ZORDPN_d'] = data1.ZORDPN.apply(lambda x : x[4])
        data1['ZORDPN_c'] = data1.ZORDPN.apply(lambda x : x[5:8])
        data1['ZORDPN_t'] = data1.ZORDPN.apply(lambda x : x[8])
        data1['ZORDPN_v'] = data1.ZORDPN.apply(lambda x : x[9:12])
        return data1
    
    def deal_with_date(data):  
        to_datetime_fmt = partial(pd.to_datetime, format='%m/%d')
        for j in ['1210_1620_ZI009_燒附日期','1210_1620_ZI019_沾附日期']:
            for i in range(data.shape[0]):
                if type(data[j].values.tolist()[i]) != str or type(data[j].values.tolist()[i]) == float:
                    data[j].values[i] = np.nan
                else:
                    temp = data[j].values.tolist()[i].find('2021')
                    if temp == -1:
                        def validate(date_text):        
                            try:
                                date_text = date_text.replace('+','')
                                date_text = date_text.replace(' ','')
                                month,day = date_text.split('/')
                                if int(day) <= 31 and int(day) > 0:
                                    datetime.datetime(int(2021),int(month),int(day))
                                    return True
                                else:
                                    return False
                            except ValueError:
                                return False
                        if not validate(data[j].values.tolist()[i]):
                            data[j].values[i] = np.nan 
                        else :
                            data[j].values[i] = data[j].values[i].replace('+','')
                            data[j].values[i] = data[j].values[i].replace(' ','')
                    else:
                        if data[j].values.tolist()[i][:4] != '2021':
                            data[j].values[i] = np.nan
                        else :
                            data[j].values[i] = str(data[j].values[i][5:10]).replace('-','/')
            data[j] = data[j].apply(to_datetime_fmt)
        return data
    
    def deal_with_d50(data):
        def check(v):
            try: k = float(v)
            except:  return False
            return True
        for j in ['1PASS_D50','2PASS_D50','3PASS_D50','4PASS_D50','5PASS_D50','6PASS_D50','7PASS_D50']:
            for i in range(data.shape[0]):
                if not check(data[j].values[i]):
                    data[j].values[i] = np.nan
                elif float(data[j].values[i]) > 1:
                    data[j].values[i] = np.nan
        for i in range(data.shape[0]):
            if data['5PASS_D50'].values[i] == 0:
                data['5PASS_D50'].values[i] = data['4PASS_D50'].values[i]
        for i in range(data.shape[0]):
            if data['6PASS_D50'].values[i] == 0:
                data['6PASS_D50'].values[i] = data['5PASS_D50'].values[i]
        for i in range(data.shape[0]):
            if data['7PASS_D50'].values[i] == 0:
                data['7PASS_D50'].values[i] = data['6PASS_D50'].values[i]
        return data
    
    def deal_with_milling(data):
        l_time = ['Milling start time研磨開始時間','Milling finish time研磨結束時間','Dopant start milling time微量分散開始時間','Dopant finish milling time微量分散結束時間','1Pass Milling Blade Time','2Pass Milling Blade Time','3Pass Milling Blade Time','4Pass Milling Blade Time','5Pass Milling Blade Time','6Pass Milling Blade Time','7Pass Milling Blade Time','8Pass Milling Blade Time','9Pass Milling Blade Time','10Pass Milling Blade Time','11Pass Milling Blade Time','12Pass Milling Blade Time','13Pass Milling Blade Time','14Pass Milling Blade Time']
        def check(v):
            try: k = int(float(v))
            except:  return False
            return True
        for j in l_time:
            data[j] = data[j].replace('\r\n','', regex=True)
            for i in range(data.shape[0]):
                if not check(data[j].values[i]) :
                    data[j].values[i] = np.nan
                elif float(data[j].values[i]) < 1 and float(data[j].values[i]) != 0:
                    data[j].values[i] = np.nan
                elif len(str(int(float(data[j].values[i])))) > 4:
                    data[j].values[i] = np.nan
        for j in range(9,18):
            for i in range(data.shape[0]):
                if data[l_time[j]].values[i] == 0:
                    data[l_time[j]].values[i] = data[l_time[j-1]].values[i]
        return data
    
    def deal_with_1220(data):
        def check(v):
            try: k = float(v)
            except:  return False
            return True
        for j in ['1210_1220_ZI022_首件墨1+2(30m laydown 1+2)','1210_1220_ZI023_首件墨3+4(30m laydown 3+4)','1210_1220_ZI024_1000m墨1+2(1000m laydown 1+2)','1210_1220_ZI025_1000m墨3+4(1000m laydown 3+4)']:
            for i in range(data.shape[0]):
                if not check(data[j].values[i]) :
                    data[j].values[i] = np.nan 
                elif float(data[j].values[i]) > 1:
                    data[j].values[i] = np.nan 
                else :
                    data[j].values[i] = float(data[j].values[i])
            data[j] = pd.to_numeric(data[j])
        for j in ['1210_1220_ZI026_尺寸長1(pattern length1)','1210_1220_ZI027_尺寸長2(pattern length2)','1210_1220_ZI028_尺寸長3(pattern length2)','1210_1220_ZI029_尺寸長4(pattern length3)','1210_1220_ZI030_尺寸寬1(pattern wide1)','1210_1220_ZI031_尺寸寬2(pattern wide2)','1210_1220_ZI032_尺寸寬3(pattern wide3)','1210_1220_ZI033_尺寸寬4(pattern wide4)']:
            for i in range(data.shape[0]):
                if not check(data[j].values[i]) :
                    data[j].values[i] = np.nan
                elif float(data[j].values[i]) > 10:
                    data[j].values[i] = np.nan
                else :
                    data[j].values[i] = float(data[j].values[i])
            data[j] = pd.to_numeric(data[j])
        return data
    
    def deal_with_last(data):
        def check(v):
            try: k = float(v)
            except:  return False
            return True
        for j in ['1210_1420_ZI018_R值#1','1210_1420_ZI019_R值#2','1210_1420_ZI020_R值#3','1210_1420_ZI021_R值#4','1210_1420_ZI022_R值#5']:
            for i in range(data.shape[0]):
                if not check(data[j].values[i]) :
                    data[j].values[i] = np.nan 
                elif float(data[j].values[i]) > 1 and float(data[j].values[i]) not in [53,54,55]:
                    data[j].values[i] = np.nan 
                else :
                    data[j].values[i] = float(data[j].values[i])
            data[j] = pd.to_numeric(data[j]) 
        j = '1210_1410_ZI006_setter或mesh 片數'
        for i in range(data.shape[0]):
            if not check(data[j].values[i]):
                data[j].values[i] = np.nan 
            elif float(data[j].values[i]) > 1000:
                data[j].values[i] = np.nan 
            else :
                data[j].values[i] = float(data[j].values[i])
        data[j] = pd.to_numeric(data[j])
        return(data)
    
    def change_type(data):
    
        def check(v):
            try: k = float(v)
            except:  return False
            return True

        #dir = "/root/QMSAS_dir/detect_model/prepare/"
        dir = "C:/Users/jamesdai/Desktop/james/0729/detect_model/prepare/"
        target_type = pd.read_csv(dir + "target_type.csv", index_col=0)
        target_type = target_type['0'].tolist()
    
        l1 = []
        j = '1210_1320_BT004_ 內層疊層層數(number of layer)'
        l1.append(j)
        for i in range(data.shape[0]):
            if not check(data[j].values[i]) :
                data[j].values[i] = np.nan  
            elif float(data[j].values[i]) > 1000:
                data[j].values[i] = np.nan 
            else :
                data[j].values[i] = float(data[j].values[i])
        data[j] = pd.to_numeric(data[j])
    
        j = '1210_1410_ZI007_燒失率 %'
        l1.append(j)    
        for i in range(data.shape[0]):
            if not check(data[j].values[i]) :
                data[j].values[i] = np.nan  
            elif float(data[j].values[i]) > 100:
                data[j].values[i] = np.nan 
            else :
                data[j].values[i] = float(data[j].values[i])
        data[j] = pd.to_numeric(data[j])
    
        for j in ['1210_1720_ZI022_IR1電壓(IR1 voltage)','1210_1720_ZI025_IR2電壓(IR2 voltage)']:
            l1.append(j)
            for i in range(data.shape[0]):
                if not check(data[j].values[i]) :
                    if data[j].values[i].find('V') != -1:
                        data[j].values[i] = data[j].values[i][data[j].values[i].find('V')] 
                    if data[j].values[i].find('n') != -1:
                        data[j].values[i] = data[j].values[i][data[j].values[i].find('V')]
                    data[j].values[i] = np.nan
                else :
                    data[j].values[i] = float(data[j].values[i])
            data[j] = pd.to_numeric(data[j])  
    
        for j in ['1210_1720_ZI024_IR1時間(IR1 soak time(ms))','1210_1720_ZI027_IR2時間(IR2 soak time(ms))']:
            l1.append(j)
            for i in range(data.shape[0]):
                if not check(data[j].values[i]) :
                    if data[j].values[i].find('MS') != -1:
                        data[j].values[i] = data[j].values[i][data[j].values[i].find('MS')]            
                    data[j].values[i] = np.nan
                elif float(data[j].values[i]) < 0:
                    data[j].values[i] = np.nan 
                else :
                    data[j].values[i] = float(data[j].values[i])
            data[j] = pd.to_numeric(data[j])
    
        for j in ['Dopant start milling time微量分散開始時間','1Pass Milling Blade Time']:
            l1.append(j)
            data[j] = data[j].replace('\r\n','', regex=True)
            data[j] = pd.to_numeric(data[j])
    
        data_type = data.columns[data.dtypes == float]
        temp = [a for a in target_type if a not in data_type.values.tolist()]
        temp.remove('1210_1810_ZI067_HALT確認異常(24hr&48hr)')
    
        for j in temp:
            for i in range(data.shape[0]):
                if not check(data[j].values[i]) :
                    data[j].values[i] = np.nan  
                else :
                    data[j].values[i] = float(data[j].values[i])
            data[j] = pd.to_numeric(data[j])
        return data
    
    def replace_same(data):
        data['1210_1210_ZI001_PET FILM種類(PET type)'].replace('X2DY-25', 'X2D-25',inplace = True)
        for i in ['WEET7','WET 7','wet-7','WET7.','WET7+','WEWT7']:
            data['1210_1410_ZI002_BBO 程式'].replace(i,'WET7',inplace = True)
        data['1210_1410_ZI002_BBO 程式'].replace('wet-9','WET9',inplace = True)
        data['1210_1410_ZI002_BBO 程式'].replace('WET 2','WET2',inplace = True)
        data['1210_1410_ZI002_BBO 程式'].replace('WET 3','WET3',inplace = True)
        data['1210_1620_ZI001_銅膏類別'].replace('C-4199','C4199',inplace = True)
        data['1210_1620_ZI011_銅導角製程'].replace('有(YES)','YES',inplace = True)
        data['1210_1620_ZI011_銅導角製程'].replace('無(NO)','NO',inplace = True)
        data['1210_1620_ZI015_外觀檢驗'].replace('合格','OK',inplace = True)
        data['1210_1210_ZI002_粉末種類(Powder type)'].replace('.0','',inplace = True)
        return data
    
    data.dropna(axis=0, how='any', thresh=None, subset=['1PASS_D50','2PASS_D50','3PASS_D50'], inplace=True)
    
    data = split_zordpn(data)

    #dir = "/root/QMSAS_dir/detect_model/prepare/"
    dir = "C:/Users/jamesdai/Desktop/james/0729/detect_model/prepare/"
    col_name = pd.read_csv(dir + "col_name.csv")['0'].values.tolist()
    col_name = col_name + ['8Pass Milling Blade Time','9Pass Milling Blade Time','10Pass Milling Blade Time','11Pass Milling Blade Time','12Pass Milling Blade Time','13Pass Milling Blade Time','14Pass Milling Blade Time']
    data = data[col_name]
    
    data = deal_with_date(data)
    data = deal_with_d50(data)
    data = deal_with_milling(data)
    data = deal_with_1220(data)
    data = deal_with_last(data)
    data = change_type(data)
    data = replace_same(data)
        
    y_list = list()
    for column in data.columns:
        if any(item in column for item in ['1210_1810']):
            y_list.append(column)
    x_list = [a for a in data.columns if a not in y_list]
    x_list.remove('FQC')
    X = data[x_list]
    
    y_list.remove('1210_1810_ZI001_首驗/重工記錄')
    Y = data[y_list]
    f = open(dir + "category_features.txt",encoding="utf-8")
    categorical_features = []
    for line in f:
        categorical_features.append(line.strip())
    f.close()    
    
    categorical_features = categorical_features + ['ZORDPN_s','ZORDPN_d','ZORDPN_c','ZORDPN_t','ZORDPN_v'] 
    categorical_features = categorical_features + ['1210_1330_ZI005_第一次真空程式 ','1210_1330_ZI007_第一次均壓程式 ','1210_1330_ZI011_第二次均壓程式 '] 
    categorical_features = categorical_features + ['1210_1220_ZI046_渲染(Rendering)']
    
    categorical_features_current = [item for item in X.columns.tolist() if item in categorical_features]    
    numeric_features_current = [item for item in X.columns.tolist() if item not in categorical_features]
    
    if train == True:
        n = X.shape[0]/150
    else:
        n = 5        
    
    for col in categorical_features_current:
        counts = X[col].value_counts()
        counts1 = counts[counts <= round(n)]
        if len(counts1) > 0 :
            repl = counts1.index
            dummy = pd.get_dummies(X[col].replace(repl,'un_common'))
            dummy.pop('un_common')
            dummy.columns = [str(col) + '__' + str(dum) for dum in dummy.columns]
    
        else:
            dummy = pd.get_dummies(X[col])
            dummy.columns = [str(col) + '__' + str(dum) for dum in dummy.columns]
        X = pd.concat([X,dummy],axis = 1)
        X.pop(col) 
    
    X['濾心壓力-中間(Filter Kpa-middle)-開始(Filter Kpa-start)'] = X['1210_1210_ZI015_濾心壓力-中間(Filter Kpa-middle)'] - X['1210_1210_ZI014_濾心壓力-開始(Filter Kpa-start)']
    X['機台壓力-中間(Machine pressure-Middle)-開始(Machine pressure-start)'] = X['1210_1210_ZI017_機台壓力-中間(Machine pressure-Middle)']-X['1210_1210_ZI016_機台壓力-開始(Machine pressure-start)']
    X['重量-中間左(Weight-middle-L)-開機左(Weight-start-L)'] = X['1210_1210_ZI021_重量-中間左(Weight-middle-L)'] - X['1210_1210_ZI019_重量-開機左(Weight-start-L)']
    X['重量-中間右(Weight-middle-R)-開機右(Weight-start-R)'] = X['1210_1210_ZI022_重量-中間右(Weight-middle-R)'] - X['1210_1210_ZI020_重量-開機右(Weight-start-R)']
    
    X['首件墨1+2(30m laydown 1+2)-3+4(30m laydown 3+4)'] = X['1210_1220_ZI022_首件墨1+2(30m laydown 1+2)'] - X['1210_1220_ZI023_首件墨3+4(30m laydown 3+4)']
    X['1000m墨1+2(1000m laydown 1+2)-3+4(1000m laydown 3+4)'] = X['1210_1220_ZI024_1000m墨1+2(1000m laydown 1+2)'] - X['1210_1220_ZI025_1000m墨3+4(1000m laydown 3+4)']   
    X['max_首件墨'] = X[['1210_1220_ZI022_首件墨1+2(30m laydown 1+2)','1210_1220_ZI023_首件墨3+4(30m laydown 3+4)']].max(axis=1)
    X['min_首件墨'] = X[['1210_1220_ZI022_首件墨1+2(30m laydown 1+2)','1210_1220_ZI023_首件墨3+4(30m laydown 3+4)']].min(axis=1)
    X['diff_首件墨'] = X['max_首件墨'] - X['min_首件墨']    
    X['max_1000m墨'] = X[['1210_1220_ZI024_1000m墨1+2(1000m laydown 1+2)','1210_1220_ZI025_1000m墨3+4(1000m laydown 3+4)']].max(axis=1)
    X['min_1000m墨'] = X[['1210_1220_ZI024_1000m墨1+2(1000m laydown 1+2)','1210_1220_ZI025_1000m墨3+4(1000m laydown 3+4)']].min(axis=1)
    X['diff_1000m墨'] = X['max_1000m墨'] - X['min_1000m墨']    
    X['max_墨'] = X[['max_首件墨','max_1000m墨']].max(axis=1)
    X['min_墨'] = X[['min_首件墨','min_1000m墨']].min(axis=1)
    X['diff_墨'] = X['max_墨'] - X['min_墨']
    
    length = [   '1210_1220_ZI026_尺寸長1(pattern length1)',
                 '1210_1220_ZI027_尺寸長2(pattern length2)',
                 '1210_1220_ZI028_尺寸長3(pattern length2)',
                 '1210_1220_ZI029_尺寸長4(pattern length3)']
    wide = [   '1210_1220_ZI030_尺寸寬1(pattern wide1)',
               '1210_1220_ZI031_尺寸寬2(pattern wide2)',
               '1210_1220_ZI032_尺寸寬3(pattern wide3)',
               '1210_1220_ZI033_尺寸寬4(pattern wide4)']
    for i in [0,1,2,3]:
        name = '面積' + str(i+1)
        X[name] = X[length[i]]*X[wide[i]]
    area = ['面積1','面積2','面積3','面積4']
    X['平均_面積'] = (X['面積1'] + X['面積2'] + X['面積3'] + X['面積4'])/4
    X['平均_長'] = (X[length[0]] + X[length[1]] + X[length[2]] + X[length[3]])/4
    X['平均_寬'] = (X[wide[0]] + X[wide[1]] + X[wide[2]] + X[wide[3]])/4
    X['max_長'] = X[length].max(axis=1)
    X['min_長'] = X[length].min(axis=1)
    X['diff_長'] = X['max_長'] - X['min_長']
    X['max_寬'] = X[wide].max(axis=1)
    X['min_寬'] = X[wide].min(axis=1)
    X['diff_寬'] = X['max_寬'] - X['min_寬']  
    X['max_面積'] = X[area].max(axis=1)
    X['min_面積'] = X[area].min(axis=1)
    X['diff_面積'] = X['max_面積'] - X['min_面積']
    X['標準差_面積'] = X[area].std(axis = 1)
    X['標準差_長'] = X[length].std(axis = 1)
    X['標準差_寬'] = X[wide].std(axis = 1)
    X['每片有效晶粒數'] = X['1210_1410_ZI005_有效晶粒數']/X['1210_1410_ZI006_setter或mesh 片數']
    R = ['1210_1420_ZI018_R值#1','1210_1420_ZI019_R值#2','1210_1420_ZI020_R值#3','1210_1420_ZI021_R值#4','1210_1420_ZI022_R值#5']
    X['平均_R值'] = (X[R[0]] + X[R[1]] + X[R[2]] + X[R[3]])/4
    X['標準差_R值'] = X[R].std(axis = 1)
    X['max_R值'] = X[R].max(axis=1)
    X['min_R值'] = X[R].min(axis=1)
    X['diff_R值'] = X['max_R值'] - X['min_R值']
    X['燒附日期-沾附日期'] = (data['1210_1620_ZI009_燒附日期']- data['1210_1620_ZI019_沾附日期'])/np.timedelta64(1, 'D')
    X.loc[X['燒附日期-沾附日期'] < -300, '燒附日期-沾附日期'] += 365
    X.loc[X['燒附日期-沾附日期'] < 0, '燒附日期-沾附日期'] = np.nan
    X['SC設定值-中間(SC-middle)-開始(SC-start)'] = X['1210_1210_ZI011_SC設定值-中間(SC-middle)'] - X['1210_1210_ZI010_SC設定值-開始(SC-start)']
    X['Pump轉速-中間(Pump-middle)-開始(Pump-start)'] = X['1210_1210_ZI013_Pump轉速-中間(Pump-middle)'] - X['1210_1210_ZI012_Pump轉速-開始(Pump-start)']
    X['R值-中間(R value-middle)-開機(R value-start)'] = X['1210_1210_ZI024_R值-中間(R value-middle)'] - X['1210_1210_ZI023_R值-開機(R value-start)']
    d = ['1210_1210_ZI026_濃度-1',
         '1210_1210_ZI027_濃度-2',
         '1210_1210_ZI028_濃度-3',
         '1210_1210_ZI029_濃度-4',
         '1210_1210_ZI030_濃度-5'
        ]
    X['平均_濃度值'] = (X[d[0]] + X[d[1]] + X[d[2]] + X[d[3]] + X[d[4]])/5
    X['標準差_濃度值'] = X[d].std(axis = 1)
    X['max_濃度值'] = X[d].max(axis=1)
    X['min_濃度值'] = X[d].min(axis=1)
    X['diff_濃度值'] = X['max_濃度值'] - X['min_濃度值']
    X['前段張力左右相減'] = X['1210_1220_ZI005_前段張力L(Left tension at front) ']-X['1210_1220_ZI006_前段張力R(Right tension at front)']
    X['後段張力左右相減'] = X['1210_1220_ZI007_後段張力L(Left tension at back)']-X['1210_1220_ZI008_後段張力R(Right tension at back)']           
    X['左段張力前後相減'] = X['1210_1220_ZI005_前段張力L(Left tension at front) ']-X['1210_1220_ZI007_後段張力L(Left tension at back)']
    X['右段張力前後相減'] = X['1210_1220_ZI006_前段張力R(Right tension at front)']-X['1210_1220_ZI008_後段張力R(Right tension at back)'] 
    force = ['前段張力左右相減','後段張力左右相減','左段張力前後相減','右段張力前後相減'] 
    X['max_張力'] = X[force].max(axis=1)
    X['min_張力'] = X[force].min(axis=1)
    X['diff_張力'] = X['max_張力'] - X['min_張力']                  
    new = ['濾心壓力-中間(Filter Kpa-middle)-開始(Filter Kpa-start)',
           '機台壓力-中間(Machine pressure-Middle)-開始(Machine pressure-start)',
           '重量-中間左(Weight-middle-L)-開機左(Weight-start-L)',
           '重量-中間右(Weight-middle-R)-開機右(Weight-start-R)',
           '首件墨1+2(30m laydown 1+2)-3+4(30m laydown 3+4)',
           '1000m墨1+2(1000m laydown 1+2)-3+4(1000m laydown 3+4)',
           'max_首件墨','min_首件墨','diff_首件墨',
           'max_1000m墨','min_1000m墨','diff_1000m墨',
           'max_墨','min_墨','diff_墨',
           'max_長','min_長','diff_長',
           'max_寬','min_寬','diff_寬',
           'max_面積','min_面積','diff_面積',
           '面積1','面積2','面積3','面積4',
           '平均_面積','平均_長','平均_寬',
           '標準差_面積','標準差_長','標準差_寬',
           '每片有效晶粒數','平均_R值','標準差_R值',
           'max_R值','min_R值','diff_R值',
           '燒附日期-沾附日期',
           'SC設定值-中間(SC-middle)-開始(SC-start)',
           'Pump轉速-中間(Pump-middle)-開始(Pump-start)',
           'R值-中間(R value-middle)-開機(R value-start)',
           '平均_濃度值','標準差_濃度值',
           'max_濃度值','min_濃度值','diff_濃度值',
           '前段張力左右相減','後段張力左右相減',
           '左段張力前後相減','右段張力前後相減',
           'max_張力','min_張力','diff_張力'
          ]
    numeric_features_current = numeric_features_current + new
    time_list1 = ["Dopant start milling time微量分散開始時間"
                    ,"Dopant finish milling time微量分散結束時間"
                    ,"Milling start time研磨開始時間"                       
                    ,"1Pass Milling Blade Time"
                    ,"2Pass Milling Blade Time"
                    ,"3Pass Milling Blade Time"
                    ,"4Pass Milling Blade Time"
                    ,"5Pass Milling Blade Time"
                    ,"6Pass Milling Blade Time"
                    ,"7Pass Milling Blade Time"
                    ,"8Pass Milling Blade Time"
                    ,"9Pass Milling Blade Time"
                    ,"10Pass Milling Blade Time"
                    ,"11Pass Milling Blade Time"
                    ,"12Pass Milling Blade Time"
                    ,"13Pass Milling Blade Time"
                    ,"14Pass Milling Blade Time"
                    ,"Milling finish time研磨結束時間"] 
    
    def minute(x):
        def check(v):
            try: k = int(float(v))
            except:  return False
            return True
        if check(x):
            x = str(int(float(x)))
            if len(x) == 4:
                x_hour = int(x[:2])
                x_minute = int(x[2:])        
            elif len(x) == 3:
                x_hour = int(x[:1])
                x_minute = int(x[1:])
            else:
                x_hour = 0
                x_minute = int(x)
            a = x_hour*60 + x_minute
        else:
            a = np.nan
        return a
    
    for col in time_list1:
        X[col] = X[col].apply(minute)
    
    time_interval_list = list()
    for i in range(0,16): 
        col_name = str(time_list1[i+1]) + '-' + str(time_list1[i])
        time_interval_list.append(col_name)
        X[col_name] = X[time_list1[i+1]] - X[time_list1[i]]
        X.loc[X[col_name] < 0, col_name] += 60*24
        #X[col_name][X[col_name] < 0] = X[col_name][X[col_name] < 0] + 60*24
    
    col_name = 'Milling finish time研磨結束時間-Dopant start milling time微量分散開始時間'
    X[col_name] = X[time_interval_list].sum(axis=1)
    
    for i in range(0,16): 
        col_name = str(time_interval_list[i]) + ' / Milling finish time研磨結束時間-Dopant start milling time微量分散開始時間'
        X[col_name] = X[time_interval_list[i]] / X['Milling finish time研磨結束時間-Dopant start milling time微量分散開始時間']
    
    for col in time_list1:
        del X[col]
    
    for item in time_list1:
        numeric_features_current.remove(item)
    
    about_time = list()
    for column in X.columns:
        if any(item in column for item in ['illing']):
            about_time.append(column)
    
    numeric_features_current = numeric_features_current + about_time
    
    slurry_list1 = ['slurry_D10_TARGET_VALUE',
                    'slurry_D50_TARGET_VALUE',
                    'slurry_D90_TARGET_VALUE',
                    'slurry_D99_TARGET_VALUE']
    slurry_list2 = ['slurry_D10_ORG_TARGET_VALUE',
                    'slurry_D50_ORG_TARGET_VALUE',
                    'slurry_D90_ORG_TARGET_VALUE',
                    'slurry_D99_ORG_TARGET_VALUE']
    for i in range(0,4): 
        col_name = str(slurry_list1[i]) + '-' + str(slurry_list2[i])
        X[col_name] = X[slurry_list1[i]] - X[slurry_list2[i]]
        col_name2 = str(slurry_list1[i]) + '-' + str(slurry_list2[i]) + ' / ' + str(slurry_list1[i])
        X[col_name2] = (X[slurry_list1[i]] - X[slurry_list2[i]]) / X[slurry_list1[i]] 
    
    for i in range(1,4):
        col_name = str(slurry_list1[i]) + '-' + str(slurry_list1[0])
        X[col_name] = X[slurry_list1[i]] - X[slurry_list1[0]]
    
        col_name2 = str(slurry_list1[i]) + '-' + str(slurry_list1[0]) + ' / ' + str(slurry_list1[0])
        X[col_name2] = (X[slurry_list1[i]] - X[slurry_list1[0]]) / X[slurry_list1[0]] 
    
        col_name3 = 'diff_' + col_name
        X[col_name3] = (X[slurry_list1[i]] - X[slurry_list1[0]]) - (X[slurry_list2[i]] - X[slurry_list2[0]]) 
        col_name4 = 'divi_' + col_name
        X[col_name4] = (X[slurry_list1[i]] - X[slurry_list1[0]]) / (X[slurry_list2[i]] - X[slurry_list2[0]])
    
    for i in range(2,4):
        col_name = str(slurry_list1[i]) + '-' + str(slurry_list1[1])
        X[col_name] = X[slurry_list1[i]] - X[slurry_list1[1]]
    
        col_name2 = str(slurry_list1[i]) + '-' + str(slurry_list1[1]) + ' / ' + str(slurry_list1[1])
        X[col_name2] = (X[slurry_list1[i]] - X[slurry_list1[1]]) / X[slurry_list1[1]]
    
        col_name = 'diff_' + col_name
        X[col_name] = (X[slurry_list1[i]] - X[slurry_list1[1]]) - (X[slurry_list2[i]] - X[slurry_list2[1]]) 
        col_name2 = 'divi_' + col_name
        X[col_name2] = (X[slurry_list1[i]] - X[slurry_list1[1]]) / (X[slurry_list2[i]] - X[slurry_list2[1]])
    
    col_name = str(slurry_list1[3]) + '-' + str(slurry_list1[2])
    X[col_name] = X[slurry_list1[3]] - X[slurry_list1[2]]
    col_name2 = str(slurry_list1[3]) + '-' + str(slurry_list1[2]) + ' / ' + str(slurry_list1[2])
    X[col_name2] = (X[slurry_list1[3]] - X[slurry_list1[2]]) / X[slurry_list1[2]]
    col_name3 = 'diff_' + col_name
    X[col_name3] = (X[slurry_list1[3]] - X[slurry_list1[2]]) - (X[slurry_list2[3]] - X[slurry_list2[2]]) 
    col_name4 = 'divi_' + col_name
    X[col_name4] = (X[slurry_list1[3]] - X[slurry_list1[2]]) / (X[slurry_list2[3]] - X[slurry_list2[2]]) 
    
    X['span_slurry_D'] = (X['slurry_D90_TARGET_VALUE'] - X['slurry_D10_TARGET_VALUE'] )/ X['slurry_D50_TARGET_VALUE']
    X['span_target_slurry_D'] = (X['slurry_D90_ORG_TARGET_VALUE'] - X['slurry_D10_ORG_TARGET_VALUE'] )/ X['slurry_D50_ORG_TARGET_VALUE']
    X['diff_span_slurry_D'] = X['span_slurry_D'] - X['span_target_slurry_D']
    X['divi_span_slurry_D'] = X['span_slurry_D'] / X['span_target_slurry_D']
    
    D_1_slurry = np.sqrt(X['slurry_D10_TARGET_VALUE']*0.1)*0.1+ np.sqrt(X['slurry_D10_TARGET_VALUE']*X['slurry_D50_TARGET_VALUE'])*0.4+ np.sqrt(X['slurry_D50_TARGET_VALUE']*X['slurry_D90_TARGET_VALUE'])*0.4+ np.sqrt(X['slurry_D99_TARGET_VALUE']*X['slurry_D90_TARGET_VALUE'])*0.1 
    
    D_2_slurry = X['slurry_D10_TARGET_VALUE']*0.1*0.1+ X['slurry_D10_TARGET_VALUE']*X['slurry_D50_TARGET_VALUE']*0.4+ X['slurry_D50_TARGET_VALUE']*X['slurry_D90_TARGET_VALUE']*0.4+ X['slurry_D99_TARGET_VALUE']*X['slurry_D90_TARGET_VALUE']*0.1 
    
    D_3_slurry = np.sqrt(X['slurry_D10_TARGET_VALUE']*0.1)**3*0.1+ np.sqrt(X['slurry_D10_TARGET_VALUE']*X['slurry_D50_TARGET_VALUE'])**3*0.4+ np.sqrt(X['slurry_D50_TARGET_VALUE']*X['slurry_D90_TARGET_VALUE'])**3*0.4+ np.sqrt(X['slurry_D99_TARGET_VALUE']*X['slurry_D90_TARGET_VALUE'])**3*0.1 
    
    D_4_slurry = np.sqrt(X['slurry_D10_TARGET_VALUE']*0.1)**4*0.1+ np.sqrt(X['slurry_D10_TARGET_VALUE']*X['slurry_D50_TARGET_VALUE'])**4*0.4+ np.sqrt(X['slurry_D50_TARGET_VALUE']*X['slurry_D90_TARGET_VALUE'])**4*0.4+ np.sqrt(X['slurry_D99_TARGET_VALUE']*X['slurry_D90_TARGET_VALUE'])**4*0.1 
    
    X['slurry_D_1_0'] = D_1_slurry
    X['slurry_D_4_3'] = D_4_slurry/D_3_slurry
    X['slurry_D_3_2'] = D_3_slurry/D_2_slurry
    X['slurry_D_2_1'] = D_2_slurry/D_1_slurry
    
    D_time_list = [
        '1PASS_D50','2PASS_D50','3PASS_D50','4PASS_D50','5PASS_D50','6PASS_D50','7PASS_D50'    
    ]
    
    for i in [0,1,2,3,4,5]:
        col_name = D_time_list[i] + '-' + D_time_list[i+1]
        X[col_name] = X[D_time_list[i]] - X[D_time_list[i+1]]
        col_name2 = col_name + '/' + time_interval_list[i + 3]
        X[col_name2] = X[col_name] / X[time_interval_list[i + 3]]
    
    col_name = D_time_list[0] + '-' + D_time_list[6]
    X[col_name] = X[D_time_list[0]] - X[D_time_list[6]]
    col_name2 = col_name + '/' + '7Pass Milling Blade Time-1Pass Milling Blade Time'
    X[col_name2] = X[col_name] / (X[time_interval_list[3]] + X[time_interval_list[4]] + X[time_interval_list[5]] + X[time_interval_list[6]] + X[time_interval_list[7]] + X[time_interval_list[8]])
    
    slip_list1 = [   'slip_D10_TARGET_VALUE',
                     'slip_D50_TARGET_VALUE',
                     'slip_D90_TARGET_VALUE',
                     'slip_D99_TARGET_VALUE']
    slip_list2 = [   'slip_D10_ORG_TARGET_VALUE',         
                     'slip_D50_ORG_TARGET_VALUE',
                     'slip_D90_ORG_TARGET_VALUE',         
                     'slip_D99_ORG_TARGET_VALUE']
    
    for i in range(0,4): 
        col_name = str(slip_list1[i]) + '-' + str(slip_list2[i])
        X[col_name] = X[slip_list1[i]] - X[slip_list2[i]]
        col_name2 = str(slip_list1[i]) + '-' + str(slip_list2[i]) + ' / ' + str(slip_list1[i])
        X[col_name2] = (X[slip_list1[i]] - X[slip_list2[i]]) / X[slip_list1[i]] 
    
    for i in range(1,4):
        col_name = str(slip_list1[i]) + '-' + str(slip_list1[0])
        X[col_name] = X[slip_list1[i]] - X[slip_list1[0]]
    
        col_name2 = str(slip_list1[i]) + '-' + str(slip_list1[0]) + ' / ' + str(slip_list1[0])
        X[col_name2] = (X[slip_list1[i]] - X[slip_list1[0]]) / X[slip_list1[0]] 
    
        col_name3 = 'diff_' + col_name
        X[col_name3] = (X[slip_list1[i]] - X[slip_list1[0]]) - (X[slip_list2[i]] - X[slip_list2[0]]) 
        col_name4 = 'divi_' + col_name
        X[col_name4] = (X[slip_list1[i]] - X[slip_list1[0]]) / (X[slip_list2[i]] - X[slip_list2[0]])
    
    for i in range(2,4):
        col_name = str(slip_list1[i]) + '-' + str(slip_list1[1])
        X[col_name] = X[slip_list1[i]] - X[slip_list1[1]]
    
        col_name2 = str(slip_list1[i]) + '-' + str(slip_list1[1]) + ' / ' + str(slip_list1[1])
        X[col_name2] = (X[slip_list1[i]] - X[slip_list1[1]]) / X[slip_list1[1]]
    
        col_name3 = 'diff_' + col_name
        X[col_name3] = (X[slip_list1[i]] - X[slip_list1[1]]) - (X[slip_list2[i]] - X[slip_list2[1]]) 
        col_name4 = 'divi_' + col_name
        X[col_name4] = (X[slip_list1[i]] - X[slip_list1[1]]) / (X[slip_list2[i]] - X[slip_list2[1]])
    
    col_name = str(slip_list1[3]) + '-' + str(slip_list1[2])
    X[col_name] = X[slip_list1[3]] - X[slip_list1[2]]
    col_name2 = str(slip_list1[3]) + '-' + str(slip_list1[2]) + ' / ' + str(slip_list1[2])
    X[col_name2] = (X[slip_list1[3]] - X[slip_list1[2]]) / X[slip_list1[2]]
    col_name3 = 'diff_' + col_name
    X[col_name3] = (X[slip_list1[3]] - X[slip_list1[2]]) - (X[slip_list2[3]] - X[slip_list2[2]]) 
    col_name4 = 'divi_' + col_name
    X[col_name4] = (X[slip_list1[3]] - X[slip_list1[2]]) / (X[slip_list2[3]] - X[slip_list2[2]])
    
    X['span_slip_D'] = (X['slip_D90_TARGET_VALUE'] - X['slip_D10_TARGET_VALUE'] )/ X['slip_D50_TARGET_VALUE']
    X['span_target_slip_D'] = (X['slip_D90_ORG_TARGET_VALUE'] - X['slip_D10_ORG_TARGET_VALUE'] )/ X['slip_D50_ORG_TARGET_VALUE']
    X['diff_span_slip_D'] = X['span_slip_D'] - X['span_target_slip_D']
    X['divi_span_slip_D'] = X['span_slip_D'] / X['span_target_slip_D']
    
    D_1_slip = np.sqrt(X['slip_D10_TARGET_VALUE']*0.1)*0.1+ np.sqrt(X['slip_D10_TARGET_VALUE']*X['slip_D50_TARGET_VALUE'])*0.4+ np.sqrt(X['slip_D50_TARGET_VALUE']*X['slip_D90_TARGET_VALUE'])*0.4+ np.sqrt(X['slip_D99_TARGET_VALUE']*X['slip_D90_TARGET_VALUE'])*0.1 
    
    D_2_slip = X['slip_D10_TARGET_VALUE']*0.1*0.1+ X['slip_D10_TARGET_VALUE']*X['slip_D50_TARGET_VALUE']*0.4+ X['slip_D50_TARGET_VALUE']*X['slip_D90_TARGET_VALUE']*0.4+ X['slip_D99_TARGET_VALUE']*X['slip_D90_TARGET_VALUE']*0.1 
    
    D_3_slip = np.sqrt(X['slip_D10_TARGET_VALUE']*0.1)**3*0.1+ np.sqrt(X['slip_D10_TARGET_VALUE']*X['slip_D50_TARGET_VALUE'])**3*0.4+ np.sqrt(X['slip_D50_TARGET_VALUE']*X['slip_D90_TARGET_VALUE'])**3*0.4+ np.sqrt(X['slip_D99_TARGET_VALUE']*X['slip_D90_TARGET_VALUE'])**3*0.1 
    
    D_4_slip = np.sqrt(X['slip_D10_TARGET_VALUE']*0.1)**4*0.1+ np.sqrt(X['slip_D10_TARGET_VALUE']*X['slip_D50_TARGET_VALUE'])**4*0.4+ np.sqrt(X['slip_D50_TARGET_VALUE']*X['slip_D90_TARGET_VALUE'])**4*0.4+ np.sqrt(X['slip_D99_TARGET_VALUE']*X['slip_D90_TARGET_VALUE'])**4*0.1 
    
    X['slip_D_1_0'] = D_1_slip
    X['slip_D_4_3'] = D_4_slip/D_3_slip
    X['slip_D_3_2'] = D_3_slip/D_2_slip
    X['slip_D_2_1'] = D_2_slip/D_1_slip
    
    about_slip = list()
    for column in X.columns:
        if any(item in column for item in ['slip_D']):
            about_slip.append(column)
    
    about_slurry = list()
    for column in X.columns:
        if any(item in column for item in ['slurry_D']):
            about_slurry.append(column)
    
    about_pass = list()
    for column in X.columns:
        if any(item in column for item in ['PASS_D']):
            about_pass.append(column)
    about_ss = about_slurry + about_slip + about_pass
    numeric_features_current = numeric_features_current + about_ss 
    
    X.dropna(axis=1, how='any', thresh=1, subset=None, inplace=True)
    X.replace([np.inf, -np.inf], 0, inplace=True)
    
    x_index = X.index
    mydata = X.copy()
    imputer = KNNImputer(n_neighbors = 3, weights =  'distance')
    imputer.fit(mydata)
    mydatatrans = imputer.transform(mydata)
    mydatatrans = pd.DataFrame(mydatatrans)
    mydatatrans.columns = mydata.columns
    X = mydatatrans.copy()
    X = X.round(5)

    ZORDPN_ = list()
    ZORDPN_S = list()
    ZORDPN_V = list()
    ZORDPN_C = list()
    ZORDPN_D = list()
    ZORDPN_T = list()
    l_1210 = list()
    l_1220 = list()
    l_1410 = list()
    l_1420 = list()
    l_1710 = list()

    for column in X.columns:
        if any(item in column for item in ['ZORDPN__']):
            ZORDPN_.append(column) 

        if any(item in column for item in ['ZORDPN_s']):
            ZORDPN_S.append(column) 

        if any(item in column for item in ['ZORDPN_v']):
            ZORDPN_V.append(column)

        if any(item in column for item in ['ZORDPN_c']):
            ZORDPN_C.append(column) 

        if any(item in column for item in ['ZORDPN_d']):
            ZORDPN_D.append(column)

        if any(item in column for item in ['ZORDPN_t']):
            ZORDPN_T.append(column) 

        if any(item in column for item in ['1210_1210']):
            l_1210.append(column) 

        if any(item in column for item in ['1210_1220']):
            l_1220.append(column)

        if any(item in column for item in ['1210_1410']):
            l_1410.append(column)

        if any(item in column for item in ['1210_1420']):
            l_1420.append(column)

        if any(item in column for item in ['1210_1710']):
            l_1710.append(column)

    list_all = l_1210[:22] + l_1220[:39] + l_1410[:2] + l_1420[:7] + ['Thermal_Ring'] + l_1710 + about_time + about_ss + new    

    if inter_type == 'A':
        inter = ZORDPN_V + ZORDPN_S + ZORDPN_C + ZORDPN_D + ZORDPN_T
    else:
        inter = ZORDPN_

    for item in inter:
        for item2 in list_all:
            new_name = str(item) + '__' + str(item2)
            X[new_name] = X[item]*X[item2]

        new_name = str(item)+'__'+'最大重量'+' / '+'內層疊層層數'
        X[new_name] = X[item]/X['1210_1320_BT004_ 內層疊層層數(number of layer)']*X['1210_1320_ZI011_最大重量(Max weight)']

        new_name = str(item)+'__'+'最小重量'+' / '+'內層疊層層數'
        X[new_name] = X[item]/X['1210_1320_BT004_ 內層疊層層數(number of layer)']*X['1210_1320_ZI012_最小重量(Min weight)'] 

        new_name = str(item)+'__'+'最大厚度'+' / '+'內層疊層層數'
        X[new_name] = X[item]/X['1210_1320_BT004_ 內層疊層層數(number of layer)']*X['1210_1330_ZI015_產品最大厚度(Thickness max)']

        new_name = str(item)+'__'+'最小厚度'+' / '+'內層疊層層數'
        X[new_name] = X[item]/X['1210_1320_BT004_ 內層疊層層數(number of layer)']*X['1210_1330_ZI016_產品最小厚度(Thickness min)']

        new_name = str(item)+'__'+'最大重量'
        X[new_name] = X[item]*X['1210_1320_ZI011_最大重量(Max weight)']

        new_name = str(item)+'__'+'最小重量'
        X[new_name] = X[item]*X['1210_1320_ZI012_最小重量(Min weight)'] 

        new_name = str(item)+'__'+'最大厚度'
        X[new_name] = X[item]*X['1210_1330_ZI015_產品最大厚度(Thickness max)']

        new_name = str(item)+'__'+'最小厚度'
        X[new_name] = X[item]*X['1210_1330_ZI016_產品最小厚度(Thickness min)']

        new_name = str(item)+'__'+'內層疊層層數'
        X[new_name] = X[item]*X['1210_1320_BT004_ 內層疊層層數(number of layer)'] 
    
    X.index = x_index
    X = X.loc[:,~X.columns.duplicated()]
    
    return X,Y

def select_imp(X,Y,train):
    x_col = X.columns
    data = pd.concat([X, Y], axis=1)
    data.dropna(axis=0, how='any', inplace=True)
    X = data[x_col]

    l = [
      '1210_1810_ZI062_HALT確認異常(<1hr)',
      '1210_1810_ZI063_HALT確認異常(10min~1hr)',
      '1210_1810_ZI064_HALT確認異常(<10min)'
    ]

    a = [0]*data.shape[0]
    for j in range(data.shape[0]):
        for col in l:
            if str(data[col].values[j]) != '0' and str(data[col].values[j]) != '0.0':
                a[j] = 1
    Y = pd.DataFrame(a).values.ravel()
    X_value = X.values
    X_IMP = X

    if train == True:

        rf = BalancedRandomForestClassifier(n_estimators=100, max_depth = 30,max_features = 0.1,n_jobs = -1)
        output = cross_validate(rf, X_value, Y, cv=5, scoring = 'roc_auc', return_estimator =True)
        feature_importances = pd.DataFrame()
        for idx,estimator in enumerate(output['estimator']):
            temp = pd.DataFrame(estimator.feature_importances_,
                                               index = X.columns,
                                                columns=['importance'])
            feature_importances = pd.concat([feature_importances,temp],axis = 1)

        feature_importances['mean'] = feature_importances.mean(axis=1)
        feature_imp = feature_importances.sort_values('mean',ascending = False)
        value = feature_imp['mean']
        feature = value.index.values
        no_zero = [value > 0]
        no_zero_f = feature[no_zero]
        no_zero_f_rf = list(no_zero_f)

        lg = lgb.LGBMClassifier(
            boosting_type='gbdt',
            objective='binary',
            metrics='auc',
            learning_rate=0.01,
            n_estimators=100,
            max_depth=30,
            num_leaves=30,
            bagging_fraction=0.6,
            feature_fraction=0.6,
            min_data_in_leaf=10,
            lambda_l1=0.001,
            lambda_l2=0.001,
            n_jobs=-1
        )
        output = cross_validate(lg, X_value, Y, cv=5, scoring='roc_auc', return_estimator=True)
        feature_importances = pd.DataFrame()
        for idx, estimator in enumerate(output['estimator']):
            temp = pd.DataFrame(estimator.feature_importances_,
                                index=X.columns,
                                columns=['importance'])
            feature_importances = pd.concat([feature_importances, temp], axis=1)
        feature_importances['mean'] = feature_importances.mean(axis=1)
        feature_imp = feature_importances.sort_values('mean', ascending=False)
        value = feature_imp['mean']
        feature = value.index.values
        no_zero = [value > 0]
        no_zero_f = feature[no_zero]
        no_zero_f_lgbm = list(no_zero_f)

        no_zero_f_1 = [a for a in no_zero_f_rf if a in no_zero_f_lgbm]
        no_zero_f_1 = list(set(no_zero_f_1))
        X_IMP = X[no_zero_f_1]
    Y = pd.DataFrame(Y)
    Y.index = X.index
    return X_IMP,Y

def read_data(dir,name,drop_dup):
    data = pd.read_csv(dir+"data/"+name)
    del data['1210_1810_ZI067_HALT確認異常(24hr&48hr)']
    data = data.drop_duplicates()
    if drop_dup:
        data = data.drop_duplicates(subset="ZORDNO_1320", keep="last")
    data.index = data["ZORDNO_1320"]
    del data["ZORDNO_1320"]
    return data

def read_data2(dir,name,drop_dup):
    data = pd.read_csv(dir+"data/"+name)
    f = open(dir + "prepare/del_col.txt", encoding="utf-8")
    del_features = []
    for line in f:
        del_features.append(line.strip())
    f.close()
    for i in del_features:
        del data[i]
    data.rename(columns={'1210_1810_ZI063_HALT確認異常(<1hr)': '1210_1810_ZI062_HALT確認異常(<1hr)'}, inplace=True)
    data.rename(columns={'1210_1810_ZI064_HALT確認異常(10min~1hr)': '1210_1810_ZI063_HALT確認異常(10min~1hr)'}, inplace=True)
    data.rename(columns={'1210_1810_ZI065_HALT確認異常(<10min)': '1210_1810_ZI064_HALT確認異常(<10min)'}, inplace=True)
    data.rename(columns={'1210_1810_ZI072_3340 IR異常': '1210_1810_ZI066_3340 IR異常'}, inplace=True)
    data = data.drop_duplicates()
    if drop_dup:
        data = data.drop_duplicates(subset="ZORDNO_1320", keep="last")
    data.index = data["ZORDNO_1320"]
    del data["ZORDNO_1320"]
    return data

