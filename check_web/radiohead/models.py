from radiohead import db, login_manager
from radiohead import bcrypt
from flask_login import UserMixin
from pymongo import MongoClient,MongoReplicaSetClient
import pandas as pd
import numpy as np

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model,UserMixin):
    id = db.Column(db.Integer(), primary_key=True)
    username = db.Column(db.String(length=30), nullable=False, unique=True)
    email_address = db.Column(db.String(length=50), nullable=False, unique=True)
    password_hash = db.Column(db.String(length=60), nullable=False)
    def __repr__(self):
        return self.username

    @property
    def password(self):
        return self.password

    @password.setter
    def password(self, plain_text_password):
        self.password_hash = bcrypt.generate_password_hash(plain_text_password).decode('utf-8')

    def check_password_correction(self, attempted_password):
        return bcrypt.check_password_hash(self.password_hash, attempted_password)

def EXPORT_MONGO(data_name):
    #mongo_client = MongoClient("mongodb://10.3.1.83:37017/", replicaset='wtckhbd',username='administrator',password='P2ssw0rd')
    mongo_client = MongoClient("10.3.1.82", 37017)
    mongo_client.admin.authenticate('administrator', 'P2ssw0rd')
    db = mongo_client["james"]
    col = db[data_name]
    cursor = col.find()
    mongo_docs = list(cursor)
    data = pd.DataFrame(list(mongo_docs))
    if 'level_0' in data.columns.tolist():
        data.pop('level_0')
    if 'index' in data.columns.tolist():
        data.pop('index')
    data.pop('_id')
    return data

def SELECT_DATA(data,x=''):
    if x != '':
        data1 = data[data['RESULT'] == 1]
        data2 = data1[data1['REAL_RESULT'] != 0]
        data3 = data2[data2['REAL_RESULT'] != 1]
        df = data3[data3['ZORDNO_1320'] == int(x)]
    else:
        data1 = data[data['REAL_RESULT'] != 0]
        data2 = data1[data1['REAL_RESULT'] != 1]
        df = data2[data2['RESULT'] == 1]
    if len(df) > 0:
        df = df.round({"PROB": 3})
        df['ZORDNO_1320'] = [int(a) for a in df['ZORDNO_1320'].values.tolist()]
        df1 = df.copy()
        df['PROB'] = 'MEDIUM'
        df['PROB'][df1['PROB'] < np.percentile(df1['PROB'], 33)] = 'LOW'
        df['PROB'][df1['PROB'] > np.percentile(df1['PROB'], 66)] = 'HIGH'
    return df.to_dict(orient='records')

def UPDATE_DATA(data_name,data,real_result,order):
    if real_result != '' and order != '':
        data.loc[data['ZORDNO_1320'] == int(float(order)), 'REAL_RESULT'] = int(float(real_result))
        #mongo_client = MongoClient("mongodb://10.3.1.83:37017/", replicaset='wtckhbd',username='administrator',password='P2ssw0rd')
        mongo_client = MongoClient("10.3.1.82", 37017)
        mongo_client.admin.authenticate('administrator', 'P2ssw0rd')
        db = mongo_client["james"]
        mycol = db[data_name]
        myquery = {'ZORDNO_1320': int(float(order))}
        newvalues = {"$set": {'REAL_RESULT': int(float(real_result))}}
        mycol.update_one(myquery, newvalues)

def UPDATE_DATA2(data_name,data,order):
    if order != '':
        data.loc[data['ZORDNO_1320'] == int(order), 'REAL_RESULT'] = np.nan
        #mongo_client = MongoClient("mongodb://10.3.1.83:37017/", replicaset='wtckhbd',username='administrator',password='P2ssw0rd')
        mongo_client = MongoClient("10.3.1.82", 37017)
        mongo_client.admin.authenticate('administrator', 'P2ssw0rd')
        db = mongo_client["james"]
        mycol = db[data_name]
        myquery = {'ZORDNO_1320': int(order)}
        newvalues = {"$set": {'REAL_RESULT': 'NaN'}}
        mycol.update_one(myquery, newvalues)

def SELECT_OK_DATA(data,x=''):
    if x != '':
        data1 = data[data['RESULT'] == 1]
        data2 = data1[data1['REAL_RESULT'] == 0]
        df = data2[data2['ZORDNO_1320'] == int(x)]
    else:
        data1 = data[data['RESULT'] == 1]
        df = data1[data1['REAL_RESULT'] == 0]
    if len(df) > 0:
        df = df.round({"PROB": 3})
        df['ZORDNO_1320'] = [int(a) for a in df['ZORDNO_1320'].values.tolist()]
        df1 = df.copy()
        df['PROB'] = 'MEDIUM'
        df['PROB'][df1['PROB'] < np.percentile(df1['PROB'], 33)] = 'LOW'
        df['PROB'][df1['PROB'] > np.percentile(df1['PROB'], 66)] = 'HIGH'
    return df.to_dict(orient='records')

def SELECT_NG_DATA(data,x=''):
    if x != '':
        data1 = data[data['RESULT'] == 1]
        data2 = data1[data1['REAL_RESULT'] == 1]
        df = data2[data2['ZORDNO_1320'] == int(x)]
    else:
        data1 = data[data['RESULT'] == 1]
        df = data1[data1['REAL_RESULT'] == 1]
    if len(df) > 0:
        df = df.round({"PROB": 3})
        df['ZORDNO_1320'] = [int(a) for a in df['ZORDNO_1320'].values.tolist()]
        df1 = df.copy()
        df['PROB'] = 'MEDIUM'
        df['PROB'][df1['PROB'] < np.percentile(df1['PROB'], 33)] = 'LOW'
        df['PROB'][df1['PROB'] > np.percentile(df1['PROB'], 66)] = 'HIGH'
    return df.to_dict(orient='records')

def P_OK_DATA(data,x=''):
    if x != '':
        data1 = data[data['status'] == 0]
        df = data1[data1['ZORDNO_1320'] == int(x)]
    else:
        df = data[data['status'] == 0]
    return df.to_dict(orient='records')

def P_NG_DATA(data,x=''):
    if x != '':
        data1 = data[data['status'] == 1]
        df = data1[data1['ZORDNO_1320'] == int(x)]
    else:
        df = data[data['status'] == 1]
    return df.to_dict(orient='records')

def M_DATA(data,x=''):
    if x != '':
        data1 = data[~data['status'].isin([0,1])]
        df = data1[data1['ZORDNO_1320'] == int(x)]
    else:
        df = data[~data['status'].isin([0,1])]
    return df.to_dict(orient='records')

