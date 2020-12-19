import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import operator
import xgboost as xgb
import lightgbm as lgb

from sklearn.preprocessing import LabelEncoder as label, StandardScaler, power_transform, OrdinalEncoder 
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_auc_score ,roc_curve,auc
from sklearn.utils import compute_class_weight
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from datetime import datetime
from imblearn.over_sampling import SMOTE
from numpy.random import RandomState
from xgboost.sklearn import XGBClassifier

import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
sns.set_palette(['#06B1F0', '#FC4B60'])
random_seed = 63445

class eda_n_featEngg():
    def __init__(self,trainf,testf):
        self.train_org = pd.read_csv(trainf)
        self.test_org = pd.read_csv(testf)
        self.tar_col = self.train_org['country_destination']
        self.cate_features = ['gender', 'signup_method', 'affiliate_channel', 'first_affiliate_tracked', 'signup_app']
        #Only those categorical features were chosen who can be analyzed where the categories in the feature plotted against
        #the target column
        self.nume_features = ['age','signup_flow']
        self.date_features = ['date_account_created', 'timestamp_first_active', 'date_first_booking']
        #Numerical features in the form of a date
        self.updt_date_features_cate = None
        self.updt_date_features_nume = None
        self.test_dfb = self.test_org['date_first_booking']
        #Used to store for the sake of identification of null values in the original data

    def ret_train_org(self):
        return self.train_org
    
    def ret_test_org(self):
        return self.test_org

    def ret_test_dfb(self):
        return self.test_dfb
    
    #Visualization of categorical features    
    def cate_viz(self): 
        for p in self.cate_features:
            j=0
            f, ax = plt.subplots(ncols=1,nrows=len(self.train_org[p].unique()),figsize=(5, 20))
            for i in self.train_org[p].unique():
                sns.countplot(self.train_org.loc[self.train_org[p] == i, 'country_destination'],label = i,ax=ax[j])
                ax[j].legend(loc="upper left")
                j=j+1

    def feat_on_cate(self):
        self.train_org['first_affiliate_tracked'] = self.train_org['first_affiliate_tracked'].fillna('null')
        self.test_org['first_affiliate_tracked'] = self.test_org['first_affiliate_tracked'].fillna('null')
        #Replacing the Null Values with the string "null" as to create another category
    
    #Visualization of numerical features
    def nume_viz(self):
        for p in self.nume_features:
            sns.boxplot(self.train_org[p])
            target = ['country_destination']
            fig, ax = plt.subplots(2, 2, figsize=(25, 20))
            for var, subplot in zip(target, ax.flatten()):
                sns.boxplot(x=var, y=p, data=self.train_org, ax=subplot)

    def feat_on_nume(self):
        self.train_org.loc[self.train_org.age > 100, 'age'] = np.nan
        self.train_org.loc[self.train_org.age < 18, 'age'] = np.nan
        self.test_org.loc[self.test_org.age > 100, 'age'] = np.nan
        self.test_org.loc[self.test_org.age < 18, 'age'] = np.nan
        #Replacing the outliers with null values
        self.train_org['age'] = self.train_org['age'].fillna(-1)
        self.test_org['age'] = self.test_org['age'].fillna(-1)
        #Fill null values with -1
        
    def feat_on_date(self):
        #Date_acc_created dissected:
        self.train_org['date_account_created'] = pd.to_datetime(self.train_org['date_account_created'])
        self.test_org['date_account_created'] = pd.to_datetime(self.test_org['date_account_created'])
        self.train_org['dac_year'] = self.train_org.date_account_created.dt.year
        self.train_org['dac_month'] = self.train_org.date_account_created.dt.month
        self.train_org['dac_day'] = self.train_org.date_account_created.dt.day
        self.test_org['dac_year'] = self.test_org.date_account_created.dt.year
        self.test_org['dac_month'] = self.test_org.date_account_created.dt.month
        self.test_org['dac_day'] = self.test_org.date_account_created.dt.day
        ##Date_first_booking dissected:
        self.train_org['date_first_booking'] = pd.to_datetime(self.train_org['date_first_booking'])
        self.test_org['date_first_booking'] = pd.to_datetime(self.test_org['date_first_booking'])
        self.train_org['dfb_year'] = self.train_org.date_first_booking.dt.year
        self.train_org['dfb_month'] = self.train_org.date_first_booking.dt.month
        self.train_org['dfb_day'] = self.train_org.date_first_booking.dt.day
        self.test_org['dfb_year'] = self.test_org.date_first_booking.dt.year
        self.test_org['dfb_month'] = self.test_org.date_first_booking.dt.month
        self.test_org['dfb_day'] = self.test_org.date_first_booking.dt.day
        #Timestamp_first_active dissected:
        tfa = np.vstack(self.train_org.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)
        self.train_org['tfa_year'] = tfa[:,0]
        self.train_org['tfa_month'] = tfa[:,1]
        self.train_org['tfa_day'] = tfa[:,2]
        tfa_t = np.vstack(self.test_org.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)
        self.test_org['tfa_year'] = tfa_t[:,0]
        self.test_org['tfa_month'] = tfa_t[:,1]
        self.test_org['tfa_day'] = tfa_t[:,2]
        #Adding them as new features in the new features list:
        self.updt_date_features_cate = ['dac_year','dfb_year','tfa_year']
        #Visualized as categorical features
        self.updt_date_features_nume = ['dac_month','dac_day','dfb_month','dfb_day','tfa_month','tfa_day']
        #Visualized as numerical features
        
    #Visualization of date-like numerical features
    def date_viz(self):
        for p in self.updt_date_features_cate:
            j=0
            f, ax = plt.subplots(ncols=1,nrows=len(self.train_org[p].unique()),figsize=(5, 20))
            for i in self.train_org[p].unique():
                sns.countplot(self.train_org.loc[self.train_org[p] == i, 'country_destination'],label = i,ax=ax[j])
                ax[j].legend(loc="upper left")
                j=j+1
        for p in self.updt_date_features_nume:
                sns.boxplot(self.train_org[p])
                target = ['country_destination']
                fig, ax = plt.subplots(2, 2, figsize=(25, 20))
                for var, subplot in zip(target, ax.flatten()):
                    sns.boxplot(x=var, y=p, data=self.train_org, ax=subplot)
                    
    #We observe that number of null values for date_first_booking is same as number of times the country destination is NDF.
    #So we remove all columns from the training data for which the value in date_first_booking was null.
    def feat_on_dfb_col(self):
        self.train_org = self.train_org.dropna(subset=['date_first_booking'])

    def heatmap(self):
        f, ax = plt.subplots(figsize=(12, 12))
        ax = sns.heatmap(self.train_org.corr(), vmin=-1, cmap="coolwarm", annot=True)
        
    #Features to drop
    def feat_by_drop(self):
        self.train_org.drop(['date_account_created','dac_day','dac_month','timestamp_first_active','tfa_day','tfa_year','date_first_booking','dfb_day','dfb_month','dfb_year'],axis = 1,inplace = True)
        self.test_org.drop(['date_account_created','dac_day','dac_month','timestamp_first_active','tfa_day','tfa_year','date_first_booking','dfb_day','dfb_month','dfb_year'],axis = 1,inplace = True)      

    def all_feat_engg(self):
        self.feat_on_dfb_col()
        self.feat_on_cate()
        self.feat_on_nume()
        self.feat_on_date()

class Encod_N_Model():
    def __init__(self,trainf,testf):
        self.train_org = trainf
        self.test_org = testf
        self.l = label()
        p={'task': 'train', 'boosting_type': 'gbdt', 'objective': 'multiclass', 'num_class': 11, 'metric': ['multi_error'], 'learning_rate': 0.03, 'num_leaves': 60, 'max_depth': 4, 'feature_fraction': 0.45, 'bagging_fraction': 0.3, 'reg_alpha': 0.15, 'reg_lambda': 0.15, 'min_child_weight': 0}
        self.lgb = lgb.LGBMClassifier(n_jobs=-1,**p)
        self.xgb = XGBClassifier(eta = 0.05, max_depth = 6, alpha = 1.0, eval_metric = "merror", objective='multi:softprob', subsample=0.5, colsample_bytree=0.3, num_class = 11, nthread = 24)
        self.rf = RandomForestClassifier(n_estimators = 225,max_depth=18,random_state=42)
        estimators=[('rf',self.rf),('xgb',self.xgb),('lgb',self.lgb)]
        self.VC = VotingClassifier(estimators,voting='soft',weights=[0.1,0.3,0.6])
        self.test_id = self.test_org['id']       #Storing the id for mapping probabilites to that user
        self.train_countries = self.train_org['country_destination']
        self.train_df = None        #After encoding both
        self.test_df = None         #train_org and test_org
        self.x_train = None             #After splitting
        self.x_test = None              #train_df into  
        self.y_train = None             #train and test
        self.y_test = None              #data
        self.preds = None                   #Running the model on x_test
        self.fin_preds = None   #Final predictions based on complete data
        self.ids = []  #list of ids to be used in the submission file
        self.cts = []  #list of countries to be used in the submission file

    def label_enc(self):
        label_encode = ['country_destination']
        self.train_org['country_destination'] = self.l.fit_transform(self.train_org['country_destination'])

    def concat_onehot_enc(self):
        self.train_org.drop(['id'],axis = 1,inplace = True)
        self.test_org.drop(['id'],axis = 1,inplace = True)
        self.train_org.drop(['country_destination'], axis = 1, inplace = True)
        self.train_org['train'] = 1      #Used for differentiating between
        self.test_org['train'] = 0       #train and test data
        combined = pd.concat([self.train_org,self.test_org])
        combined = pd.get_dummies(combined)
        self.train_df = combined[combined["train"] == 1]
        self.test_df = combined[combined["train"] == 0]
        self.train_df.drop(["train"],axis = 1,inplace = True)   #Drop the unnecessary
        self.test_df.drop(["train"],axis = 1,inplace = True)    #column

    def train_test_spl(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.train_df,self.train_countries,test_size=0.3,shuffle=True,random_state=random_seed,stratify=self.train_countries)

    def metrics_res(self):
        def metrics(true, preds):
            accuracy = accuracy_score(true, preds)
            recall = recall_score(true, preds,average='weighted')
            precision = precision_score(true, preds,average='weighted')
            f1score = f1_score(true, preds,average='weighted')
            print ('accuracy: {}, recall: {}, precision: {}, f1-score: {}'.format(accuracy, recall, precision, f1score))
        metrics(self.y_test, self.preds)

    def eval_of_preds(self):
        self.VC = self.VC.fit(self.x_train,self.y_train)
        self.preds = self.VC.predict(self.x_test)

    def final_preds(self,test_dfb):
        self.VC.fit(self.train_df,self.train_countries)
        self.fin_preds = self.VC.predict_proba(self.test_df)
        bool_dfb = pd.isnull(test_dfb)
        for i in range(len(self.test_id)):
            if bool_dfb[i]:                       #All the three probabilties NDF when date_first_booking 
                self.ids += [self.test_id[i]] * 3 #column is null
                self.cts += ['NDF','NDF','NDF']
            else:
                idx = self.test_id[i]
                self.ids += [idx] * 3
                self.cts += self.l.inverse_transform(np.argsort(self.fin_preds[i])[::-1])[:3].tolist()

    def final_sub_file(self):
        sub = pd.DataFrame(np.column_stack((self.ids, self.cts)), columns=['id', 'country_destination'])
        sub.to_csv('sub.csv',index=False)
        
    def run_the_model(self):
        self.label_enc()
        self.concat_onehot_enc()
        self.train_test_spl()
        self.eval_of_preds()         #Evaluating predictions for x_train
        self.metrics_res()           #Calculation of metric like accuracy

if __name__ == "__main__":
    train_file = "../input/airbnb-new-user/train.csv"
    test_file = "../input/airbnb-new-user/test.csv"

    eda = eda_n_featEngg(train_file,test_file)

    eda.all_feat_engg()
    eda.cate_viz()
    eda.nume_viz()
    eda.date_viz()
    eda.heatmap()
    eda.feat_by_drop()

    train = eda.ret_train_org()
    test = eda.ret_test_org()
    test_dfb = eda.ret_test_dfb()

    model = Encod_N_Model(train,test)

    model.run_the_model()
    model.final_preds(test_dfb)
    model.final_sub_file()