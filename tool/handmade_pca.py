#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from pprint import pprint

from sklearn import metrics
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import sklearn.svm as svm

from imblearn.over_sampling import SMOTE
import xgboost as xgb

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tool.visualize_tool import twod_visualization, threed_visualization
from .model.DL_model import dnn

class handmade_pca:
    
    tf.random.set_seed(2)
    
    def __init__(self, exp_path, label_data):
        # Desingnate x & y Variable
        self.binary_model_kind = ['로지스틱 회귀',
                           '서포트벡터머신',
                           '랜덤포레스트',
                           'XGBOOST',
                           'DNN'
                          ]
        self.category_model_kind = ['랜덤포레스트',
                           '서포트벡터머신',
                           'XGBOOST',
                           'DNN'
                          ]
        self.trained_model = {}
        self.ment = {'OPENING' : "\n'.set_data()'를 통해 변수를 지정하고 분석을 위한 데이터를 생성해주세요.",
                    'SMOTE' : "\n데이터 불균형이 존재합니다. '.smote()'를 통한 데이터 불균형 해소를 추천합니다.'\nSMOTE는 최근접 이웃(k-NN) 알고리즘을 기반으로 하여 데이터를 새로 생성해 종속변수 데이터가 같은 비율을 갖게 하는 알고리즘 입니다.",
                    'PCA' : "\n독립변수의 수가 너무 많다면 '.pca()'를 통한 차원 축소를 추천합니다.",
                    'SPLIT' : "\n분석을 위한 데이터 생성이 완료되었습니다. '.train_split()'을 통해 데이터 분할을 진행하세요.",
                    'TO_NUMBER' : "\n분석 진행을 위해 '.to_number()'를 통한 문자 형태의 범주형 데이터를 정수로 변환해주세요."}

        label_data['passed_visual_inspection'] = label_data['passed_visual_inspection'].fillna(2)
        label_data = label_data.replace('yes',0)
        label_data = label_data.replace('no',1)

        print('라벨링 진행중..')
        for raw in tqdm(range(len(label_data))):
            if label_data.loc[raw,'machining_finalized']==1:
                label_data.loc[raw,'error'] = 1
            elif label_data.loc[raw,'passed_visual_inspection']==1:
                label_data.loc[raw,'error'] = 1
            else:
                label_data.loc[raw,'error'] = 0

        label_data = label_data[['feedrate','clamp_pressure','tool_condition','error']]
        label_data['error'] = label_data['error'].astype('int')

        file = os.listdir(exp_path)

        print('데이터프레임 결합중...')
        for i in tqdm(range(len(file))):
            df = pd.read_csv(os.path.join(exp_path,file[i]))

    #         df = df.drop(['Machining_Process'],axis=1)
    #         df = df.drop(['M_sequence_number'],axis=1)
    #         df = df.drop(['M_CURRENT_PROGRAM_NUMBER'],axis=1)

            df['error']=label_data.loc[i,'error']

            try:
                total_df = pd.concat([total_df,df])
                total_df = total_df.reset_index(drop=True)
            except:
                total_df = df
           
        self.df = total_df
        print(self.ment['OPENING'])
    
    def set_data(self):
        # Generate DataFrame with Choosen Variable
        print('변수:')
        for idx, col in enumerate(self.df.columns.tolist()):
            print(idx+1, col)
        print("\n'.하나 이상의 독립변수와 종속변수 하나를 지정해주세요.")
        print("위의 보기를 확인하고 독립변수들의 번호를 x, 종속변수의 번호를 y에 넣어주세요.")
        print('\n***독립변수 선택이 어렵다면 날짜, 번호 관련 변수를 제외하고 모두 선택해주세요.***')
        print("\nex1) x: 1 2 3 4 5 8 9 10 / y = 46")
        print("ex2) x: 1:5 8:10 / y = 46")
        print('ex3) x: 1:5 8 9 10 / y = 46')
        
        try:
            x = input('x: ')
            x = x.split(' ')

            origin = x[0]
            for i in range(len(x)):
                if origin != x[i]:
                    i += 2
                if i != len(x)-1:
                    origin = x[i+1]
                if ':' in str(x[i]):
                    x[i] = x[i].split(':')
                    t = [j for j in range(int(x[i][0])-1,int(x[i][1]))]
                    x.pop(i)
                    for k in sorted(t,reverse=True):
                        x.insert(i, k)

                else:
                    x[i]=int(x[i])
                    x[i] -= 1

            y = int(input('y: '))
            y -= 1
            y = [y]
            
            self.x_data = self.df.iloc[:,x]
            self.y_data = self.df.iloc[:,y]

            self.df = pd.concat([self.x_data,self.y_data],axis=1)

            y_count = self.y_data.iloc[:,0].value_counts()
            
            info = [self.df.dtypes[col] for col in self.df.columns]
            if "O" in info:
                print(self.ment['TO_NUMBER'])
                
                return self.df
                
            else:   
#                 self.x_data = StandardScaler().fit_transform(self.x_data)   
#                 x_df = pd.DataFrame(self.x_data, columns=self.df.columns[:-1])
#                 y_df = pd.DataFrame(self.y_data, columns=['error'])
#                 self.scaled_df = pd.DataFrame(pd.concat([x_df, y_df], axis=1))
                scaler = StandardScaler()
                scaler.fit(self.df)
                scaled_arr = scaler.transform(self.df)
                self.scaled_df = pd.DataFrame(scaled_arr, columns=self.df.columns)
                self.scaled_df['error']=self.df['error']
                try:
                    for val in range(len(y_count)):
                        if round(y_count[val]/len(self.y_data),2) != round(1/len(y_count),2):
                            print(self.ment['SMOTE'])
                            break
                        else:
                            print(self.ment["SPLIT"])
                            print(self.ment['PCA'])
                except:
                    print(self.ment["SPLIT"])
                    print(self.ment["PCA"])

                return self.df
            
            
        except:
            raise Exception('번호를 다시 한 번 확인해주세요.')
            
    def to_number(self):
        for col in self.df.columns:
            if self.df.dtypes[col]=='O':
                uni = self.df[col].unique()
                for i, val in enumerate(uni):
                    self.df[col] = self.df[col].replace(val, i)
        
#         self.x_data = self.df.iloc[:,:-1]
#         self.y_data = self.df.iloc[:,-1]
        
#         self.x_data = StandardScaler().fit_transform(self.x_data)  
#         x_df = pd.DataFrame(self.x_data,  columns=self.df.columns[:-1])
#         y_df = pd.DataFrame(self.y_data, columns=['error'])          
#         self.scaled_df = pd.DataFrame(pd.concat([x_df, y_df], axis=1))
        scaler = StandardScaler()
        scaler.fit(self.df)
        scaled_arr = scaler.transform(self.df)
        self.scaled_df = pd.DataFrame(scaled_arr, columns=self.df.columns)
        self.scaled_df['error']=self.df['error']
                    
        y_count = self.y_data.iloc[:,0].value_counts()
        try:
            for val in range(len(y_count)):
                if round(y_count[val]/len(self.y_data),2) != round(1/len(y_count),2):
                    print(self.ment['SMOTE'])
                    break
                else:
                    print(self.ment["SPLIT"])
                    print(self.ment['PCA'])
        except:
            print(self.ment["SPLIT"])
            print(self.ment["PCA"])
    
    def smote(self):
        # Relaxing Data Unbalance with SMOTE
        x_arr = self.scaled_df.iloc[:,:-1].to_numpy()
        y_arr = self.scaled_df.iloc[:,-1].to_numpy()
        
        smote = SMOTE(random_state=2)
        new_x_arr,new_y_arr = smote.fit_resample(x_arr,y_arr)
        print('기존 데이터 형태: ', x_arr.shape, y_arr.shape)
        print('SMOTE 적용 후 데이터 형태: ', new_x_arr.shape, new_y_arr.shape)
        print('SMOTE 적용 후 종속변수 데이터 분포: \n', pd.Series(new_y_arr).value_counts())
        
        x_df = pd.DataFrame(new_x_arr, columns=self.scaled_df.columns[:-1])
        y_df = pd.DataFrame(new_y_arr, columns=[self.scaled_df.columns[-1]])
        self.scaled_df = pd.concat([x_df,y_df], axis=1)
        
        self.x_data = self.scaled_df.iloc[:,:-1]
        self.y_data = self.scaled_df.iloc[:,-1]
        
        print(self.ment["SPLIT"])
        print(self.ment["PCA"])
        
    def pca(self):
        # Choose Number of Components & Do PCA
        
#         col = self.x_data.columns
#         x = StandardScaler().fit_transform(self.x_data)

        features =self.x_data.T
        cov_mat = np.cov(features)

        values, vectors = np.linalg.eig(cov_mat)

        explained_variances = []
        for i in range(len(values)):
            explained_variances.append(values[i] / np.sum(values))
        explained_variances = sorted(explained_variances, reverse=True) 

        plt.plot(explained_variances)
        plt.xlabel('Number of Components')
        plt.ylabel('Explained Variance')
        plt.title('Scree Plot')
        plt.xticks(np.arange(0,len(explained_variances),2))
        plt.show()

        df_for_explain = pd.DataFrame({'고유값' : np.array(values),
                           '기여율(설명력)' : np.array(explained_variances)})
        df_for_explain['누적 기여율'] = df_for_explain['기여율(설명력)'].cumsum()

        idx = df_for_explain.index
        df_for_explain = df_for_explain.T
        df_for_explain.columns = [f'PC{i+1}' for i in idx]

        for col in range(len(df_for_explain.columns)):
            if df_for_explain.iloc[2,col] > 0.8:
                num_PC = col+1
                break
                
        try:
            print(df_for_explain)
            print("\n위의 'Scree Plot'과 표를 참고하여 주성분 개수를 입력해주세요.")
            print("일반적으로 Scree Plot의 기울기가 완만해지는 시점이나 누적 기여율이 80%~90%인 시점에서 주성분 개수를 결정합니다.")
            print(f'\n추천 주성분 개수: {num_PC}')
            print('0을 입력하면 추천 개수로 PCA를 진행합니다.')
            compo = int(input(f"주성분 개수: "))
            
            if compo == 0:
                compo = num_PC

            do_pca = PCA(n_components=compo)
            printcipalComponents = do_pca.fit_transform(self.x_data)
            pca_df = pd.DataFrame(data=printcipalComponents,
                                  columns = [f'PC{num+1}' for num in range(len(printcipalComponents[0]))])
            self.scaled_df = pd.concat([pca_df,self.y_data],axis=1)
            print('')
            if compo == 2:
                print("시각화가 가능합니다. '.pca_visualize(dimension='2d')'을 통해 확인 가능합니다.")
            if compo == 3:
                print("시각화가 가능합니다. '.pca_visualize(dimension='3d')'을 통해 확인 가능합니다.")
            
            print("주성분 설정이 끝났습니다. 'train_split()'을 통해 학습용 데이터를 생성해주세요.")

            return self.scaled_df
        
        except:
            raise Exception('입력하신 주성분 개수를 다시 한 번 확인해주세요')
            
    def pca_visualize(self, dimension='d'):
        if dimension=='2d':
            try:
                twod_visualization(self.scaled_df)
                print("이제 'train_split()'을 통해 학습용 데이터를 생성해주세요.")
            except:
                print('데이터의 차원이 달라 2차원 시각화가 불가능 합니다.')
        elif dimension=='3d':
            try:
                threed_visualization(self.scaled_df)
                print("이제 'train_split()'을 통해 학습용 데이터를 생성해주세요.")
            except:
                print('데이터의 차원이 달라 3차원 시각화가 불가능 합니다.')
        else:
            raise Exception('파라미터를 다시 확인해주세요')
    
    def train_split(self):
        try:
            print('원하는 훈련용/시험용 데이터 비율을 입력해주세요. (추천: 0.8 0.2)')
            print('숫자 0을 입력하면 추천 개수로 데이터 분할을 진행합니다.')
            ratio = input('\n원하는 데이터 비율: ')

            if ratio == '0':
                ratio = '0.8 0.2'

            ratio = ratio.split(' ')
            add = 0
            for r in range(len(ratio)):
                ratio[r] = float(ratio[r])
                add += ratio[r]
            if int(add) != 1:
                raise Exception('비율합이 1이 되도록 설정해주세요.')

            x_data = self.scaled_df.iloc[:,:len(self.scaled_df.columns)-1].to_numpy()
            y_data = self.scaled_df.iloc[:,-1].to_numpy()

            x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=ratio[1], random_state=42)
            print(f'학습용 데이터 수: {len(x_train)}')
            print(f'시험용 데이터 수: {len(x_test)}')
            
            self.train_data_ratio = ratio

            self.x_train = x_train
            self.x_test = x_test
            self.y_train = y_train
            self.y_test = y_test

            print("데이터 분할을 완료했습니다. '.train()'를 통해 학습을 시작해주세요.\n")
            
        except:
            raise Exception('입력하신 데이터 비율을 다시 한 번 확인해주세요.')
            
    def train(self):
        if len(np.unique(self.y_test)) <= 2:
            for idx, model in enumerate(self.binary_model_kind):
                print((idx+1), model)
            kind = int(input('학습을 진행하기 위해 모델 리스트 중 원하는 모델의 번호를 입력해주세요.'))

            if kind == 1:
                print("\n로지스틱 회귀 모델을 선택했습니다.")

                self.LR = LogisticRegression()
                print('학습 중...')
                self.LR.fit(self.x_train, self.y_train)
                
                self.trained_model['로지스틱 회귀'] = self.LR
                
            if kind == 2:
                print("\n서포트벡터머신 모델을 선택했습니다.")

                self.SVM = svm.SVC(kernel='rbf', C=3, gamma=0.25)
                print('학습 중...')
                self.SVM.fit(self.x_train, self.y_train)
                
                self.trained_model['서포트벡터머신'] = self.SVM

            if kind == 3:
                print("\n랜덤 포레스트 모델을 선택했습니다.")

                self.RF = RandomForestClassifier(random_state=333, n_estimators = 200, criterion='entropy', bootstrap=True, max_depth=10,min_samples_leaf=5,min_samples_split=2)
                print('학습 중...')
                self.RF.fit(self.x_train, self.y_train)
                
                self.trained_model['랜덤 포레스트'] = self.RF
                
            if kind == 4:
                print("\nXGBOOST 모델을 선택했습니다. 학습을 진행하기 위해 학습용 데이터에서 검증용 데이터를 추출 해야합니다.")
                print(f"원하는 검증용 데이터 비율을 전체 데이터셋에 대한 비율로 입력해주세요 (추천: {self.train_data_ratio[1]})")
                print('숫자 0을 입력하면 추천 개수로 데이터 분할을 진행합니다.')
                val_ratio = input('\n원하는 데이터 비율: ')

                if val_ratio == '0':
                    val_ratio = self.train_data_ratio[1]
                val_ratio = float(val_ratio)
                val_ratio /= self.train_data_ratio[0]

                x_train, x_val, y_train, y_val = train_test_split(self.x_train, self.y_train, test_size=val_ratio, random_state=42, stratify=self.y_train)
                
                dtrain = xgb.DMatrix(data=x_train, label = y_train)
                dval = xgb.DMatrix(data=x_val, label=y_val)
                self.dtest = xgb.DMatrix(data=self.x_test, label=self.y_test)

                params = {'max_depth' : 10,
                         'eta' : 0.1, 
                         'objective' : 'binary:logistic',
                         'eval_metric' : 'error',
                          'colsample_bytree':0.8,
                          'min_child_weight':1,
                          'reg_lambda':1,
                          'reg_alpha':0.1
                         }

                num_rounds = 500

                wlist = [(dtrain, 'train'), (dval,'eval')]
                
                self.XGB = xgb.train(params = params, dtrain=dtrain, num_boost_round=num_rounds, evals=wlist, early_stopping_rounds=100)
                
                self.trained_model['XGBOOST'] = self.XGB

            if kind == 5:
                print("\nDNN 모델을 선택했습니다. 학습을 진행하기 위해 학습용 데이터에서 검증용 데이터를 추출 해야합니다.")
                print(f"원하는 검증용 데이터 비율을 전체 데이터셋에 대한 비율로 입력해주세요 (추천: {self.train_data_ratio[1]})")
                print('숫자 0을 입력하면 추천 개수로 데이터 분할을 진행합니다.')
                val_ratio = input('\n원하는 데이터 비율: ')

                if val_ratio == '0':
                    val_ratio = self.train_data_ratio[1]
                val_ratio = float(val_ratio)
                val_ratio /= self.train_data_ratio[0]

                x_train, x_val, y_train, y_val = train_test_split(self.x_train, self.y_train, test_size=val_ratio, random_state=42, stratify=self.y_train)

                x_train = x_train.reshape(x_train.shape[0],-1,1)
                x_val = x_val.reshape(x_val.shape[0],-1,1)

                self.DNN = dnn(x_train.shape[1], len(np.unique(self.y_test))-1)

                self.DNN.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

                callbacks = [EarlyStopping(monitor='val_loss',patience=6, mode='min'),ModelCheckpoint('./best.h5', monitor='val_loss',save_best_only=True, mode='min')]

                self.DNN.fit(x_train, y_train, epochs=300,
                               validation_data = (x_val, y_val),
                               batch_size=128, verbose=1, callbacks=callbacks)
                
                self.trained_model['DNN'] = self.DNN
                
        else:
            for idx, model in enumerate(self.category_model_kind):
                print((idx+1), model)
            kind = int(input('학습을 진행하기 위해 모델 리스트 중 원하는 모델의 번호를 입력해주세요.'))
            
            if kind == 1:
                print("\n서포트벡터머신 모델을 선택했습니다.")

                self.SVM = svm.SVC(kernel='rbf', C=3, gamma=0.25)
                print('학습 중...')
                self.SVM.fit(self.x_train, self.y_train)
                
                self.trained_model['서포트벡터머신'] = self.SVM

            if kind == 2:
                print("\n랜덤 포레스트 모델을 선택했습니다.")

                self.RF = RandomForestClassifier(n_estimators = 200, criterion='entropy', bootstrap=True, random_state=42, max_depth=10,min_samples_leaf=5,min_samples_split=2)
                print('학습 중...')
                self.RF.fit(self.x_train, self.y_train)
                
                self.trained_model['랜덤 포레스트'] = self.RF
            
            if kind == 3:
                print("\nXGBOOST 모델을 선택했습니다. 학습을 진행하기 위해 학습용 데이터에서 검증용 데이터를 추출 해야합니다.")
                print(f"원하는 검증용 데이터 비율을 전체 데이터셋에 대한 비율로 입력해주세요 (추천: {self.train_data_ratio[1]})")
                print('숫자 0을 입력하면 추천 개수로 데이터 분할을 진행합니다.')
                val_ratio = input('\n원하는 데이터 비율: ')

                if val_ratio == '0':
                    val_ratio = self.train_data_ratio[1]
                val_ratio = float(val_ratio)
                val_ratio /= self.train_data_ratio[0]

                x_train, x_val, y_train, y_val = train_test_split(self.x_train, self.y_train, test_size=val_ratio, random_state=42, stratify=y_train)
                
                dtrain = xgb.DMatrix(data=x_train, label = y_train)
                dval = xgb.DMatrix(data=x_val, label=y_val)

                params = {'max_depth' : 10,
                         'eta' : 0.1, 
                         'objective' : 'binary:logistic',
                         'eval_metric' : 'error',
                          'colsample_bytree':0.8,
                          'min_child_weight':1,
                          'reg_lambda':1,
                          'reg_alpha':0.1
                         }

                num_rounds = 500

                wlist = [(dtrain, 'train'), (dval,'eval')]
                
                self.XGB = xgb.train(params = params, dtrain=dtrain, num_boost_round=num_rounds, evals=wlist, early_stopping_rounds=100)
                
                self.trained_model['XGBOOST'] = self.XGB

            if kind == 4:
                print("\nDNN 모델을 선택했습니다. 학습을 진행하기 위해 학습용 데이터에서 검증용 데이터를 추출 해야합니다.")
                print(f"원하는 검증용 데이터 비율을 전체 데이터셋에 대한 비율로 입력해주세요 (추천: {self.train_data_ratio[1]})")
                print('숫자 0을 입력하면 추천 개수로 데이터 분할을 진행합니다.')
                val_ratio = input('\n원하는 데이터 비율: ')

                if val_ratio == '0':
                    val_ratio = self.train_data_ratio[1]
                val_ratio = float(val_ratio)
                val_ratio /= self.train_data_ratio[0]

                x_train, x_val, y_train, y_val = train_test_split(self.x_train, self.y_train, test_size=val_ratio, random_state=42, stratify=self.y_train)

                x_train = x_train.reshape(x_train.shape[0],-1,1)
                x_val = x_val.reshape(x_val.shape[0],-1,1)
                self.dtest = xgb.DMatrix(data=self.x_test, label=self.y_test)

                self.DNN = dnn(x_train.shape[1], len(np.unique(self.y_test))-1)

                self.DNN.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

                callbacks = [EarlyStopping(monitor='val_loss',patience=6, mode='min'),ModelCheckpoint('./best.h5', monitor='val_loss',save_best_only=True, mode='min')]

                self.DNN.fit(x_train, y_train, epochs=300,
                               validation_data = (x_val, y_val),
                               batch_size=128, verbose=1, callbacks=callbacks)
                
                self.trained_model['DNN'] = self.DNN
                
        print("학습이 종료되었습니다.")
        print("다른 모델을 추가로 학습하려면 다시 '.train()'를 실행하세요.")
        print("학습한 모델들의 평가지표를 확인하려면 '.evaluate()'를 실행하세요.")
                
    def evaluate(self):
        print("평가가 진행 중 입니다.")
        matric_dict = {}
        for name in self.trained_model.keys():
            model = self.trained_model[name]
            
            try:
                y_pred = model.predict(self.x_test)
            except:
                try:
                    if model.model_name == 'DNN':
                        x_test = self.x_test.reshape(self.x_test.shape[0],-1,1)
                        y_pred = model.predict(x_test)
                except:
                    y_pred = model.predict(self.dtest)

            y_pred = y_pred.reshape(-1)

            for y in range(len(y_pred)):
                if y_pred[y] >= 0.5:
                    y_pred[y] = 1
                else:
                    y_pred[y] = 0

            self.y_pred = y_pred

            accuracy = metrics.accuracy_score(self.y_test, self.y_pred)
            precision = metrics.precision_score(self.y_test, self.y_pred)
            recall = metrics.recall_score(self.y_test, self.y_pred)
            f1 = metrics.f1_score(self.y_test, self.y_pred)
            auc = metrics.roc_auc_score(self.y_test, self.y_pred)
            
            
            matric_list = [accuracy, precision, recall, f1, auc]
            matric_dict[name] = matric_list
            
        matric_df = pd.DataFrame(matric_dict, index = ['정확도','정밀도','재현율','F1 점수', 'AUC'])
        self.matric_df = matric_df.T
        print("모델 평가가 완료되었습니다. '.predict()'를 통해 예측을 진행하세요.")
        
        return self.matric_df