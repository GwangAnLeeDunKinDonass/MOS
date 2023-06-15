#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import metrics

from imblearn.over_sampling import SMOTE

from tensorflow.keras.callbacks import EarlyStopping

from tool.visualize_tool import twod_visualization, threed_visualization
from .model.DL_model import dnn

class handmade_pca:
    
    def __init__(self, df):
        # Desingnate x & y Variable
        self.df = df
        self.model_kind = {'dnn':'딥러닝 모델',
                           'random_forest':'랜덤포레스트 모델'}
        
        print('변수:')
        for idx, col in enumerate(df.columns.tolist()):
            print(idx+1, col)
        print("\n'.하나 이상의 독립변수와 종속변수 하나를 지정해주세요.")
        print("위의 보기를 확인하고 독립변수들의 번호를 x, 종속변수의 번호를 y에 넣어주세요.")
        print('\n***독립변수 선택이 어렵다면 날짜, 번호 관련 변수를 제외하고 모두 선택해주세요.***')
        print("\n지정 후, .set_data()를 통해 분석을 위한 데이터를 생성해주세요.")
        print("wnex1) x: 1 2 3 4 5 8 9 10 / y = 46")
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

            self.x = x
            self.y = y
            
        except:
            raise Exception('번호를 다시 한 번 확인해주세요.')
        
        
    def set_data(self):
        # Generate DataFrame with Choosen Variable
        try:
            self.x_data = self.df.iloc[:,self.x]
            self.y_data = self.df.iloc[:,self.y]

            self.df = pd.concat([self.x_data,self.y_data],axis=1)

            y_count = self.y_data.iloc[:,0].value_counts()
            for val in range(len(y_count)):
                if round(y_count[val]/len(self.y_data),2) != round(1/len(y_count),2):
                    print("데이터 불균형이 존재합니다. '.smote()'를 통해 데이터 불균형을 해소하세요")
                    print('SMOTE는 최근접 이웃(k-NN) 알고리즘을 기반으로 하여 데이터를 새로 생성해 종속변수 데이터가 같은 비율을 갖게 하는 알고리즘 입니다.')
                    break
                else:
                    print('.pca()를 통해 주성분분석을 진행하세요.')

            return self.df
    
        except:
            raise Exception('앞서 입력하신 번호가 중복되거나 범위 안에 있는지 다시 한 번 확인해주세요.')
            
    def smote(self):
        # Relaxing Data Unbalance with SMOTE
        x_arr = self.df.iloc[:,:-1].to_numpy()
        y_arr = self.df.iloc[:,-1].to_numpy()
        
        smote = SMOTE(random_state=42)
        new_x_arr,new_y_arr = smote.fit_resample(x_arr,y_arr)
        print('기존 데이터 형태: ', x_arr.shape, y_arr.shape)
        print('SMOTE 적용 후 데이터 형태: ', new_x_arr.shape, new_y_arr.shape)
        print('SMOTE 적용 후 종속변수 데이터 분포: \n', pd.Series(new_y_arr).value_counts())
        
        x_df = pd.DataFrame(new_x_arr, columns=self.df.columns[:-1])
        y_df = pd.DataFrame(new_y_arr, columns=[self.df.columns[-1]])
        self.df = pd.concat([x_df,y_df], axis=1)
        
        print("\n데이터 불균형이 해소 되었습니다. '.pca()'를 통해 주성분분석을 진행하세요.")
        
        return self.df
        
    def pca(self):
        # Choose Number of Components & Do PCA
        try:
            col = self.x_data.columns
            x = StandardScaler().fit_transform(self.x_data)

            features =x.T
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

            print(df_for_explain)
            print("\n위의 'Scree Plot'과 표를 참고하여 주성분 개수를 입력해주세요.")
            print("일반적으로 Scree Plot의 기울기가 완만해지는 시점이나 누적 기여율이 80%~90%인 시점에서 주성분 개수를 결정합니다.")
            print(f'추천 주성분 개수: {num_PC}')
            compo = int(input(f"주성분 개수: "))

            do_pca = PCA(n_components=compo)
            printcipalComponents = do_pca.fit_transform(x)
            pca_df = pd.DataFrame(data=printcipalComponents,
                                  columns = [f'PC{num+1}' for num in range(len(printcipalComponents[0]))])
            self.pca_df = pd.concat([pca_df,self.y_data],axis=1)
            print('')
            if compo == 2:
                print("시각화가 가능합니다. '.pca_visualize(dimension='2d')'을 통해 확인 가능합니다.")
            if compo == 3:
                print("시각화가 가능합니다. '.pca_visualize(dimension='3d')'을 통해 확인 가능합니다.")
            
            print("주성분 설정이 끝났습니다. 'train_split()'을 통해 학습용 데이터를 생성해주세요.")

            return self.pca_df
        
        except:
            raise Exception('입력하신 주성분 개수를 다시 한 번 확인해주세요')
            
    def pca_visualize(self, dimension='d'):
        if dimension=='2d':
            try:
                twod_visualization(self.pca_df)
                print("이제 'train_split()'을 통해 학습용 데이터를 생성해주세요.")
            except:
                print('데이터의 차원이 달라 2차원 시각화가 불가능 합니다.')
        elif dimension=='3d':
            try:
                threed_visualization(self.pca_df)
                print("이제 'train_split()'을 통해 학습용 데이터를 생성해주세요.")
            except:
                print('데이터의 차원이 달라 3차원 시각화가 불가능 합니다.')
        else:
            raise Exception('파라미터를 다시 확인해주세요')
            
    def train_split(self):
        ratio = input('원하는 훈련용/검증용/시험용 데이터 비율을 입력해주세요. (추천: 0.6 0.2 0.2)\n')
        ratio = ratio.split(' ')
        add = 0
        for r in range(len(ratio)):
            ratio[r] = float(ratio[r])
            add += ratio[r]
        if int(add) != 1:
            raise Exception('비율합이 1이 되도록 설정해주세요.')
        ratio[1] = ratio[1] * (1/(1-ratio[2]))

        x_data = self.pca_df.iloc[:,:len(self.pca_df.columns)-1].to_numpy()
        y_data = self.pca_df.iloc[:,-1].to_numpy()
        
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=ratio[2], random_state=0)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=ratio[1], random_state=0)
        print(f'학습용 데이터 수: {len(x_train)}')
        print(f'검증용 데이터 수: {len(x_val)}')
        print(f'시험용 데이터 수: {len(x_test)}')
        
        x_train = x_train.reshape(x_train.shape[0],-1,1)
        x_val = x_val.reshape(x_val.shape[0],-1,1)
        x_test = x_test.reshape(x_test.shape[0],-1,1)
        
        self.x_train = x_train
        self.x_val = x_val
        self.x_test = x_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        print("데이터 분할을 완료했습니다. '.train()'를 통해 학습을 시작해주세요.")
        
    def train(self):
        pprint(self.model_kind)
        kind = input('학습을 진행하기 위해 모델 리스트 중 원하는 모델을 골라주세요.\n')
        if kind == 'dnn':
            self.model = dnn(self.x_train.shape[1])
            
            self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
            
            callbacks = [EarlyStopping(monitor='val_loss',patience=5, mode='min')]
            
            self.model.fit(self.x_train, self.y_train, epochs=100,
                           validation_data = (self.x_val, self.y_val),
                           batch_size=64, verbose=1, callbacks=callbacks)
            
            print("학습이 종료되었습니다.'.predict()'를 통해 예측 및 분류를 진행해주세요.")
            
    def predict(self):
        y_pred = self.model.predict(self.x_test)

        y_pred = y_pred.reshape(-1)

        for y in range(len(y_pred)):
            if y_pred[y] >= 0.5:
                y_pred[y] = 1
            else:
                y_pred[y] = 0

        self.y_pred = y_pred.reshape(-1,1)
        
        accuracy = metrics.accuracy_score(self.y_test, y_pred)
        print("정확도:", accuracy)

        precision = metrics.precision_score(self.y_test, y_pred)
        print("정밀도:", precision)

        recall = metrics.recall_score(self.y_test, y_pred)
        print("재현율:", recall)

        f1 = metrics.f1_score(self.y_test, y_pred)
        print("f1 점수:", f1)
        
        print("학습이 종료되었습니다. 다른 모델을 학습하려면 다시 '.train()'를 실행하세요.")

