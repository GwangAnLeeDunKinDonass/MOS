#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from .handmade_pca import handmade_pca

VERSION = '3.10.0'

print("Handmade PCA가 초기화 되었습니다." )
print("pandas 라이브러리를 불러온 후, 데이터 경로 지정과 'label_data = pandas.read_csv(데이터 경로)' 코드를 통해 라벨 데이터를 준비해주세요.")
print("데이터를 준비했다면, 'pca = handmade_pca(exp 데이터 경로, 라벨 데이터)' 코드를 통해 분석을 시작해주세요.") 