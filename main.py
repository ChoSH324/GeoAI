import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor, TheilSenRegressor, SGDRegressor
from sklearn.svm import SVR
from sklearn.isotonic import IsotonicRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.base import clone
from sklearn.inspection import permutation_importance

## Load the data
data = pd.read_csv('training.csv')    # training.csv 파일을 읽어 data 변수에 저장
TRUE = pd.read_csv('test_ori.csv')    # test_ori.csv 파일을 읽어 TRUE 변수에 저장

## 1. 공분산 matrix
corr_matrix = data.corr()   # 공분산 행렬을 corr_matrix 변수에 저장
 # 공분산 행렬을 삼각형으로
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))   # heatmap에서 상삼각 영역을 숨기도록 행렬의 mask를 설정한다.
plt.style.use('seaborn-v0_8-whitegrid')        # seaborn의 'whitegrid' 스타일을 그래프의 디자인으로 설정한다.
plt.rcParams['figure.figsize'] = (12, 8)       # figure 크기를 12,8로 설정한다.

plt.figure(figsize=(15, 12))                   # 또 다른 figrue 크기를 15, 12로 설정한다.
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, mask=mask)    # 공분산 행렬을 heatmap으로 시각화한다.
plt.title('Feature Correlation Heatmap')   # 제목 설정
plt.tight_layout()       # 레이아웃을 자동으로 설정해 요소들이 겹치지 않도록 한다.
plt.show()        # 그래프 출력

## 2. features의 box plot 확인
plt.figure(figsize=(20, 10))    # figure 사이즈 설정
sns.boxplot(data=data.iloc[:, :-1])    # f1 ~ f19에 대한 boxplot 출력
plt.title('Feature BoxPlot')    # figure 제목 설정
plt.xticks(rotation=90)         # x축 텍스트를 90도 회전하여 표시
plt.show()                      # box plot 출력
#표준화가 필요하다는 것을 확인한다

# 3 'Value'와 특성들 간의 산점도
fig, axes = plt.subplots(5, 4, figsize=(20, 25))   # return 값 두개 받고, subplot을 출력
axes = axes.ravel()    # 다차원 배열을 1차원 배열로 변형
for i, col in enumerate(data.columns[:-1]):        # 'Value'를 제외한 모든 열
    sns.scatterplot(data=data, x=col, y='Value', ax=axes[i])  # scatter plot 출력
    axes[i].set_title(f'{col} vs Value')           # subplot 별 제목 설정
plt.tight_layout()    # 여백 조정
plt.show()            # 산점도 출력

