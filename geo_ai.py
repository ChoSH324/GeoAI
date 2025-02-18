import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# from statsmodels.stats.outliers_influence import variance_inflation_factor
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

##  data 나누기 train data / test data
X = data.drop(columns=['Value'])     # data에서 Value 열 제외하여 X 그룹으로 설정
y = data['Value']                    # data의 Value 열을 y 그룹으로 설정
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

##TRUE data
TRUE_X = TRUE.drop(columns=['Value'])  # test 데이터 (f1 ~ f19)
TRUE_Y= TRUE['Value']                  # 실제 타겟 변수

## feature 표준화
scaler = StandardScaler()              # 표준화 함수를 scaler 변수에 저장
X_train_scaled = scaler.fit_transform(X_train)     # X_train 데이터에 fit, transform 한번에 적용.
X_test_scaled = scaler.transform(X_test)           # X_test 데이터에 transform 적용
## TRUE 표준화
TRUE_X_scaled = scaler.fit_transform(TRUE_X)       # TRUE_X 데이터에 fit, transform 한번에 적용해 표준화.
#fit은 평균과 표준편차를 계산하는 매서드, transform은 평균과 표준편차를 이용해 표준화 하는 매서드이다.

## 성능 평가 함수
def evaluate_model(model, X, y, model_name):
    y_pred = model.predict(X)           # 모델을 이용해 x 데이터에 대한 예측값 계산
    mse = mean_squared_error(y, y_pred) # 원본 데이터와 예측값에 대한 MSE 계산
    r2 = r2_score(y, y_pred)            # 결정계수 계산
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')    # 5-fold 교차검증 수행, 음수 MSE 사용됨
    cv_mse = -cv_scores.mean()          # 평균 MSE 계산하고 음수로 반환된 값을 양수로 변환
    # sklearn의 교차 검증 기능은 클수록 좋은 효용함수를 기대하므로
    # 비용함수인 MSE의 값(음수)에 -를 붙여 양수로 변환
    print(f"{model_name}:")       # 모델명 출력
    print(f"  MSE: {mse:.4f}")    # MSE 계산값 출력
    print(f"  R2: {r2:.4f}")      # 결정계수 계산값 출력
    print(f"  5-fold CV MSE: {cv_mse:.4f}")     # 5-fold CV MSE 계산값 출력


## Linear Regression 모델에 대한 하이퍼파라미터 그리드 정의
param1_grids = {
    'Ridge': {'alpha': np.logspace(-3, 2, 20)},
    'Lasso': {'alpha': np.logspace(-3, 2, 20)},
    'ElasticNet': {'alpha': np.logspace(-3, 2, 20), 'l1_ratio': np.arange(0.1, 1.0, 0.1)},
    'SGDRegressor' : {'alpha' : np.logspace(-3,2,20),'l1_ratio': np.arange(0.1, 1.0, 0.1)}
}  # 하이퍼파라미터를 Grid Search 방법을 통해 최적의 값을 찾기 위해 그리드 설정

## Linear Model을 만들기 위한 구조
Linear_models = {}    # Linear model을 지정하기 위한 빈 디렉토리 생성
for name, model in [
    ('Ridge', Ridge(random_state=42)),
    ('Lasso', Lasso(random_state=42)),
    ('ElasticNet', ElasticNet(random_state=42)),
    ('SGDRegressor', SGDRegressor(random_state=42)),
]:    # Ridge, Lasso, ElasticNet, SGDRegressor에 대한 반복문 실행
    if name in param1_grids:     # 하이퍼파라미터 그리드 내의 모델에 대한 조건문 실행
        grid_search = GridSearchCV(model, param1_grids[name], cv=5, scoring='neg_mean_squared_error')   # mse가 낮을수록 좋으므로 mse의 음수값을 넣어 변수를 평가한다
        grid_search.fit(X_train_scaled, y_train)            # x를 입력데이터로, y를 정답 데이터로 하여 모델 학습
        Linear_models[name] = grid_search.best_estimator_   # 최적의 estimator 반환
    else:                        # 해당 모델에 대해 그리드가 없는 경우
        Linear_model.fit(X_train_scaled, y_train)           # x를 입력데이터로, y를 정답 데이터로 하여 모델 학습
        Linear_models[name] = model                         # 학습한 모델을 반환한다.

## Voting Linear Regressor
voting_Linear_regressor = VotingRegressor(estimators=list(Linear_models.items()))       # Linear_models 안의 모델들에 Voting regressor 적용
voting_Linear_regressor.fit(X_train_scaled, y_train)   # x를 입력데이터로, y를 정답 데이터로 하여 모델 학습

## Linear Regression 모델 평가
print("\nVoting Regressor 성능:")      # 모델명 출력
evaluate_model(voting_Linear_regressor, X_test_scaled, y_test, "Voting Regressor")

## Final Model에 대한 하이퍼파라미터 그리드 정의
param_grids = {
    'HuberRegressor': {'epsilon': np.arange(1.0, 10, 0.1),'alpha' : np.logspace(-7, 0, 20)},
    'KNeighborsRegressor': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance'],'leaf_size' :np.arange(10, 105, 5) },
    'SVR' : {'gamma': ['scale', 'auto'],'C': np.logspace(-2, 2, 10),'epsilon' : np.logspace(-4, 2, 20)},
    'RandomForestRegressor' : {'n_estimators': np.arange(50, 201, 50),'max_features' : np.arange(1, 20, 1)},
    'MLPRegressor' : {'hidden_layer_sizes' : np.arange(50, 301, 50),'alpha' : np.logspace(-7, 0, 20)}
}    # 하이퍼파라미터를 Grid Search 방법을 통해 최적의 값을 찾기 위해 그리드 설정

## Final Model을 만들기 위한 구조
Final_models = {}    # Final model을 저장하기 위한 빈 디렉토리 생성
for name, model in [
    ('voting_Linear_regressor', voting_Linear_regressor),
    ('HuberRegressor', HuberRegressor(max_iter=10000)),
    ('KNeighborsRegressor', KNeighborsRegressor()),
    ('SVR', SVR(max_iter=10000)),
    ('RandomForestRegressor',RandomForestRegressor(random_state=42)),
    ('MLPRegressor',MLPRegressor(random_state=42,max_iter=10000))
]:    # 각 모델에 대해 반복문 실행할 것.
    if name in param_grids:                 # 하이퍼파라미터 그리드 딕셔너리에 존재하는 모델일 경우
        grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='neg_mean_squared_error')    # 최적의 하이퍼파라미터 구하기
        grid_search.fit(X_train_scaled, y_train)          # 학습 데이터로 모델 학습
        Final_models[name] = grid_search.best_estimator_  # 최적의 하이퍼파라미터 조합 찾고 모델 저장
    else:   # 그 외의 경우
        model.fit(X_train_scaled, y_train)  # Final_model을 학습하여 model 변수에 저장
        Final_models[name]=model            # 학습된 모델 저장

## Final Model을 Stacking Regressor
stacking_regressor = StackingRegressor(estimators=list(Final_models.items()))    # 최종 모델을 stacking resgressor에 적용
stacking_regressor.fit(X_train_scaled, y_train)                                  # 모델 학습

## Fianl Model을 Voting Regressor
voting_regressor = VotingRegressor(estimators=list(Final_models.items()))      # 최종 모델을 voting resgressor에 적용
voting_regressor.fit(X_train_scaled, y_train)

## 앙상블 Final Stacking Regression 모델 평가
print("\nStacking Regressor 성능:")    # 모델명 출력
evaluate_model(stacking_regressor, X_test_scaled, y_test, "Final Stacking Regressor")
# evaluate_model 사용자정의함수를 이용해 최종 stacking 모델의 성능 평가

## 앙상블 Final Voting Regression 모델 평가
print("\nVoting Regressor 성능:")     # 모델명 출력
evaluate_model(voting_regressor, X_test_scaled, y_test, "Final Voting Regressor")
# evaluate_model 사용자정의함수를 이용해 voting 최종 모델의 성능 평가

# RandomForest 모델 학습 및 변수 중요도 추출
rf_model = Final_models['RandomForestRegressor']  # RandomForestRegressor 모델 가져오기
feature_importances = rf_model.feature_importances_  # 변수 중요도 추출
feature_names = X.columns  # 특성 이름 가져오기

# 변수 중요도 정렬
indices = np.argsort(feature_importances)[::-1]  # 중요도 높은 순으로 정렬
top_k = min(19, len(feature_names))   # 최대로 표시할 변수의 수를 19개로 제한
top_indices = indices[:top_k]         # 상위 19개의 변수 인덱스 선택
top_feature_importances = feature_importances[top_indices]      # 상위 19개의 변수 중요도
top_feature_names = [feature_names[i] for i in top_indices]     # 상위 19개의 변수 이름 저장

# 변수 중요도 시각화
plt.figure(figsize=(19, 6))           # 그래프의 크기 설정 (19,6)
plt.title("Top  Feature Importances - RandomForestRegressor")   # 그래프 제목 설정
plt.bar(range(top_k), top_feature_importances, align='center')  # 변수 중요도를 막대그래프로 시각화
plt.xticks(range(top_k), top_feature_names, rotation=45, ha='right')    # x축에 변수 이름 표시
plt.tight_layout()    # 레이아웃 조정
plt.show()       # 그래프 출력

# 변수의 영향 방향 확인
def check_feature_impact(model, X, feature_names):
    impacts = []
    for feature in feature_names:
        X_plus = X.copy()   # 데이터 복사
        X_plus[feature] += X_plus[feature].std()    # 해당 변수에 표준편차만큼 값을 증가
        y_pred_plus = model.predict(X_plus)         # 변수 증가 후 예측값 계산

        X_minus = X.copy()  # 데이터 복사
        X_minus[feature] -= X_minus[feature].std()  # 해당 변수에 표준편차만큼 값을 감소
        y_pred_minus = model.predict(X_minus)       # 변수 감소 후 예측값 계산

        avg_impact = (y_pred_plus - y_pred_minus).mean()     # 예측값 변화의 평균 계산
        impacts.append('Positive' if avg_impact > 0 else 'Negative')   # 변화 방향에 따라 영향 표시 Positive/Negative

    return impacts

# 변수 영향 방향 계산
feature_impacts = check_feature_impact(rf_model, X, top_feature_names)

# 결과 출력
print("\nFeature Importances and Impacts:")
for name, importance, impact in zip(top_feature_names, top_feature_importances, feature_impacts):
    print(f"{name}: Importance = {importance:.4f}, Impact = {impact}")      # 반복문 이용해 출력문 작성

# 회귀계수 출력
for name, model in Final_models.items():
    if hasattr(model, 'coef_'):    # 최종 ensemble 모델의 단일모델 중 회귀 계수를 가지고 있는 모델 (HuberRegressor)
        print(f"\n{name} Coefficients:")   # 모델 이름 출력
        for feature, coef in zip(X.columns, model.coef_):
            print(f"  {feature}: {coef:.4f}")     # 각 feature와 그의 회귀 계수를 소수점 4자리까지 출력
for name, model in Linear_models.items():
    if hasattr(model, 'coef_'):    # 모델이 회귀 계수를 가지고 있는 경우
        print(f"\n{name} Coefficients:")    # 모델 이름 출력
        for feature, coef in zip(X.columns, model.coef_):
            print(f"  {feature}: {coef:.4f}")     # 각 feature와 그의 회귀 계수를 소수점 4자리까지 출력

## 최종 성능 평가 함수
def final_evaluate_model(model, X, y, model_name):
    y_pred = model.predict(X)  # 모델로 예측값 생성
    me = np.mean(y_pred - y)   # Mean Error
    rmse = np.sqrt(mean_squared_error(y, y_pred))  # Root Mean Squared Error
    mae = mean_absolute_error(y, y_pred)  # Mean Absolute Error
    cc = np.corrcoef(y, y_pred)[0, 1]  # Correlation Coefficient

    # 성능 지표 출력
    print(f"{model_name}:")
    print(f"  ME: {me:.4f}")      # 평균 오차 출력
    print(f"  RMSE: {rmse:.4f}")  # RMSE 출력
    print(f"  MAE: {mae:.4f}")    # MAE 출력
    print(f"  CC: {cc:.4f}")      # 상관계수 출력

    return y_pred                 # 예측값 반환

# 모든 모델에 대해 평가 수행 (실제 타겟 변수 사용)
predictions = {}   # 각 모델의 예측값을 저장할 딕셔너리
for name, model in Final_models.items():    # 최종 ensemble 모델과 그 안의 각 모델에 대한 평가
    predictions[name] = final_evaluate_model(model, TRUE_X_scaled, TRUE_Y, name)    # 평가결과 저장
for name, model in Linear_models.items():   # Voting Linear 모델과 그 안의 각 모델에 대한 평가
    predictions[name] = final_evaluate_model(model, TRUE_X_scaled, TRUE_Y, name)    # 평가결과 저장
# Stacking Regressor와 Voting Regressor 평가
predictions['Stacking Regressor'] = final_evaluate_model(stacking_regressor, TRUE_X_scaled, TRUE_Y, "Final Stacking Regressor")      # 평가결과 출력
predictions['Voting Regressor'] = final_evaluate_model(voting_regressor, TRUE_X_scaled, TRUE_Y, "Final Voting Regressor")            # 평가결과 출력


y_pred = stacking_regressor.predict(TRUE_X_scaled)     # 실제 타겟 변수 이용해 예측한 결과 저장
# y_pred와 y를 사용하여 산점도 그리기
plt.figure(figsize=(10, 10))      # 그래프 크기 설정
plt.scatter(TRUE_Y, y_pred, alpha=0.5)    # 실제값과 예측값의 산점도

# y=x 선 그리기
min_val = min(TRUE_Y.min(), y_pred.min())   # y 값의 최소값
max_val = max(TRUE_Y.max(), y_pred.max())   # y 값의 최대값
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)   # y = x 선 그리기

# 그래프 축 설정
plt.xlabel('Actual Values')       # x축 레이블
plt.ylabel('Predicted Values')     # y축 레이블
plt.title('Actual vs Predicted Values')   # 그래프 제목

# 축 범위
plt.xlim(min_val, max_val)   # x축 범위
plt.ylim(min_val, max_val)    # y축 범위

# 그리드
plt.grid(True, linestyle='--', alpha=0.7)   # 점선 그리드

# 선에 대한 텍스트 추가 [y=x]
plt.text(max_val, min_val, 'y=x', color='red', ha='right', va='bottom')

plt.tight_layout()   # 그래프 레이아웃 조정
plt.show()   # 그래프 출력

## Linear Model을 만들기 위한 구조 without Standardzation
Linear_models_unscaled = {}    # 저장하기 위한 딕셔너리
for name, model in [
    ('Ridge', Ridge(random_state=42)),
    ('Lasso', Lasso(random_state=42)),
    ('ElasticNet', ElasticNet(random_state=42)),
    ('SGDRegressor', SGDRegressor(random_state=42)),
]:
    if name in param1_grids_unscaled:     # 하이퍼파라미터 그리드 딕셔너리에 존재하는 모델일 경우
        grid_search = GridSearchCV(model, param1_grids_unscaled[name], cv=5, scoring='neg_mean_squared_error')    # 최적의 하이퍼파라미터 구하기
        grid_search.fit(X_train, y_train)          # train 데이터로 모델 학습
        Linear_models_unscaled[name] = grid_search.best_estimator_     # 최적의 하이퍼파라미터 저장
    else:                                  # 그 외의 모델의 경우
        Linear_model_unscaled.fit(X_train, y_train)  # train 데이터로 모델 학습
        Linear_models[name] = model       # 그외의 모델 명 저장

## Voting Linear Regressor without Standardzation
voting_Linear_regressor_unscaled = VotingRegressor(estimators=list(Linear_models_unscaled.items()))  # voting regressor 적용
voting_Linear_regressor_unscaled.fit(X_train, y_train)                     # 모델 학습


## Voting Linear Regression 모델 평가 without Standardzation
print("\nVoting Regressor without Standardzation 성능:")
evaluate_model(voting_Linear_regressor_unscaled, X_test, y_test, "Voting Regressor without Standardzation")     # 모델 성능 평가 결과 출력


## Final Model을 만들기 위한 구조 without Standardzation
Final_models_unscaled = {}     # 저장하기 위한 딕셔너리
for name, model in [
    ('voting_Linear_regressor', voting_Linear_regressor),
    ('HuberRegressor', HuberRegressor(max_iter=10000)),
    ('KNeighborsRegressor', KNeighborsRegressor()),
    ('SVR', SVR(max_iter=10000)),
    ('RandomForestRegressor',RandomForestRegressor(random_state=42)),
    ('MLPRegressor',MLPRegressor(random_state=42,max_iter=10000))
]:
    if name in param_grids:   # 하이퍼파라미터 그리드 딕셔너리에 존재하는 모델일 경우
        grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)    # train 데이터로 모델 학습
        Final_models[name] = grid_search.best_estimator_    # 최적의 하이퍼파라미터 저장

    else:    # 그 외의 모델일 경우
        model.fit(X_train, y_train)  # Final_model을 model로 변경
        Final_models_unscaled[name]=model  # 그외의 모델명 저장

## Final Model을 Stacking Regressor without Standardzation
stacking_regressor_unscaled = StackingRegressor(estimators=list(Final_models_unscaled.items()))  # 최종모델에 Stacking regressor 적용
stacking_regressor_unscaled.fit(X_train, y_train)    # 모델 학습

## Fianl Model을 Voting Regressor without Standardzation
voting_regressor_unscaled = VotingRegressor(estimators=list(Final_models_unscaled.items()))   # 최종모델에 Voting regressor 적용
voting_regressor_unscaled.fit(X_train, y_train)    # 모델 학습

# 표준화되지 않은 모델의 회귀계수 출력
print("\n표준화되지 않은 모델의 회귀계수:")

for name, model in Final_models_unscaled.items():
    if hasattr(model, 'coef_'):       # 최종 ensemble 모델의 단일모델 중 회귀 계수를 가지고 있는 모델 (HuberRegressor)
        print(f"\n{name} Coefficients:")   # 모델 이름 출력
        for feature, coef in zip(X.columns, model.coef_):
            print(f"  {feature}: {coef:.4f}")    # 각 feature와 그의 회귀 계수를 소수점 4자리까지 출력

for name, model in Linear_models_unscaled.items():
    if hasattr(model, 'coef_'):     # 모델이 회귀 계수를 가지고 있는 경우
        print(f"\n{name} Coefficients:")   # 모델 이름 출력
        for feature, coef in zip(X.columns, model.coef_):
            print(f"  {feature}: {coef:.4f}")    # 각 feature와 그의 회귀 계수를 소수점 4자리까지 출력


# 모든 모델에 대해 평가 수행  (비표준화 데이터)
predictions = {}   # 각 모델의 예측값을 저장할 딕셔너리
for name, model in Linear_models_unscaled.items():    # Voting Linear 모델과 그 안의 각 모델에 대한 평가
    predictions[name] = final_evaluate_model(model, TRUE_X, TRUE_Y, name)    # 평가결과 저장
for name, model in Final_models_unscaled.items():    # 최종 ensemble 모델과 그 안의 각 모델에 대한 평가
    predictions[name] = final_evaluate_model(model, TRUE_X, TRUE_Y, name)    # 평가결과 저장
# Stacking Regressor와 Voting Regressor 평가
predictions['Stacking Regressor without Standardzation'] = final_evaluate_model(stacking_regressor_unscaled, TRUE_X, TRUE_Y, "Final Stacking Regressor without Standardzation")    # 평가결과 출력
predictions['Voting Regressor without Standardzation'] = final_evaluate_model(voting_regressor_unscaled, TRUE_X, TRUE_Y, "Final Voting Regressor without Standardzation")    # 평가결과 출력