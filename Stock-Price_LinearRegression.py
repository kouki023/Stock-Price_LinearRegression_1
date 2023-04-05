import numpy as np
import pandas as pd
from pandas_datareader import data
import matplotlib.pyplot as plt
import yfinance as yf

# 線形回帰モデルのLinearRegressionをインポート
from sklearn.linear_model import LinearRegression
# 時系列分割のためTimeSeriesSplitのインポート
from sklearn.model_selection import TimeSeriesSplit
# 予測精度検証のためMSEをインポート
from sklearn.metrics import mean_squared_error as mse


import warnings
warnings.simplefilter("ignore")

pd.set_option("display.max_rows", 10)

start = "2018-01-01"
end = "2023-03-31"

yf.pdr_override()
data_master = data.get_data_yahoo("^N225", start, end)


# 曜日情報を追加(0:月曜日〜4:金曜日)

data_master['weekday'] = data_master.index.weekday

# data_techinicalにデータをコピー
data_technical = data_master.copy()

# 移動平均を追加
SMA1 = 5   #短期5日
SMA2 = 10  #中期10日
SMA3 = 15  #長期15日
data_technical['SMA1'] = data_technical['Close'].rolling(SMA1).mean() #短期移動平均の算出
data_technical['SMA2'] = data_technical['Close'].rolling(SMA2).mean() #中期移動平均の算出
data_technical['SMA3'] = data_technical['Close'].rolling(SMA3).mean() #長期移動平均の算出



# OpenとCloseの差分を実体Bodyとして計算
data_technical['Body'] = data_technical['Open'] - data_technical['Close']
# 前日終値との差分Close_diffを計算
data_technical['Close_diff'] = data_technical['Close'].diff(1)
# 目的変数となる翌日の終値Close_nextの追加
data_technical['Close_next'] = data_technical['Close'].shift(-1)

# 欠損値がある行を削除
data_technical = data_technical.dropna(how='any')


# 2018年〜2020年を学習用データとする
train = data_technical['2018-01-01' : '2023-01-31']

test = data_technical['2023-02-01' :]

# 学習用データとテストデータそれぞれを説明変数と目的変数に分離する
X_train = train.drop(columns=['Close_next']) #学習用データ説明変数
y_train = train['Close_next'] #学習用データ目的変数
X_test = test.drop(columns=['Close_next']) #テストデータ説明変数
y_test = test['Close_next'] #テストデータ目的変数

# 時系列分割交差検証
valid_scores = []
tscv = TimeSeriesSplit(n_splits=4)
for fold, (train_indices, valid_indices) in enumerate(tscv.split(X_train)):
    X_train_cv, X_valid_cv = X_train.iloc[train_indices], X_train.iloc[valid_indices]
    y_train_cv, y_valid_cv = y_train.iloc[train_indices], y_train.iloc[valid_indices]
    # 線形回帰モデルのインスタンス化
    model = LinearRegression()
    # モデル学習
    model.fit(X_train_cv, y_train_cv)
    # 予測
    y_valid_pred = model.predict(X_valid_cv)
    # 予測精度(RMSE)の算出
    score = np.sqrt(mse(y_valid_cv, y_valid_pred))
    # 予測精度スコアをリストに格納
    valid_scores.append(score)

    print(f'valid_scores: {valid_scores}')

cv_score = np.mean(valid_scores)
print(f'CV score: {cv_score}')

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = np.sqrt(mse(y_test, y_pred))
print(f'RMSE: {score}')

df_result = test[['Close_next']]
df_result['Close_pred'] = y_pred

# 実際のデータと予測データをデータフレームにまとめる
df_result = test[['Close_next']]
df_result['Close_pred'] = y_pred

df_result["diff"] = df_result["Close_next"] - df_result["Close_pred"]
print(df_result)



