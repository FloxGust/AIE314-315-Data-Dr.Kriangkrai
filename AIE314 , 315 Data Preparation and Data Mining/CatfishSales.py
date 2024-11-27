# Import pandas
print("\n_____________________________________________________ 1 ____________________________________________________\n")
import pandas as pd
# Read data
dataset = pd.read_csv('catfish_sales_1986_2001.csv', parse_dates=[0])
# Printing head of the DataFrame
dataset.head()
print(dataset)
 
 # อ่านไฟล์ catfish_sales_1986_2001.csv แปลงคอลัมน์แรกให้อยู่ในตำแหน่งindexที่ 0 และ print head 5 ตำแหน่งบนสุด
 
print("\n_____________________________________________________ 1 ____________________________________________________\n")
import plotly.express as px
# Limiting DataFrame to specific date
mask = (dataset['Date'] <= '2000-01-01')
dataset = dataset.loc[mask]
# Plotting a part of DataFrame
fig = px.line(dataset, x='Date', y="Sales", title='Catfish sales 1986-2000')
fig.show()
 
 #import plotly สร้างdata frame ที่กำหนดวันที่ ก่อน 2000-01-01 และนำ dataframe มา plot กราฟเส้น
 
 
print("\n_____________________________________________________ 1 ____________________________________________________\n")
import plotly.express as px
fig = px.box(dataset, y="Sales", title='Catfish sales 1986-2000')
fig.show()
 #plot กราฟแท่ง ของ dataset Catfish sales 1986-2000
 
 
 
 
 
print("\n_____________________________________________________ 1 ____________________________________________________\n")
# convert the column (it's a string) to datetime type
datetime_series = pd.to_datetime(dataset['Date'])
# create datetime index passing the datetime series
datetime_index = pd.DatetimeIndex(datetime_series.values)
# datetime_index
period_index = pd.PeriodIndex(datetime_index, freq='M')
# period_index
dataset = dataset.set_index(period_index)
# we don't need the column anymore
dataset.drop('Date',axis=1,inplace=True)
dataset.head()
print(dataset.head())
 
 
 
 
print("\n_____________________________________________________ 1 ____________________________________________________\n")
import plotly.graph_objects as go
from sktime.forecasting.model_selection import temporal_train_test_split
# Splitting dataset (test dataset size is last 12 periods/months)
y_train, y_test = temporal_train_test_split(dataset, test_size=12)
# Visualizing train/test dataset
fig = go.Figure()
fig.add_trace(go.Scatter(
    name="Train DataSet", x=y_train.index.astype(str), y=y_train['Sales']
))
fig.add_trace(go.Scatter(
    name="Test DataSet", x=y_test.index.astype(str), y=y_test['Sales']
))
fig.update_layout(
    title="Splitted dataset"
)
fig.show()
 
 # แบ่งข้อมูล dataset ออกเป็น train 88% และ test 12%
 
 
 
 
 
print("\n_____________________________________________________ 1 ____________________________________________________\n")
from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(y_train['Sales'], order=(1, 1, 1), seasonal_order=(1,0,1,12))
model_fit = model.fit()
y_pred = model_fit.predict(start=len(y_train), end=len(y_train)+11, exog=None, dynamic=True)
 
fig = go.Figure()
fig.add_trace(go.Scatter(
    name="Train DataSet", x=y_train.index.astype(str), y=y_train['Sales']
))
fig.add_trace(go.Scatter(
    name="Test DataSet", x=y_test.index.astype(str), y=y_test['Sales']
))
fig.add_trace(go.Scatter(
    name="Prediction", x=y_pred.index.astype(str), y=y_pred.values
))
fig.update_layout(
    title="Predicted vs actual values"
)
fig.show()
 
 #สร้างกราฟการเปรียบเทียบ ค่าที่ทำนายได้กับค่าจริง
 
 
 
 
 
print("\n_____________________________________________________ 1 ____________________________________________________\n")
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
mae = mean_absolute_error(list(y_test['Sales']), list(y_pred))
mape = mean_absolute_percentage_error(list(y_test['Sales']), list(y_pred))
print('MAE: %.3f' % mae)
print('MAPE: %.3f' % mape)

#MAE (Mean Absolute Error): ค่าที่ต่ำบ่งบอกว่าโมเดลทำนายได้ใกล้เคียงกับค่าจริงโดยเฉลี่ย
#MAPE (Mean Absolute Percentage Error): ค่าที่ต่ำแสดงว่าโมเดลมีความแม่นยำในการทำนายมากขึ้น โดยเฉพาะในแง่ของเปอร์เซ็นต์ความผิดพลาด
#MAE =: 872.091
#MAPE =: 0.034

 
print("\n_____________________________________________________ 1 ____________________________________________________\n")
from datetime import datetime
# Cloning good dataset
broken_dataset = dataset.copy()
# Breaking clonned dataset with random anomaly
broken_dataset.loc[datetime(1998, 12, 1),['Sales']] = 1000
 
import plotly.express as px
# Plotting DataFrame
fig = px.line(
    broken_dataset,
    x=broken_dataset.index.astype(str),
    y=broken_dataset['Sales']
)
fig.update_layout(
    yaxis_title='Sales',
    xaxis_title='Date',
    title='Catfish Sales 1986-2000 (broken)'
)
fig.show()
 
 
 
 
 
 
 
print("\n_____________________________________________________ 1 ____________________________________________________\n")
import plotly.express as px
fig = px.box(broken_dataset, y="Sales")
fig.show()
 
print("\n_____________________________________________________ 1 ____________________________________________________\n")
import plotly.graph_objects as go
from sktime.forecasting.model_selection import temporal_train_test_split
# Splitting dataset (test dataset size is last 12 periods/months)
y_train, y_test = temporal_train_test_split(broken_dataset, test_size=12)
# Visualizing train/test dataset
fig = go.Figure()
fig.add_trace(go.Scatter(
    name="Train DataSet", x=y_train.index.astype(str), y=y_train['Sales']
))
fig.add_trace(go.Scatter(
    name="Test DataSet", x=y_test.index.astype(str), y=y_test['Sales']
))
fig.update_layout(
    title="Splitted dataset"
)
fig.show()
 
#สร้างกราฟการโดยแบ่งช่วงของข้อมูล Test ออกมา 12% 
 
 
 
 
print("\n_____________________________________________________ 1 ____________________________________________________\n")
from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(y_train['Sales'], order=(1, 1, 1), seasonal_order=(1,0,1,12))
model_fit = model.fit()
y_pred = model_fit.predict(start=len(y_train), end=len(y_train)+11, exog=None, dynamic=True)
 
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(
    name="Train DataSet", x=y_train.index.astype(str), y=y_train['Sales']
))
fig.add_trace(go.Scatter(
    name="Test DataSet", x=y_test.index.astype(str), y=y_test['Sales']
))
fig.add_trace(go.Scatter(
    name="Prediction", x=y_pred.index.astype(str), y=y_pred.values
))
fig.update_layout(
    yaxis_title='Sales',
    xaxis_title='Date',
    title='Catfish Sales 1986-2000 incorrect predictions'
)
fig.show()
 
#สร้างกราฟการทำนายที่ผิดพลาด
 
 
 
 
print("\n_____________________________________________________ 1 ____________________________________________________\n")
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
mae = mean_absolute_error(list(y_test['Sales']), list(y_pred))
mape = mean_absolute_percentage_error(list(y_test['Sales']), list(y_pred))
print('MAE: %.3f' % mae)
print('MAPE: %.3f' % mape)
 
#MAE = 8401.402
#MAPE = 0.345 
 
 
 
print("\n_____________________________________________________ 1 ____________________________________________________\n")
# importing the isloation forest
from sklearn.ensemble import IsolationForest
# copying dataset
isf_dataset = broken_dataset.copy()
# initializing Isolation Forest
clf = IsolationForest(max_samples='auto', contamination=0.01)
# training
clf.fit(isf_dataset)
# finding anomalies
isf_dataset['Anomaly'] = clf.predict(isf_dataset)
# saving anomalies to a separate dataset for visualization purposes
anomalies = isf_dataset.query('Anomaly == -1')
 
# contamination: กำหนดไว้ที่ 0.001 (คาดการณ์ว่ามีความผิดปกติ 1% ของข้อมูล)
 
 
 
 
 
print("\n_____________________________________________________ 1 ____________________________________________________\n")
import plotly.graph_objects as go
b1 = go.Scatter(x=isf_dataset.index.astype(str),
                y=isf_dataset['Sales'],
                name="Dataset",
                mode='markers'
               )
b2 = go.Scatter(x=anomalies.index.astype(str),
                y=anomalies['Sales'],
                name="Anomalies",
                mode='markers',
                marker=dict(color='red', size=6,
                            line=dict(color='red', width=1))
               )
layout = go.Layout(
    title="Isolation Forest results",
    yaxis_title='Sales',
    xaxis_title='Date',
    hovermode='closest'
)
data = [b1, b2]
fig = go.Figure(data=data, layout=layout)
fig.show()
 
 
#สร้างกราฟการกระจายของยอดขายทั้งหมด กำหนดความผิดปกติด้วยสีแดง
 
 
 
 
 
print("\n_____________________________________________________ 1 ____________________________________________________\n")
# Importing then local outlier factor
from sklearn.neighbors import LocalOutlierFactor
# copying dataset
lof_dataset = broken_dataset.copy()
# initializing the Local Outlier Factor algorithm
clf = LocalOutlierFactor(n_neighbors=10)
# training and finding anomalies
lof_dataset['Anomaly'] = clf.fit_predict(lof_dataset)
# saving anomalies to another dataset for visualization purposes
anomalies = isf_dataset.query('Anomaly == -1')

#การกำหนดจำนวนเพื่อนบ้าน (Neighbors) ที่จะใช้ในการคำนวณระยะห่างเพื่อหาความหนาแน่นของจุดข้อมูลนั้น ๆ หากจุดข้อมูลใดมีความหนาแน่นต่ำเมื่อเทียบกับเพื่อนบ้าน ถือว่าเป็นความผิดปกติ (Anomaly & Outlier) ซึ่งกำหนดช่วงไว้ที่ 10


