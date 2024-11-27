import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

data = pd.read_csv("glass.csv")

x_c = data.columns[1]
y_c = data.columns[3]

X = data.iloc[:, 1:3]

model = IsolationForest(contamination=0.05)
data['anomaly'] = model.fit_predict(X)

anomaly_points = data[data['anomaly'] == -1]

print(anomaly_points)

plt.figure(figsize=(10, 6))
plt.scatter(data[x_c], data[y_c], color='blue', label='Normal')
plt.scatter(anomaly_points[x_c], anomaly_points[y_c], color='red', label='Anomaly')
plt.title(f'Column {x_c}: Column {y_c}')
plt.xlabel(x_c)
plt.ylabel(y_c)
plt.legend()
plt.show()


print("________________________________________________สรุปผลการทดลอง____________________________________________________")
# คุณลักษณะที่เลือก: แกน X: คอลัมน์ที่ 1 (data.columns[1]) และ แกน Y: คอลัมน์ที่ 3 (data.columns[3])
# จำนวนความผิดปกติที่ตรวจพบ: Isolation Forest ตรวจพบจุดข้อมูลที่ถูกระบุว่าเป็นความผิดปกติ (anomaly) และแสดงผลออกมาในคอลัมน์ anomaly ในรูปของ -1  ของ DataFrame
# โมเดล Isolation Forest สามารถตรวจจับความผิดปกติในข้อมูลได้โดยไม่ต้องเขียนสูตรการคำนวน  การปรับค่าพารามิเตอร์นั้นขึ้นอยู่กับช่วงที่ต้องการ 
# โดยช่วงค่าที่ปรับแต่งในระดับ contamination: กำหนดไว้ที่ 0.05 (คิดเป็น 5% ของข้อมูลเป็นความผิดปกติ) ทำให้จำนวนจุดที่ถูกระบุว่าเป็นความผิดปกติไม่มากเกินไป เหมาะสมกับข้อมูลที่มีความเป็นไปได้สูงที่จะไม่มีความผิดปกติมากนัก

print("\nคุณลักษณะที่เลือก: แกน X: คอลัมน์ที่ 1 (data.columns[1]) และ แกน Y: คอลัมน์ที่ 3 (data.columns[3])\nจำนวนความผิดปกติที่ตรวจพบ: Isolation Forest ตรวจพบจุดข้อมูลที่ถูกระบุว่าเป็นความผิดปกติ (anomaly) และแสดงผลออกมาในคอลัมน์ anomaly ในรูปของ -1  ของ DataFrame\nโมเดล Isolation Forest สามารถตรวจจับความผิดปกติในข้อมูลได้โดยไม่ต้องเขียนสูตรการคำนวน  การปรับค่าพารามิเตอร์นั้นขึ้นอยู่กับช่วงที่ต้องการ \nโดยช่วงค่าที่ปรับแต่งในระดับ contamination: กำหนดไว้ที่ 0.05 (คิดเป็น 5% ของข้อมูลเป็นความผิดปกติ) ทำให้จำนวนจุดที่ถูกระบุว่าเป็นความผิดปกติไม่มากเกินไป เหมาะสมกับข้อมูลที่มีความเป็นไปได้สูงที่จะไม่มีความผิดปกติมากนัก\n")