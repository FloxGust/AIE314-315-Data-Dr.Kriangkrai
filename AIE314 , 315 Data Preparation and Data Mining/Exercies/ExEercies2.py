import numpy as np
import matplotlib.pyplot as plt
 
print("\n____________________________________Excercies 1-2__________________________________________")
 
points = {'A': (1, 1), 'B': (2, 3), 'C': (6, 7), 'D': (7, 8), 'E': (6, 10)}
 
x = [p[0] for p in points.values()]
y = [p[1] for p in points.values()]
 
plt.scatter(x, y)
 
for point, coords in points.items():
    plt.text(coords[0], coords[1], point, fontsize=12)
 
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot of Points')
plt.grid(True)
plt.show()
 
print("\n____________________________________Excercies 4__________________________________________")
 
def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2))
 
labels = list(points.keys())
n = len(points)
similarity_matrix = np.zeros((n, n))
 
for i in range(n):
    for j in range(n):
        similarity_matrix[i, j] = euclidean_distance(points[labels[i]], points[labels[j]])
 
similarity_matrix = np.around(similarity_matrix, decimals=2)
 
print("\nEuclidean Distance Similarity Matrix:")
print(similarity_matrix,"\n\n")
 
print("\n____________________________________Excercies 5__________________________________________")
 
points_new = {
    'A': [2.5, 3.1, 4.0, 5.2, 1.8],
    'B': [1.2, 2.8, 3.5, 4.1, 2.3],
    'C': [3.3, 3.7, 2.9, 5.0, 1.5],
    'D': [2.0, 3.0, 4.5, 4.8, 2.0],
    'E': [1.8, 2.5, 3.8, 4.3, 2.7]
}
labels_new = list(points_new.keys())
n_new = len(points_new)
sm_new = np.zeros((n_new, n_new))
 
for i in range(n_new):
    for j in range(n_new):
        sm_new[i, j] = euclidean_distance(points_new[labels_new[i]], points_new[labels_new[j]])
 
print("Euclidean Distance Similarity Matrix (New Data):")
sm_new = np.around(sm_new, decimals=2)
 
print(sm_new,"\n\n")
 
print("\n____________________________________Excercies 6__________________________________________")
 
def manhattan_distance(p1, p2):
    return np.sum(np.abs(np.array(p1) - np.array(p2)))
 
sm_manhattan = np.zeros((n, n))
 
for i in range(n):
    for j in range(n):
        sm_manhattan[i, j] = manhattan_distance(points[labels[i]], points[labels[j]])
 
print("Manhattan Distance Similarity Matrix:")
print(sm_manhattan,"\n\n")


points_array = np.array(list(points.values()))

cov_matrix = np.cov(points_array, rowvar=False)
inv_cov_matrix = np.linalg.inv(cov_matrix)

sm_mahalanobis = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        diff = points_array[i] - points_array[j]
        sm_mahalanobis[i, j] = np.sqrt(np.dot(np.dot(diff, inv_cov_matrix), diff.T))

sm_mahalanobis = np.around(sm_mahalanobis, decimals=2)
print("Mahalanobis Distance Similarity Matrix:")
print(sm_mahalanobis,"\n\n")
 
 
print("\n____________________________________Excercies 7__________________________________________")
Exercies7 = "\nหลังจากที่เราได้คำนวณระยะทาง Euclidean, Manhattan และ Mahalanobis เราจะเห็นว่าระยะทางแต่ละแบบมีลักษณะการวัดที่แตกต่างกัน:\n Euclidean distance: เป็นการวัดระยะทางเชิงเรขาคณิต\n Manhattan distance: วัดระยะทางโดยการคำนึงถึงความแตกต่างแบบแนวแกน\n Mahalanobis distance: ใช้การกระจายของข้อมูลเข้ามาคำนวณ ทำให้เหมาะสมกับข้อมูลที่มีการกระจายตัวไม่เท่ากัน\n ทั้งนี้การเลือกใช้ระยะทางแต่ละแบบขึ้นอยู่กับลักษณะข้อมูลและวัตถุประสงค์ของการวิเคราะห์\n"
print("\n7. สรุปผลการทดลองที่ได้จากการคำนวณด้วยตัวเองและการคำนวณโดยใช้โปรแกรม Python\n ", Exercies7)