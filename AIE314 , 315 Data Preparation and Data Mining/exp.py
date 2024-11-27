print('.............................Cross-Validation...................')
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
 
np.random.seed(34)
 
def f(x):
    return  0.2*x**3 - 0.3*x**2 - 0.4*x + 3.14
 
N = 100
sigma = 0.25
a, b = 0, 5
 
# N samples from a uniform distribution
X = np.random.uniform(a, b, N)
 
# N sampled from a Gaussian dist N(0, sigma)
eps = np.random.normal(0, sigma, N)
 
# Signal
x = np.linspace(a, b, 200)
signal = pd.DataFrame({'x': x, 'signal': f(x)})
 
# Creating our artificial data
y = f(X) + eps
obs = pd.DataFrame({'x': X, 'y': y})
 
x = np.linspace(a, b, 200)
 
fig, ax = plt.subplots(figsize=(10,8))
ax.plot(signal['x'], signal['signal'], '--', label='True signal', linewidth=3)
ax.plot(obs['x'], obs['y'], 'o', label='Observations')
ax.legend()
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.show()
 
print('..............................Validation set...........................')
from sklearn import model_selection
 
x_train, x_test, \
y_train, y_test = model_selection.train_test_split(obs['x'], obs['y'],
                                                   test_size=0.2, random_state=34)
 
x_train= x_train.values.reshape(-1, 1)
x_test= x_test.values.reshape(-1, 1)
 
from sklearn import linear_model
lm = linear_model.LinearRegression()
vs_lm = lm.fit(x_train, y_train)
 
vs_pred = vs_lm.predict(x_test)
from sklearn.metrics import mean_squared_error
vs_mse = mean_squared_error(y_test, vs_pred)
print("Linear model MSE: ", vs_mse)
 
from sklearn.preprocessing import PolynomialFeatures
 
# Quadratic
qm = linear_model.LinearRegression()
poly2 = PolynomialFeatures(degree=2)
x_train2 = poly2.fit_transform(x_train)
x_test2 = poly2.fit_transform(x_test)
 
vs_qm = qm.fit(x_train2, y_train)
vs_qm_pred = vs_qm.predict(x_test2)
print("Quadratic polynomial MSE: ", mean_squared_error(y_test, vs_qm_pred))
 
# cubic
cm = linear_model.LinearRegression()
poly3 = PolynomialFeatures(degree=3)
x_train3 = poly3.fit_transform(x_train)
x_test3 = poly3.fit_transform(x_test)
 
vs_cm = cm.fit(x_train3, y_train)
vs_cm_pred = vs_cm.predict(x_test3)
print("Cubic polynomial MSE: ",mean_squared_error(y_test, vs_cm_pred))
print('Cubic Coefficients:', cm.coef_)
 
prediction = pd.DataFrame({'x': x_test[:,0], 'y': vs_cm_pred}).sort_values(by='x')
fig, ax = plt.subplots(figsize=(10,8))
ax.plot(signal['x'], signal['signal'], '--', label='True signal', linewidth=3)
ax.plot(obs['x'], obs['y'], 'o', label='Observations')
ax.plot(prediction['x'], prediction['y'], '-', label='Cubic Model', lw=2)
ax.legend()
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.show()
 
print('..................................k-fold Cross-Validation...........................')
from sklearn.model_selection import KFold
 
kcv = KFold(n_splits=10, random_state=34, shuffle=True)
 
from sklearn.model_selection import cross_val_score
 
for d in range(1, 9):
    poly = PolynomialFeatures(degree=d)
    X_now = poly.fit_transform(obs['x'].values.reshape(-1, 1))
    model = lm.fit(X_now, obs['y'])
    scores = cross_val_score(model, X_now, obs['y'], scoring='neg_mean_squared_error', cv=kcv, n_jobs=1)
   
    print(f'Degree-{d} polynomial MSE: {np.mean(np.abs(scores)):.5f}, STD: {np.std(scores):.5f}')
 
print('............................Leave-One-Out Cross-Validation.......................')
from sklearn.model_selection import LeaveOneOut
loocv = LeaveOneOut()
 
for d in range(1, 9):
    poly = PolynomialFeatures(degree=d)
    X_now = poly.fit_transform(obs['x'].values.reshape(-1, 1))
    model = lm.fit(X_now, obs['y'])
    scores = cross_val_score(model, X_now, obs['y'], scoring='neg_mean_squared_error', cv=loocv, n_jobs=1)
   
    print(f'Degree-{d} polynomial MSE: {np.mean(np.abs(scores)):.5f}, STD: {np.std(scores):.5f}')
 
print('.......................................Leave-P-Out (LPOCV).....................')
from sklearn.model_selection import LeavePOut
 
for p in range(1,11):
    lpocv = LeavePOut(p)
    print(f'For p={p} we create {lpocv.get_n_splits(X)} samples ')
 
print('..................................Shuffle Splits...........................')
from sklearn.model_selection import ShuffleSplit
ss = ShuffleSplit(n_splits=10, test_size=0.2, random_state=34)
 
for d in range(1, 9):
    poly = PolynomialFeatures(degree=d)
    X_now = poly.fit_transform(obs['x'].values.reshape(-1, 1))
    model = lm.fit(X_now, obs['y'])
    scores = cross_val_score(model, X_now, obs['y'], scoring='neg_mean_squared_error', cv=ss, n_jobs=1)
   
    print(f'Degree-{d} polynomial MSE: {np.mean(np.abs(scores)):.5f}, STD: {np.std(scores):.5f}')
    
print('........................................Hyperparameter Tuning............................')

lambda_range = np.linspace(0.1, 100, 100)
lambda_grid = [{'alpha': lambda_range}]

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

poly = PolynomialFeatures(degree=3)
X_now = poly.fit_transform(x_train)

kcv = KFold(n_splits=10, random_state=34, shuffle=True)

model_ridge = Ridge(max_iter=10000)
cv_ridge = GridSearchCV(estimator=model_ridge, param_grid=lambda_grid, 
                       cv=kcv)
cv_ridge.fit(x_train, y_train)

cv_ridge.best_params_['alpha']
best_ridge = Ridge(alpha=cv_ridge.best_params_['alpha'], max_iter=10000)
best_ridge.fit(X_now, y_train)
print('Cubic Coefficients:', best_ridge.coef_)

ridge_pred = best_ridge.predict(poly.fit_transform(x_test))
prediction = pd.DataFrame({'x': x_test[:,0], 'y': ridge_pred}).sort_values(by='x')

fig, ax = plt.subplots(figsize=(10,8))
ax.plot(signal['x'], signal['signal'], '--', label='True signal', linewidth=3)
ax.plot(obs['x'], obs['y'], 'o', label='Observations')
ax.plot(prediction['x'], prediction['y'], '-', label='Regularised Ridge Model', lw=2)
ax.legend()
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.show()