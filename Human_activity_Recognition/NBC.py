import numpy as np

X_train = np.loadtxt('features.csv',usecols = (0,1,2,3,4,5),delimiter = ',')
# X_train = np.loadtxt('features.txt')
Y_train = np.loadtxt('targets.txt',dtype=int)
data_dim = X_train.shape[1]


# seperate data by class
X_train_0 = np.array([x for x, y in zip(X_train, Y_train) if y == 0])
X_train_1 = np.array([x for x, y in zip(X_train, Y_train) if y == 1])
X_train_2 = np.array([x for x, y in zip(X_train, Y_train) if y == 2])
X_train_3 = np.array([x for x, y in zip(X_train, Y_train) if y == 3])


# Compute in-class mean
mean_0 = np.mean(X_train_0, axis = 0)
mean_1 = np.mean(X_train_1, axis = 0)
mean_2 = np.mean(X_train_2, axis = 0)
mean_3 = np.mean(X_train_3, axis = 0)


# Compute in-class covariance
cov_0 = np.zeros((data_dim, data_dim))
cov_1 = np.zeros((data_dim, data_dim))
cov_2 = np.zeros((data_dim, data_dim))
cov_3 = np.zeros((data_dim, data_dim))

for x in X_train_0:
    cov_0 += np.dot(np.transpose([x - mean_0]), [x - mean_0]) / X_train_0.shape[0]
for x in X_train_1:
    cov_1 += np.dot(np.transpose([x - mean_1]), [x - mean_1]) / X_train_1.shape[0]
for x in X_train_2:
    cov_2 += np.dot(np.transpose([x - mean_2]), [x - mean_2]) / X_train_2.shape[0]
for x in X_train_3:
    cov_3 += np.dot(np.transpose([x - mean_3]), [x - mean_3]) / X_train_3.shape[0]

# Shared covariance is taken as a weighted average of individual in-class covariance.
cov_01 = (cov_0 * X_train_0.shape[0] + cov_1 * X_train_1.shape[0]) / (X_train_0.shape[0] + X_train_1.shape[0])
u, s, v = np.linalg.svd(cov_01, full_matrices=False)
inv = np.matmul(v.T * 1 / s, u.T)
# Directly compute weights and bias
w01 = np.dot(inv, mean_0 - mean_1)
b01 =  (-0.5) * np.dot(mean_0, np.dot(inv, mean_0)) + 0.5 * np.dot(mean_1, np.dot(inv, mean_1))\
    + np.log(float(X_train_0.shape[0]) / X_train_1.shape[0])

cov_02 = (cov_0 * X_train_0.shape[0] + cov_2 * X_train_2.shape[0]) / (X_train_0.shape[0] + X_train_2.shape[0])
u, s, v = np.linalg.svd(cov_02, full_matrices=False)
inv = np.matmul(v.T * 1 / s, u.T)
# Directly compute weights and bias
w02 = np.dot(inv, mean_0 - mean_2)
b02 =  (-0.5) * np.dot(mean_0, np.dot(inv, mean_0)) + 0.5 * np.dot(mean_2, np.dot(inv, mean_2))\
    + np.log(float(X_train_0.shape[0]) / X_train_2.shape[0]) 

cov_03 = (cov_0 * X_train_0.shape[0] + cov_3 * X_train_3.shape[0]) / (X_train_0.shape[0] + X_train_3.shape[0])
u, s, v = np.linalg.svd(cov_03, full_matrices=False)
inv = np.matmul(v.T * 1 / s, u.T)
# Directly compute weights and bias
w03 = np.dot(inv, mean_0 - mean_3)
b03 =  (-0.5) * np.dot(mean_0, np.dot(inv, mean_0)) + 0.5 * np.dot(mean_3, np.dot(inv, mean_3))\
    + np.log(float(X_train_0.shape[0]) / X_train_3.shape[0])

cov_12 = (cov_1 * X_train_1.shape[0] + cov_2 * X_train_2.shape[0]) / (X_train_1.shape[0] + X_train_2.shape[0])
u, s, v = np.linalg.svd(cov_12, full_matrices=False)
inv = np.matmul(v.T * 1 / s, u.T)
# Directly compute weights and bias
w12 = np.dot(inv, mean_1 - mean_2)
b12 =  (-0.5) * np.dot(mean_1, np.dot(inv, mean_1)) + 0.5 * np.dot(mean_2, np.dot(inv, mean_2))\
    + np.log(float(X_train_1.shape[0]) / X_train_2.shape[0]) 

cov_13 = (cov_1 * X_train_1.shape[0] + cov_3 * X_train_3.shape[0]) / (X_train_1.shape[0] + X_train_3.shape[0])
u, s, v = np.linalg.svd(cov_13, full_matrices=False)
inv = np.matmul(v.T * 1 / s, u.T)
# Directly compute weights and bias
w13 = np.dot(inv, mean_1 - mean_3)
b13 =  (-0.5) * np.dot(mean_1, np.dot(inv, mean_1)) + 0.5 * np.dot(mean_3, np.dot(inv, mean_3))\
    + np.log(float(X_train_1.shape[0]) / X_train_3.shape[0])

cov_23 = (cov_2 * X_train_2.shape[0] + cov_3 * X_train_3.shape[0]) / (X_train_2.shape[0] + X_train_3.shape[0])
u, s, v = np.linalg.svd(cov_23, full_matrices=False)
inv = np.matmul(v.T * 1 / s, u.T)
# Directly compute weights and bias
w23 = np.dot(inv, mean_2 - mean_3)
b23 =  (-0.5) * np.dot(mean_2, np.dot(inv, mean_2)) + 0.5 * np.dot(mean_3, np.dot(inv, mean_3))\
    + np.log(float(X_train_2.shape[0]) / X_train_3.shape[0]) 


def predict(x):
    z01 = np.dot(x,w01) + b01
    z02 = np.dot(x,w02) + b02
    z03 = np.dot(x,w03) + b03
    z12 = np.dot(x,w12) + b12
    z13 = np.dot(x,w13) + b13
    z23 = np.dot(x,w23) + b23
    
    p = np.zeros((4))
    p[0] = 1 / (1 + np.exp(-z01)+ np.exp(-z02)+ np.exp(-z03))
    p[1] = 1 / (1 + np.exp(z01) + np.exp(-z12)+ np.exp(-z13))
    p[2] = 1 / (1 + np.exp(z02) + np.exp(z12) + np.exp(-z23))
    p[3] = 1 / (1 + np.exp(z03) + np.exp(z13) + np.exp(z23))
    
    #print(p)
    print(np.argmax(p))

for i in range(120):
    predict(X_train[i,:])
    
    

print("w01:",w01,"\nb01:",b01,"\n\nw02:",w02,"\nb02:",b02,"\n\nw03:",w03,"\nb03:",b03)
print("\n\nw12:",w12,"\nb12:",b12,"\n\nw13:",w13,"\nb13:",b13,"\n\nw23:",w23,"\nb23:",b23)
