import numpy as np
    
Matrix_A = [
    [10,5,0,0,0],
    [3,10,-2,0,0],
    [0,2,-9,-5,0],
    [0,0,5,16,-4],
    [0,0,0,-8,16]]
Matrix_B = [-120,-91,5,-74,-56]

P = [-Matrix_A[0][1] / Matrix_A[0][0]]
Q = [Matrix_B[0] / Matrix_A[0][0]]

for i in range (1,len(Matrix_A)):    
    if i == len(Matrix_A)-1:
        Pi = 0
        Qi = (Matrix_B[i] - Matrix_A[i][i-1]*Q[i-1])/(Matrix_A[i][i] + Matrix_A[i][i-1]*P[i-1])
        P.append(Pi)
        Q.append(Qi)
    else:
        Pi = (-Matrix_A[i][i+1]) / (Matrix_A[i][i] + Matrix_A[i][i-1] * P[i-1])
        Qi = (Matrix_B[i] - Matrix_A[i][i-1] * Q[i-1]) / (Matrix_A[i][i] + Matrix_A[i][i-1] * P[i-1])
        P.append(Pi)
        Q.append(Qi)

X = []

for i in range (len(Matrix_A) - 1,-1,-1):
    if i == len(Matrix_A) - 1:
         X.insert(0,Q[i])
    else:
        Xi = P[i] * X[0] + Q[i]
        X.insert(0,Xi)
print (X)

print(np.linalg.solve(Matrix_A,Matrix_B))