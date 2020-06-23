import numpy as np
from copy import copy, deepcopy
from functools import reduce

matrix_A = [[1, 2, -2, 6],
[-3, -5, 14, 13],
[1, 2, -2, -2],
[-2, -4, 5, 10]]

matrix_B = [24, 41, 0, 20]

def get_column(A, k):
    column = [[0] for i in range(len(A))]
    for i in range(len(A)):
        column[i][0] = A[i][k]
    return column

def column_to_vec(column):
    vec = []
    for i in range (len(column)):
        vec.append(column[i][0])
    return vec

def transpose(A):
    C = [[A[j][i] for j in range(len(A))]
    for i in range(len(A[0]))] 
    return  C

def swap(a, i, j):
    t = a[i]
    a[i] = a[j]
    a[j]= t

def LUP_decomposition(A):
    A_ = deepcopy(A)
    size = len(A_)
    P = [i for i in range(size)]
    k_ = 0

    for k in range(size):
        flag = 0
        for i in range(k, size):
            if abs(A_[i][k]) > flag:
                flag = abs(A_[i][k])
                k_ = i 
        if flag == 0:
            print('Вырожденная матрица\n')
            return -1
            
        swap(P, k ,k_)
        swap(A_, k, k_)

        for i in range(k + 1, size):
            A_[i][k] = A_[i][k] / A_[k][k]
            for j in range(k + 1, size):
                A_[i][j] = A_[i][j] - A_[i][k] * A_[k][j] 
    return A_, P
    

def LUP_solve(A, P, B):
    size = len(A)
    X = [0 for i in range(size)]
    Y = [0 for i in range(size)]

    for i in range(size):
        if i == 0:
            Y[i] = B[P[i]]
        else:
            suma_y = sum(map(lambda u, y: u * y, A[i][:i], Y[:i]))
            Y[i] = B[P[i]] - suma_y

    for i in range(size - 1, -1, -1):
        if i == size - 1:
            X[i] = Y[i] / A[i][i]
        else:
            suma_x = sum(map(lambda l, x: l * x, A[i][i + 1 :], X[i + 1 :]))
            X[i] = (Y[i] - suma_x) / A[i][i]
    return X

def LUP_solution(A, B):
    A_ = deepcopy(A)
    B_ = B
    A_, P = LUP_decomposition(A_)
    X = LUP_solve(A_, P, B_)
    return X

def LUP_inverse(A):
    A_ = deepcopy(A)
    E = [[1 if i == j else 0 for i in range (len(A))] for j in range(len(A))]
    inv =[]

    for i in range(len(E)):
        column = get_column(E, i)
        column = column_to_vec(column)
        inv.append(LUP_solution(A_, column))
    inv = transpose(inv)
    return inv

def LUP_determinant(A):
    A_ = deepcopy(A)
    A_ = LUP_decomposition(A_)[0]
    return reduce(lambda x, y: x * y, [A_[i][i] for i in range(len(A_))])
    #reduce(lambda x, y: x*y, [A[i][i]]) эквивалентно ((A[0][0]*A[1][1])*A[2][2])...

#print(LUP_solution(matrix_A, matrix_B))  
#print(np.linalg.solve(matrix_A, matrix_B))
 
#print(LUP_inverse(matrix_A))
#print(np.linalg.inv(np.array(matrix_A)))

#print(LUP_determinant(matrix_A))
#print(np.linalg.det(matrix_A))


