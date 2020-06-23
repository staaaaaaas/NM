import numpy as np
from numpy.linalg import norm, inv

matrix_A = [
[19,-4,-9,-1],
[-2,20,-2,-7],
[6,-5,-25,9],
[0,-3,-9,12]
]

matrix_B = [[100],[-5],[34],[69]]


def transpose(A):
    C = [[A[j][i] for j in range(len(A))]
    for i in range(len(A[0]))] 
    return  C

def transpose_1(A):
    AT = [[0] for i in range(len(A))]
    for i in range(0, len(A)):
        AT[i][0] = A[i]
    return AT


def minus(A, B):
    C = []
    for i in range(0,len(A)):
        c_=[]
        for j in range(0,len(A[0])):
            minus_ = A[i][j] - B[i][j]
            c_.append(minus_)
        C.append(c_)
    return C


def plus(A, B):
    C = []
    for i in range(0,len(A)):
        c_=[]
        for j in range(0,len(A[0])):
            plus_ = A[i][j] + B[i][j]
            c_.append(plus_)
        C.append(c_)
    return C

def mul(A, B):
    C = []

    for i in range(0,len(A)):
        c_ = []
        for j in range(0,len(B[0])):
            elem = 0
            for k in range(0,len(B)):
                elem += A[i][k] * B[k][j]
            c_.append(elem)
        C.append(c_)
    return C

def diagonal(A):
    diag_ = [] 
    for i in range (0,len(A)):
        for j in range (0,len(A)):
            if i == j :
                diag_.append(A[i][j])
    return diag_

def find_alpha_beta(A, B):
    size = len(A)
    alpha = [[0] * size for i in range(size)]
    beta = [0 for i in range(size)]

    for i in range(size):
        beta[i] = B[i][0] / (A[i][i] ) # + 1e-3
        for j in range(size):
            if i!=j:
                alpha[i][j] = -A[i][j] / (A[i][i] ) #  1e-3
    return alpha, beta

def find_norm(A): 
    C = []
    for i in range(0,len(A)):
        suma=0
        for j in range(0,len(A[0])):
            suma += abs(A[i][j])
        C.append(suma)
    return(max(C))

def finish_simple_iter(x, x_prev, norm_alpha, eps):
    norma = find_norm(minus(x,x_prev))
    if norm_alpha == 1:
        return norma <= eps
    else:
        tmp = norm_alpha / (1 - norm_alpha)
        return tmp * norma <= eps

def simple_iter(A, B, eps):
    alpha, beta = find_alpha_beta(A,B)
    beta = transpose_1(beta)
    x = beta
    k = 0

    while True:
        k += 1
        x_i = plus(beta, mul(alpha, x)) 
        if finish_simple_iter(x_i, x, find_norm(alpha), eps):
            break
        
        x = x_i
    return x_i , k

def finish_zeidel(x, x_prev, norm_alpha, norm_c, eps):
    norma = norm(x - x_prev)
    if norm_alpha == 1:
        return norma <= eps
    else:
        tmp = norm_c / (1 - norm_alpha)
        return tmp * norma <= eps

def zeidel(A, B, eps):
    alpha, beta = find_alpha_beta(A,B)
    beta = transpose_1(beta)
    size = len(alpha)

    np_alpha = np.array(alpha)
    np_beta = np.array(beta)
    B = np.tril(np_alpha, -1)
    C = np_alpha - B

    arg1 = inv(np.eye(size, size) - B).dot(C)
    arg2 =  inv(np.eye(size, size) - B).dot(np_beta)
    
    x = arg2
    k = 0

    while True:
        k += 1
        x_i = arg2 + arg1.dot(x)
        if finish_zeidel(x_i, x, norm(arg1), norm(C), eps):
            break
        x = x_i
    return x_i, k


print('simple iteration:')
print(simple_iter(matrix_A,matrix_B, 0.01))

print('zeidel:')
print(zeidel(matrix_A,matrix_B, 0.01))

print('numpy:')
print(np.linalg.solve(matrix_A, matrix_B))

