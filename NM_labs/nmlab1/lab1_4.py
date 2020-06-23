import math
import numpy as np

matrix_A = [
[-7,4,5],
[4,-6,-9],
[5,-9,-8],
]

def transpose(A):
    C = [[A[j][i] for j in range(len(A))]
    for i in range(len(A[0]))] 
    return  C

def transpose_1(A):
    AT = [[0] for i in range(len(A))]
    for i in range(0, len(A)):
        AT[i][0] = A[i]
    return AT

def mul(A,B):
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
            else:
                continue
    return diag_

def minus(A, B):
    C = []
    for i in range(0,len(A)):
        c_=[]
        for j in range(0,len(A[0])):
            minus_ = A[i][j] - B[i][j]
            c_.append(minus_)
        C.append(c_)
    return C

def mul_num(A, x):
    C = []
    for i in range(0,len(A)):
        c_=[]
        for j in range(0,len(A[0])):
            mmul = A[i][j] * x
            c_.append(mmul)
        C.append(c_)
    return C


def find_id_max(A):
    i_max = j_max = 0
    elem_max = A[0][0]
    for i in range(len(A)):
        for j in range(i+1, len(A)):
            if abs(A[i][j]) > elem_max:
                elem_max = abs(A[i][j])
                i_max = i
                j_max = j
    return i_max, j_max


def find_phi(a_ii, a_jj, a_ij):
     return math.pi / 4 if a_ii == a_jj else \
            0.5 * math.atan(2 * a_ij / (a_ii - a_jj))

def find_t(A):
    C = math.sqrt(sum([A[i][j] ** 2 for i in range(len(A)) 
        for j in range(i + 1, len(A))]))
    return C

def jacobi(A,eps):
    size = len(A)
    eigen_vectors = [[1 if i == j else 0 for j in range(size)] for i in range(size)]
    while True:
        matrix_U = [[1 if i == j else 0 for j in range(size)] for i in range(size)]
        i, j = find_id_max(A)
        phi = find_phi(A[i][i], A[j][j], A[i][j])
        
        matrix_U[i][j] = -math.sin(phi)
        matrix_U[j][i] = math.sin(phi)
        matrix_U[i][i] = matrix_U[j][j] = math.cos(phi)

        matrix_UT = transpose(matrix_U)
        A = mul(mul(matrix_UT,A), matrix_U)

        eigen_vectors = mul(eigen_vectors, matrix_U)

        if find_t(A) < eps:
            break
    eigen_values = diagonal(A)
    return eigen_values, eigen_vectors

def cheker(A, ev_num, eps):
    eigen_values, eigen_vectors = jacobi(A, eps)
    evec = [[0] for i in range(len(A))]
    size = len(eigen_vectors)
    transpose_1(eigen_values)
    k = 0 
    for i in range(0,size):
        evec[i][0] = eigen_vectors[i][ev_num] 
    if minus(mul(A, evec), mul_num(evec, eigen_values[ev_num])) <= [[eps], [eps], [eps]]:
        k=1
    else:
        k = -1
    return 'yes' if k == 1 else \
            'no'




print('Jacobi:')
print(jacobi(matrix_A, 0.01))
print(cheker(matrix_A, 0, 0.01))

#print('numpy:')

#print(np.linalg.eig(matrix_A))
    


    
    
    
