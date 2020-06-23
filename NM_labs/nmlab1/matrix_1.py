import numpy as np

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


def minus(A,B):
    C = []
    for i in range(0,len(A)):
        c_=[]
        for j in range(0,len(A[0])):
            minus_ = A[i][j] - B[i][j]
            c_.append(minus_)
        C.append(c_)
    return C


def plus(A,B):
    C = []
    for i in range(0,len(A)):
        c_=[]
        for j in range(0,len(A[0])):
            plus_ = A[i][j] + B[i][j]
            c_.append(plus_)
        C.append(c_)
    return C

def diagonal(A):
    diag_ = [] 
    for i in range (0,len(A)):
        for j in range (0,len(A)):
            if i == j :
                diag_.append(A[i][j])
    return diag_

matrix_B = [[100],[-5],[34],[69]]

def vec_to_column(vec):
    column = [[0] for i in range(len(vec))]
    for i in range(0, len(vec)):
        column[i][0] = vec[i]
    return column

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

def get_data(self):
        return [round(i, 4) for i in self]
    

def mul_list(A, B):
    res = [A[i] * B[i] for i in range(len(A))]
    return res


L = [[0 if i <= j else 1 for i in range(4)] for j in range(4)]
print(L)