import matrix
import math


def f_1(x):
    return 4 * x[0] - math.cos(x[1])


def df_1_1(x):
    return 4


def df_1_2(x):
    return math.sin(x[1])


def f_2(x):
    return 4 * x[1] - math.e ** x[0]


def df_2_1(x):
    return -math.e ** x[0]


def df_2_2(x):
    return 4


def A_1(x):
    data = matrix.Matrix([[0, 0], [0, 0]])
    data.data[0][0] = f_1(x)
    data.data[1][0] = f_2(x)
    data.data[0][1] = df_1_2(x)
    data.data[1][1] = df_2_2(x)
    return data.determinant(data.data)


def A_2(x):
    data = matrix.Matrix([[0, 0], [0, 0]])
    data.data[0][0] = df_1_1(x)
    data.data[1][0] = df_2_1(x)
    data.data[0][1] = f_1(x)
    data.data[1][1] = f_2(x)
    return data.determinant(data.data)


def J(x):
    data = matrix.Matrix([[0, 0], [0, 0]])
    data.data[0][0] = df_1_1(x)
    data.data[1][0] = df_2_1(x)
    data.data[0][1] = df_1_2(x)
    data.data[1][1] = df_2_2(x)
    return data.determinant(data.data)


def newton_plus_one(x):
    y = [0, 0]
    y[0] = x[0] - A_1(x) / J(x)
    y[1] = x[1] - A_2(x) / J(x)
    return y


def newton_method(eps):
    count = 0
    x = [0, 0]
    x_plus_one = newton_plus_one(x)
    while max(abs(x[0] - x_plus_one[0]), abs(x[1] - x_plus_one[1])) > eps:
        x = x_plus_one
        x_plus_one = newton_plus_one(x)
        count += 1
    return x_plus_one, count


def phi_1(x):
    return math.cos(x[1]) / 4


def phi_2(x):
    return math.e ** x[0] / 4


def iteration_plus_one(x):
    y = [0, 0]
    y[0] = phi_1(x)
    y[1] = phi_2(x)
    return y


def iteration_method(eps):
    count = 0
    x = [0, 0]
    x_plus_one = iteration_plus_one(x)
    while max(abs(x[0] - x_plus_one[0]), abs(x[1] - x_plus_one[1])) > eps:
        x = x_plus_one
        x_plus_one = iteration_plus_one(x)
        count += 1
    return x_plus_one, count


def main():
    eps = 0.00001
    print(newton_method(eps))
    print(iteration_method(eps))


main()
