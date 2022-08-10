"""
this program will calculate generalized nth-fib numbers
the seq form is: H_{n+2}=aH_{n+1}+bH_{n},H_0,H_1=p,q
ref= https://ithelp.ithome.com.tw/articles/10233462
"""

import numpy as np

from sympy import symbols
from sympy import Eq
from sympy import solve

from dataclasses import dataclass

from typing import List


@dataclass
class Constant:
    a: int = 1
    b: int = 1
    p: int = 0
    q: int = 1


def gen_fib_1(n: int) -> int:
    # set up coefficients
    a, b, p, q = Constant.a, Constant.b, Constant.p, Constant.q

    if n == 0:
        return p

    if n == 1:
        return q

    pre = q
    pre_pre = p
    ans = 0

    for i in range(2, n + 1):
        ans = a * pre + b * pre_pre
        pre_pre = pre
        pre = ans

    return ans


def gen_fib_2(n: int) -> int:
    # set up coefficients
    a, b, p, q = Constant.a, Constant.b, Constant.p, Constant.q

    if n == 0:
        return p

    if n == 1:
        return q

    mat_a = np.array([[a, b],
                      [1, 0], ])

    u_0 = np.array([[q],
                    [p], ])

    u_k = np.linalg.matrix_power(mat_a, n - 1) @ u_0

    return int(u_k[0])


def gen_fib_3(n: int) -> int:
    # set up coefficients
    a, b, p, q = Constant.a, Constant.b, Constant.p, Constant.q

    if n == 0:
        return p

    if n == 1:
        return q

    mat_a = np.array([[a, b],
                      [1, 0], ])

    u_0 = np.array([[q],
                    [p], ])

    u_k = matrix_fast_power(mat_a, n - 1) @ u_0

    return int(u_k[0])


def gen_fib_4(n: int) -> int:
    # set up coefficients
    a, b, p, q = Constant.a, Constant.b, Constant.p, Constant.q

    if n == 0:
        return p

    if n == 1:
        return q

    u_0 = np.array([[q],
                    [p], ])

    roots = get_roots_of_c_polynomial(a, b)
    alpha, beta = roots[0], roots[1]

    mat_s = np.array([[alpha, beta],
                      [1, 1], ])

    mat_lambda = np.array([[alpha, 0],
                           [0, beta], ])

    mat_inv_s = np.linalg.inv(mat_s)

    u_k = mat_s @ matrix_fast_power(mat_lambda, n - 1) @ mat_inv_s @ u_0

    return int(np.round(u_k[0]))


def gen_fib_5(n: int) -> int:
    # set up coefficients
    a, b, p, q = Constant.a, Constant.b, Constant.p, Constant.q

    if n == 0:
        return p

    if n == 1:
        return q

    u_0 = np.array([[q],
                    [p], ])

    roots = get_roots_of_c_polynomial(a, b)
    alpha, beta = roots[0], roots[1]

    mat_s = np.array([[alpha, beta],
                      [1, 1], ])

    alpha_power_k = num_fast_power(alpha, n - 1)
    beta_power_k = num_fast_power(beta, n - 1)
    mat_lambda_power_k = np.array([[alpha_power_k, 0],
                                   [0, beta_power_k], ])

    mat_inv_s = np.linalg.inv(mat_s)

    u_k = mat_s @ mat_lambda_power_k @ mat_inv_s @ u_0

    return int(np.round(u_k[0]))


def matrix_fast_power(mat_a: np.array, k) -> np.array:
    if k == 1:
        return mat_a

    if k % 2 == 0:
        h = matrix_fast_power(mat_a, k // 2)
        return h @ h

    else:
        h = matrix_fast_power(mat_a, (k - 1) // 2)
        return h @ h @ mat_a


def num_fast_power(v: float, k) -> float:
    if k == 1:
        return v

    if k % 2 == 0:
        h = num_fast_power(v, k // 2)
        return h * h

    else:
        h = num_fast_power(v, (k - 1) // 2)
        return h * h * v


def get_roots_of_c_polynomial(a: int, b: int) -> List[float]:
    mat_a = np.array([[a, b],
                      [1, 0], ])

    roots = np.roots(np.poly(mat_a))
    # sort in descending order
    roots = -np.sort(-roots)

    return roots


def get_roots_of_c_polynomial_2(a: int, b: int) -> List[float]:
    x = symbols('x')
    eq1 = Eq((x ** 2 - a * x - b), 0)
    sol = solve(eq1)

    roots = [float(r) for r in sol]
    roots.sort(reverse=True)
    return roots


def main():
    print("this program will calculate generalized nth-fib numbers")
    print("the seq form is: H_{n+2}=aH_{n+1}+bH_{n},H_0,H_1=p,q ")
    print("coefficients a,b,p,q= ", Constant.a, Constant.b, Constant.p, Constant.q)

    for i in range(15):
        print(i, gen_fib_1(i), gen_fib_2(i), gen_fib_3(i), gen_fib_4(i), gen_fib_5(i))


if __name__ == "__main__":
    main()
