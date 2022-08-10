"""
this program will calculate generalized nth-fib numbers
the seq form is: H_{n+3}=aH_{n+2}+b_{n+1}+cH_{n},H_0,H_1,H_2=p,q,r
ref= https://ithelp.ithome.com.tw/articles/10233462
"""

import numpy as np

from dataclasses import dataclass

from typing import List


@dataclass
class Constant:
    a: int = 1
    b: int = 1
    c: int = 0
    p: int = 0
    q: int = 1
    r: int = 1


def tri_fib_1(n: int) -> int:
    # set up coefficients
    a, b, c, p, q, r = Constant.a, Constant.b, Constant.c, Constant.p, Constant.q, Constant.r

    if n == 0:
        return p

    if n == 1:
        return q

    if n == 2:
        return r

    pre = r
    pre_pre = q
    pre_pre_pre = p
    ans = 0

    for i in range(3, n + 1):
        ans = a * pre + b * pre_pre + c * pre_pre_pre
        pre_pre_pre = pre_pre
        pre_pre = pre
        pre = ans

    return ans


def tri_fib_3(n: int) -> int:
    # set up coefficients
    a, b, c, p, q, r = Constant.a, Constant.b, Constant.c, Constant.p, Constant.q, Constant.r

    if n == 0:
        return p

    if n == 1:
        return q

    if n == 2:
        return r

    mat_a = np.array([[a, b, c],
                      [1, 0, 0],
                      [0, 1, 0]])

    u_0 = np.array([[r],
                    [q],
                    [p], ])

    u_k = matrix_fast_power(mat_a, n - 2) @ u_0

    return int(u_k[0])


def tri_fib_5(n: int) -> int:
    # set up coefficients
    a, b, c, p, q, r = Constant.a, Constant.b, Constant.c, Constant.p, Constant.q, Constant.r

    if n == 0:
        return p

    if n == 1:
        return q

    if n == 2:
        return r

    u_0 = np.array([[r],
                    [q],
                    [p], ])

    # need to check whether values are legal
    roots = get_roots_of_c_polynomial(a, b, c)
    alpha, beta, gamma = roots[0], roots[1], roots[2]

    mat_s = np.array([[alpha * alpha, beta ** 2, gamma ** 2],
                      [alpha, beta, gamma],
                      [1, 1, 1], ])

    alpha_power_k = num_fast_power(alpha, n - 2)
    beta_power_k = num_fast_power(beta, n - 2)
    gamma_power_k = num_fast_power(beta, n - 2)
    mat_lambda_power_k = np.array([[alpha_power_k, 0, 0],
                                   [0, beta_power_k, 0],
                                   [0, 0, gamma_power_k], ])

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


def get_roots_of_c_polynomial(a: int, b: int, c: int) -> List[float]:
    mat_a = np.array([[a, b, c],
                      [1, 0, 0],
                      [0, 1, 0]])

    roots = np.roots(np.poly(mat_a))
    # sort in descending order
    roots = np.sort(roots)[::-1]

    return roots


def main():
    for i in range(15):
        print(i, tri_fib_1(i), tri_fib_3(i), tri_fib_5(i))


if __name__ == "__main__":
    main()
