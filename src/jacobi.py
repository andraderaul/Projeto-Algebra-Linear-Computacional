# Método de Jacobi

import numpy as np

def Jacobi(A, b, x0, tol, N):

    A = A.astype('double')
    b = b.astype('double')
    x0 = x0.astype('double')

    n = np.shape(A)[0]
    x = np.zeros(n)
    it = 0
    errs = []

    while it < N:
        it += 1

        #iteracao de Jacobi
        for i in np.arange(n):
            x[i] = b[i]
            for j in np.concatenate((np.arange(0, i), np.arange(i + 1, n))):
                x[i] -= A[i, j] * x0[j]
            x[i] /= A[i, i]

        #erro
        e = np.linalg.norm(x - x0, np.inf)
        errs.append(e)
        #print(e, it)
        #tolerancia
        if (e < tol):
            return it, x, errs

        x0 = np.copy(x)

    raise NameError('num max de iteracoes excedido')
