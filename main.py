from numpy import array, transpose, dot, identity, zeros
from scipy import linalg
from jacobi import Jacobi
from seidel import Seidel
import matplotlib.pyplot as plt

#plotar o grafico
def plotar(data1, data2):

    x = array(range(len(data1)))
    plt.plot(x,data1, label='Jacobi', color='orange')
    plt.legend()
    plt.plot(x, data1)
    plt.plot(x, data1, 'k', color='orange')
    
    y = array(range(len(data2)))
    plt.plot(y, data2, label='Seidel', color='blue')
    plt.legend()
    plt.plot(y, data2)
    plt.plot(y, data2, 'k', color='blue')
    
    plt.xlabel('iteracoes')
    plt.ylabel('erro')
    
    plt.show()


def main():
    # primeira instancia do experimento (exemplo do livro)
    A = array([[-1, 0, 0, 0, 1, -1, 0],
               [1, 1, 0, 0, 0, 0, -1],
               [0, -1, -1, 0, 0, 1, 0],
               [0, 0, 1, -1, 0, 0, 0]])

    At = transpose(A)
    K = identity(7)

    AKAt = dot(dot(A, K), At)
    #print(At)
    b = transpose(array([[8, 4, 9, 3]]))

    p = linalg.solve(AKAt, b)

    x0 = transpose([zeros(4)])
    tol = 0.00001
    N = 10000

    #decomposicao LU
    D, L, U = linalg.lu(AKAt)
    print(D)
    print(L)
    print(U)
    
    ly = linalg.solve(L,b)
    ux = linalg.solve(U, ly)

    print('Linalg.solve: ', transpose(linalg.solve(AKAt, b))[0])
    print('sol ldu: ', transpose(ux)[0])

    it, sol, errosJacobi = Jacobi(AKAt, b, x0, tol, N)
    print('jacobi: ', sol, ' it: ', it)
   
    it, sol, errosSeidel = Seidel(AKAt, b, x0, tol, N)
    print('Seidel: ', transpose(sol)[0], ' it: ', it)
    
    print('p: ', transpose(p)[0])
    
    plotar(errosJacobi, errosSeidel)


def teste(): 
    
    # segunda instancia do experimento
    B = array(
        [ #a1  a2 a3 a4 a5 a6 a7 a8 a9 a10a11
        [-1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], #v1
        [1, 1, 0, 0, -1, 1, -1, 0, 0, 0, 0, 0], #v2
        [0, -1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0], #v3
        [0, 0, -1, 0, 0, 0, 0, 0, 1, 1, 0, 0], #v4
        [0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 1, 0], #v5
        [0, 0, 0, 0, 0, -1, 1, 0, 0, 0, -1, 1], #v6
        [0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1], #v7 
    ])

    Bt = transpose(B)
    K = identity(12)

    BKBt = dot(dot(B, K), Bt)

    b = transpose(array([[10, 5, 3, 6, 2, 8, 9]]))

    #decomposicao LU
    D, L, U = linalg.lu(BKBt)
    ly = linalg.solve(L,b)
    ux = linalg.solve(U, ly)
    

    print('Linalg.solve: ',transpose(linalg.solve(BKBt, b))[0]) #linalg.solve
    print('sol ldu: ', transpose(ux)[0])

    x0 = transpose([zeros(7)])
    tol = 0.00001
    N = 10000

    it, sol, errosJacobi = Jacobi(BKBt, b, x0, tol, N)
    print('jacobi: ', sol, ' it: ', it)
   
    it, sol, errosSeidel = Seidel(BKBt, b, x0, tol, N)
    print('Seidel: ', transpose(sol)[0], ' it: ', it)

    plotar(errosJacobi, errosSeidel)

main()  
teste()
