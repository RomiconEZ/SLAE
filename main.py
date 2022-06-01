# This is a sample Python script.
import sys
import functools as ft
import numpy as np
import random
import math
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def get_L(m):
    L = m.copy()
    for i in range(L.shape[0]):
            L[i, i] = 1
            L[i, i+1 :] = 0
    return np.matrix(L)


def get_U(m):
    U = m.copy()
    for i in range(1, U.shape[0]):
        U[i, :i] = 0
    return np.matrix(U)


def gauss(x):
    n=len(x)
    if abs(x[0])>1e-8:
        t=x[1:n]/x[0] # Находим коэффициенты по столбцу для обнуления стобца под элементом
    else:
        t=np.zeros((n-1,))
    return t


def gauss_app(C,t,k,raz):
    n=C.shape[0]
    for i in range(1,n):
        C[i,:]= C[i,:]-np.multiply(t[i-1],C[0,:])# Вычитаем строку, умноженную на коэф. из всех строк ниже ее
    return C

def perestanovka_strok_stolbcov(A,k,raz,kolvo_nech_perest):
    ind = np.unravel_index(np.argmax(abs(A[k:,k:])), A[k:,k:].shape)# нахожу индекс максимального элемента в урезанной матрице A
    ind=list(ind)
    ind[0] += k# прибавляю k, чтобы найти номера строки и столбца в основной матрице A
    ind[1] += k

    Mp = np.eye(raz)
    if ind[0] != 0: # Если были перестановки строк в матрице
        Mp[k, k] = 0
        Mp[ind[0], ind[0]] = 0
        Mp[k, ind[0]] = 1
        Mp[ind[0], k] = 1
        if kolvo_nech_perest==True:
            kolvo_nech_perest=False
        else:
            kolvo_nech_perest = True

    Mq = np.eye(raz)

    if ind[1] != 0: # Если были перестановки столбцов в матрице
        Mq[k, k] = 0
        Mq[ind[1], ind[1]] = 0
        Mq[k, ind[1]] = 1
        Mq[ind[1], k] = 1
        if kolvo_nech_perest == True:
            kolvo_nech_perest = False
        else:
            kolvo_nech_perest = True

    return [Mp,Mq,kolvo_nech_perest]

def IsklyucheniyeGaussaSVneshnimProizvedeniyem(A):
    n=A.shape[0]
    Mp = np.eye(n)# Общая матрица перестановки строк
    Mq = np.eye(n)# Общая матрица перестановки столбцов
    kolvo_nech_perest = True
    # False - нечетное количество нечетных перестановок
    # True - четное количество нечетных перестановок


    # Подсчитывается количетство нечетных перестановок в матрице, так как для перемещения двух произвольных
    # строк/столбцов в матрице необходимо 2(s-k)-1 транспозиция, соответственно нечетное количество
    # нечетных перестановок даст нам нечетное количество транспозиция в целом, тогда определитель поменяет знак


    for k in range(0,n):
        M_p_q_per = perestanovka_strok_stolbcov(A,k,n,kolvo_nech_perest)

        kolvo_nech_perest = M_p_q_per[2]

        # Переставляем строки и столбцы в матрице A
        A = np.matmul(M_p_q_per[0], A)
        A = np.matmul(A, M_p_q_per[1])

        # Меняем общиие матрицы перестановок для A
        Mp = np.matmul(M_p_q_per[0], Mp)
        Mq = np.matmul(Mq,M_p_q_per[1])

        t=gauss(A[k:n,k])
        A[k+1:n,k]=t
        A[k:n,k+1:n]=gauss_app(A[k:n,k+1:n],t,k,n)

    return [A,Mp,Mq,int(kolvo_nech_perest)]

def solve(Check,A, b, LU, P ,Q):
    L = get_L(LU)
    U = get_U(LU)
    b_init=b
    b = np.matmul(P, b)
    y = np.empty((b.shape[0], 1))
    x = np.empty((b.shape[0], 1))
    for i in range(0, b.shape[0]):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]

    if Check[0]==False:
        for i in range(b.shape[0] - 1, -1, -1):
            x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    else:
        # for j in range (0,(A.shape[0]-Check[1])-1):
        #     x[j]=0
        for i in range(b.shape[0] - 1 - (A.shape[0]-Check[1]), -1, -1):
            x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]

    x_init=np.matmul(Q,x)
    if not np.isclose(np.matmul(A, x_init), b_init, rtol=1e-05, atol=1e-08, equal_nan=False).all():
      print('Неправильный ответ')
    return x_init

def obratnay_matrica(A):
    n = A.shape[0]
    A_obr = np.eye(n)
    LU, P, Q, kolvo_nech_perest = IsklyucheniyeGaussaSVneshnimProizvedeniyem(A.copy())
    for i in range(0, n):
        e = np.zeros((n,1))
        e[i,0]=1
        A_obr_i=solve(False,A, e, LU, P, Q)
        A_obr[:,i]=np.ravel(A_obr_i)
    return A_obr

def det(A,kolvo_nech_perest):
    if kolvo_nech_perest==True:
        return ft.reduce(lambda a, b: a*b, np.diag(get_U(IsklyucheniyeGaussaSVneshnimProizvedeniyem(A)[0])))
    else:
        return -(ft.reduce(lambda a, b: a*b, np.diag(get_U(IsklyucheniyeGaussaSVneshnimProizvedeniyem(A)[0]))))
def Norma_Matrix(A):
    Norma=0
    if len(A.shape)>1:
        for i in range(0,A.shape[0]):
            Norma+=np.max(abs(A[i,:]))
    else:
        for i in range(0, A.shape[0]):
            Norma+=A[i]
    return Norma


def householder_reflection(A):
    (r, c) = np.shape(A)
    Q = np.identity(r) #квадратная матрица с единицами на главной диагонали
    R = np.copy(np.float64(A))
    for cnt in range(r - 1): #cnt - count
        #v - вектор нормали
        x = R[cnt:, cnt]
        e = np.zeros_like(x) #вектор со всеми нулями на подобии x
        e[0] = np.linalg.norm(x)
        #e - /alpha * z
        u = x - e
        # u - (y-/alpha * z)
        if not np.isclose(u, np.zeros((1, u.shape[0])), rtol=1e-08, atol=1e-10, equal_nan=False).all():
            v = u / np.linalg.norm(u) #делаем единичную длину
        else:
            v=u
        Q_cnt = np.identity(r)
        Q_cnt[cnt:, cnt:] -= 2.0 * np.outer(v, v) #U = E-2ww^T
        R = np.dot(Q_cnt, R)  # R=H(n-1)*...*H(2)*H(1)*A
        #Q = U
        Q = np.dot(Q, Q_cnt)  # Q = H (N-1) * ... * H (2) * H (1) H - матрица саморегулятора
    return (Q, R)

def solve_QR(A, b, Q, R):
    b_init=b.copy()
    b = np.matmul(np.transpose(Q), b)
    y = np.empty((b.shape[0], 1))
    for i in range(b.shape[0]-1, -1, -1):
        y[i] = (b[i] - np.dot(R[i, i + 1:], y[i + 1:])) / R[i, i]
    if not np.isclose(np.matmul(A, y), b_init, rtol=1e-05, atol=1e-08, equal_nan=False).all():
        print('Неправильный ответ')
    return y

def degenerate_check_and_rang(U):
    kolvo_nul=0
    for i in range(0,U.shape[0]):
        if abs(U[i,i]) < 1e-10:
            kolvo_nul+=1
    if kolvo_nul!=0:
        return [True,U.shape[0]-kolvo_nul]
    else:
        return [False,U.shape[0]]

def Proverka_na_NE_nul_elementov_matrix(A):
    return np.isclose(A, 0, rtol=1e-08, atol=1e-08, equal_nan=False).any()


def Jacobi_SLAE(A,b):
    if Proverka_na_NE_nul_elementov_matrix(A):
        x_new = b.copy()
        A_extend = np.concatenate((A.copy(), b.copy()), axis=1)
        for i in range(0, A_extend.shape[0]):
            a_i_i = A_extend[i, i]
            A_extend[i, 0:A_extend.shape[1]-1] = -A_extend[i, 0:A_extend.shape[1]-1] / a_i_i
            A_extend[i, A_extend.shape[1]-1]= A_extend[i, A_extend.shape[1]-1]/a_i_i
            A_extend[i, i] = 0

        q=Norma_Matrix(A_extend[:,0:A_extend.shape[1]-1])
        # x = B * x + C
        norm_C = Norma_Matrix(A_extend[:,A.shape[1]])
        apriornay_ocenka=math.log((1-q)*(1e-8)/norm_C, q)
        aposteriornay_ocenka=0
        while (not np.isclose(np.matmul(A, x_new), b, rtol=1e-08, atol=1e-08, equal_nan=False).all()):
            x=x_new.copy()
            for i in range(0, b.shape[0]):
                x_new[i] = A_extend[i, A_extend.shape[1]-1].copy()
                for j in range(0, A_extend.shape[1]-1):
                    x_new[i] += x[j]*A_extend[i, j]
            aposteriornay_ocenka+=1
        print (f'Априорная оценка: {int(apriornay_ocenka)+1}, апостериорная оценка: {int(aposteriornay_ocenka)+1}')
        return x_new
    else: return -1

def Zeidel_SLAE(A,b):
    if Proverka_na_NE_nul_elementov_matrix(A):
        x_new = b.copy()
        A_extend = np.concatenate((A.copy(), b.copy()), axis=1)
        for i in range(0, A_extend.shape[0]):
            a_i_i = A_extend[i, i]
            A_extend[i, 0:A_extend.shape[1]-1] = -A_extend[i, 0:A_extend.shape[1]-1] / a_i_i
            A_extend[i, A_extend.shape[1]-1]= A_extend[i, A_extend.shape[1]-1]/a_i_i
            A_extend[i, i] = 0

        q=Norma_Matrix(A_extend[:,0:A_extend.shape[1]-1])
        # x = B * x + C
        norm_C = Norma_Matrix(A_extend[:,A.shape[1]])
        apriornay_ocenka=math.log((1-q)*(1e-8)/norm_C, q)
        aposteriornay_ocenka=0
        while (not np.isclose(x_new, x, rtol=1e-08, atol=1e-08, equal_nan=False).all()):
            x=x_new.copy()
            for i in range(0, b.shape[0]):
                x_new[i] = A_extend[i, A_extend.shape[1]-1].copy()
                for j in range(0, i):
                    x_new[i] += x_new[j]*A_extend[i, j]
                for j in range(i, A_extend.shape[1]-1):
                    x_new[i] += x[j]*A_extend[i, j]
            aposteriornay_ocenka+=1
        print (f'Априорная оценка: {int(apriornay_ocenka)+1}, апостериорная оценка: {int(aposteriornay_ocenka)+1}')
        return x_new
    else: return -1

def Random_Matrix():
    n = np.random.randint(2, 4)
    A = np.random.sample((n, n))
    b = np.random.sample((n, 1))
    return (A,b,n)
def Random_Matrix_with_Diag_dom():
    n = np.random.randint(3, 5)
    A = np.random.sample((n, n))
    for i in range(0, n):
        A[i, i] = np.random.randint(n+1,n*n)
    while np.linalg.det(A)<=0:
        A[0,0] = np.random.randint(n+1,n*n)
    b = np.random.sample((n, 1))
    return (A,b,n)

#A,b,n = Random_Matrix()
#A,b,n = Random_Matrix_with_Diag_dom()
#A[:,2]=A[:,1]-A[:,0]*2
#A[:,3]=3*A[:,1]-A[:,0]
A=np.array([[1,0,1],[2,0,1],[0,0,0]])
n=3
#b=np.matmul(A,b)
b=np.array([[1],[0],[5]])

print("**********************************************")
print("LU - разложение произвольной матрицы A")
print("Матрица A")
print(A)
LU, P, Q, kolvo_nech_perest=IsklyucheniyeGaussaSVneshnimProizvedeniyem(A.copy())
U=get_U(LU)
L=get_L(LU)
Check=degenerate_check_and_rang(U)
print("Матрица L")
print(L)
print("Матрица U")
print(U)
print("Матрица L*U")
print(np.matmul(L,U))
print("Матрица A с переставленными строками и столбцами")
print(np.matmul(np.matmul(P, A), Q))

if (Check[0] == True):
    print(f"Матрица A вырождена с рангом {Check[1]}")
print("**********************************************")
print("                                              ")
print("**********************************************")
print("Проверка системы Ax=b на совместимость")
if (Check[0] == True):
    print("Матрица A расширенная")
    A_extend = np.concatenate((A, b), axis=1)
    print(A_extend)
    b_=np.matmul(P,b)
    y = np.empty((b_.shape[0], 1))
    for i in range(0, b_.shape[0]):
        y[i] = (b_[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    j=y.shape[0]-1
    nul_elem_v_y=0
    while  (np.isclose(y[j], 0., rtol=1e-06, atol=1e-06, equal_nan=False)==True)&(j>=0):
        nul_elem_v_y+=1
        j-=1
    Rang_A_extend=y.shape[0]-nul_elem_v_y
    # A_extend_square = np.concatenate((A_extend, np.zeros((1, n+1)) ), axis=0)
    # LU_A_ext, P_A_ext, Q_A_ext, kolvo_nech_perest_A_ext=IsklyucheniyeGaussaSVneshnimProizvedeniyem(A_extend_square.copy())
    # U_A_ext = get_U(LU_A_ext)
    # L_A_ext = get_L(LU_A_ext)
    # Check_A_ext = degenerate_check_and_rang(U_A_ext)
    print(f"Ранг матрица A расширенная: {Rang_A_extend}")
    if Check[1]!=Rang_A_extend:
        print(f"Система несовместна с рангами {Check[1]} у матрицы A")
        print(f"и {Rang_A_extend} у матрицы A расширенной")
    else:
        print(f"Частное решение системы Ax=b")
        print(solve(Check, A.copy(), b.copy(), LU, P, Q))

print("**********************************************")
print("                                              ")
print("**********************************************")
if (Check[0] != True):
    print("Решение системы Ax=b через LU-разложение")
    print("Матрица A:")
    print(A)
    print("Вектор b:")
    print(b)
    print("Ответ:")
    print(solve(Check,A.copy(),b.copy(),LU, P, Q))
print("**********************************************")
print("                                              ")
print("**********************************************")
print("Определитель матрицы A")
if (Check[0] != True):
    print(det(A.copy(),kolvo_nech_perest))
print("**********************************************")
print("                                              ")
print("**********************************************")
print("Обратная матрица к A")
if (Check[0] != True):
    print(obratnay_matrica(A.copy()))
print("**********************************************")
print("                                              ")
print("**********************************************")
print("Число обусловленности A")
if (Check[0] != True):
    print(np.max(abs(A))*np.max(obratnay_matrica(abs(A.copy()))))
print("**********************************************")
print("                                              ")
print("**********************************************")
print("Решение системы Ax=b через QR-разложение")
print("Матрица A:")
print(A)
print("Вектор b:")
print(b)
Q,R = householder_reflection(A.copy())
print("Матрица Q:")
print(Q)
print("Матрица R:")
print(R)
print("Ответ:")
if (Check[0] != True):
    print(solve_QR(A,b,Q,R))
print("**********************************************")
print("                                              ")
print("**********************************************")
print("Решение системы Ax=b методом Якоби")
print("Матрица A:")
print(A)
print("Вектор b:")
print(b)
print("Ответ:")
print(Jacobi_SLAE(A,b))
print("**********************************************")
print("                                              ")
print("**********************************************")
print("Решение системы Ax=b методом Зейделя")
print("Матрица A:")
print(A)
print("Вектор b:")
print(b)
print("Ответ:")
print(Zeidel_SLAE(A,b))