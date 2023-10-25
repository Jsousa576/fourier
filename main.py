import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import math
import warnings
from scipy.interpolate import make_interp_spline

x = np.array([0, 0.0013, 0.0047, 0.0073, 0.0099, 0.0132, 0.0172, 0.0212, 0.0252, 0.0284, 0.0306, 0.0318, 0.0337, 0.0356, 0.037,
              0.0383, 0.0391, 0.0403, 0.0417, 0.0423, 0.0437, 0.0452, 0.0476, 0.0515, 0.0555, 0.0594, 0.0634, 0.0673, 0.0713,
              0.0753, 0.0792, 0.0832, 0.0871, 0.0911, 0.0951, 0.099, 0.103, 0.1069, 0.1109, 0.1149, 0.1188, 0.1228, 0.1267,
              0.1307, 0.1347, 0.1386, 0.1426, 0.1465, 0.1505, 0.1545, 0.1578, 0.1607, 0.1634, 0.1651, 0.1669, 0.1684, 0.17,
              0.1716, 0.173, 0.1739, 0.1749, 0.1759, 0.1762, 0.1775, 0.1789, 0.1799, 0.1809, 0.1828, 0.1838, 0.1855, 0.1877,
              0.1893, 0.1914, 0.194, 0.1973, 0.2013, 0.2053, 0.2092, 0.2132, 0.2168, 0.2201, 0.2232, 0.2257, 0.2274, 0.2291,
              0.231, 0.2329, 0.2343, 0.2356, 0.2369, 0.2383, 0.2396, 0.2409, 0.2419, 0.2435, 0.2445, 0.2461, 0.2475, 0.2488,
              0.2501, 0.2515, 0.2534, 0.2553, 0.2571, 0.259, 0.2608, 0.2629, 0.2647, 0.2672, 0.2689, 0.2719, 0.2736, 0.2765,
              0.2802, 0.2835, 0.2871, 0.291, 0.295, 0.299, 0.3029, 0.3069, 0.3108, 0.3148, 0.3184, 0.3217, 0.3252, 0.328,
              0.3306, 0.3333, 0.3362, 0.3382, 0.3402, 0.3422, 0.3442, 0.3458, 0.3472, 0.3492, 0.350, 0.350])

y = np.array([0, 0.7201, 1.7988, 2.5799, 3.2634, 4.0011, 4.5652, 4.8581, 4.5736, 3.8817, 3.1169, 2.4757, 1.6252, 0.7573, -0.0564,
              -0.9189, -1.879, -2.9747, -4.2028, -5.1011, -5.9343, -6.902, -7.694, -8.4425, -8.8873, -9.0392, -9.0284, -8.8548,
              -8.5402, -8.1605, -7.6506, -7.1732, -6.7067, -6.3596, -6.0558, -5.9039, -5.9582, -6.186, -6.5223, -6.8912, -7.2817,
              -7.618, -7.8567, -8.0411, -8.0954, -8.052, -7.8675, -7.5638, -7.0973, -6.4463, -5.7358, -5.01, -4.2223, -3.4868,
              -2.6406, -1.7423, -0.7236, 0.2908, 1.4191, 2.3521, 3.6539, 4.9124, 5.8671, 6.876, 7.9826, 8.9807, 9.9354, 10.9899,
              11.8231, 12.7214, 13.6736, 14.5245, 15.3544, 16.1518, 16.9003, 17.4319, 17.6815, 17.6272, 17.3235, 16.8353,
              16.0672, 15.1103, 14.1013, 13.3528, 12.5782, 11.7255, 10.7816, 9.9842, 9.2031, 8.3894, 7.6083, 6.7133, 5.8671,
              4.9623, 4.0282, 3.1874, 2.2414, 1.3594, 0.5783, -0.154, -1.0393, -1.9658, -2.9009, -3.7775, -4.7431, -5.5177,
              -6.3658, -7.1841, -7.9912, -8.7072, -9.56, -10.1458, -11.0029, -11.8317, -12.5347, -13.1835, -13.7042, -14.0622,
              -14.225, -14.149, -13.8236, -13.292, -12.5868, -11.7927, -10.9985, -10.1621, -9.202, -8.3557, -7.477, -6.6354,
              -5.7585, -5.0143, -4.2006, -3.3002, -2.5299, -1.7163, -0.8983, -0.0455, 0])

#INPUTS
grau_poli = 6               #Grau do polinômio
T = 0.35                    #Período
w = 2 * np.pi / T           #Frequência
t = sp.symbols('t') 
termos_avaliados = [0,20]#,2,4,10,14,16,20,22,24]
keq = 300 #N/m
meq = 5 #kg
ceq = 5 #Ns/m

#Eliminar aviso de erros do polyfit
warnings.filterwarnings('ignore', category=np.RankWarning)

for N in termos_avaliados:
    intervals = [0, 0.0099, 0.0172, 0.0212, 0.0284, 0.0318, 0.037, 0.0383, 0.0417, 0.0423, 0.0452, 0.0594, 0.0792, 
                0.1188, 0.1347, 0.1505, 0.1607, 0.1716, 0.1759, 0.1893, 0.1973, 0.2092, 0.2168, 0.2232, 
                0.2343, 0.2419, 0.2501, 0.2608, 0.2736, 0.291, 0.3029, 0.3148, 0.3252, 0.3362, 0.350]

    intervalx = [[] for _ in range(len(intervals) - 1)]
    intervaly = [[] for _ in range(len(intervals) - 1)]

    for i in range(len(x)):
        for j in range(len(intervals) - 1):
            if intervals[j] <= x[i] < intervals[j + 1]:
                intervalx[j].append(x[i])
                intervaly[j].append(y[i])
                break

    equations = []

    for i in range(len(intervalx)):

        coeffs = np.polyfit(intervalx[i], intervaly[i], deg=grau_poli)
        if grau_poli == 6:
            equations.append(coeffs[0] * t ** 6 + coeffs[1] * t ** 5 + coeffs[2] * t ** 4 + coeffs[3] * t ** 3 + coeffs[4] * t ** 2 + coeffs[5] * t  + coeffs[6])
        elif grau_poli == 5:
            equations.append(coeffs[0] * t ** 5 + coeffs[1] * t ** 4 + coeffs[2] * t ** 3 + coeffs[3] * t ** 2 + coeffs[4] * t + coeffs[5])
        elif grau_poli == 4:
            equations.append(coeffs[0] * t ** 4 + coeffs[1] * t ** 3 + coeffs[2] * t ** 2 + coeffs[3] * t + coeffs[4])
        elif grau_poli == 3:
            equations.append(coeffs[0] * t ** 3 + coeffs[1] * t ** 2 + coeffs[2] * t + coeffs[3])
        elif grau_poli == 2:
            equations.append(coeffs[0] * t ** 2 + coeffs[1] * t + coeffs[2])
        elif grau_poli == 1:
            equations.append(coeffs[0] * t + coeffs[1])

    a0_vet = []

    for i in range(len(equations)):
        a0_vet.append(sp.integrate(equations[i], (t, intervals[i], intervals[i + 1])))

    a0 = 2 / T * sum(a0_vet)

    # Cálculo ak
    ak = []
    bk = []
    for i in range(N):
        a = []
        b = []
        for j in range(len(equations)):
            a.append(sp.integrate(equations[j] * sp.cos((i + 1) * w * t), (t, intervals[j], intervals[j + 1])))
            b.append(sp.integrate(equations[j] * sp.sin((i + 1) * w * t), (t, intervals[j], intervals[j + 1])))

        ak.append(2 / T * round(sum(a), 4))     # Termos ak
        bk.append(2 / T * round(sum(b), 4))     # Termos bk

    # Cálculo função final
    we = [0]
    F0 = [a0 * 1 / 2]
    result = []
    for i in range(N):
        result.append(ak[i] * sp.cos((i + 1) * w * t) + bk[i] * sp.sin((i + 1) * w * t))
        F0.append(ak[i])
        F0.append(bk[i])
        we.append((i + 1) * sp.pi)
        we.append((i + 1) * sp.pi)

    f = a0 * 1 / 2 + sum(result)                 # Função

    ccri = 2*np.sqrt(keq*meq)
    omegan = round(math.sqrt(keq/meq), 4)
    zeta = round(math.sqrt(ceq/ccri),4)

    uest = []
    for i in range(len(F0)):
        uest.append(F0[i]/keq)

    beta = []
    for i in range(len(we)):
        beta.append(round(we[i]/omegan,4))

    M = []
    for i in range(len(beta)):
        M.append(round(1/(math.sqrt(((1-beta[i]**2)**2) + (2*beta[i]*zeta)**2 )),4))

    phi = []
    for i in range(len(beta)):
        phi.append(round(math.atan((2*beta[i]*zeta)/(1-beta[i]**2)),4))

    # Pontos Fourier
    xfourier = list(np.linspace(0, T, 100))           

    yf = []
    for i in range(len(xfourier)):
        yf.append(f.subs(t, xfourier[i]))

    u=[]
    for i in range(len(F0)):
        if i == 0:
            u.append(uest[i] * M[i] * sp.cos(we[i] * t - phi[i]))
        else:
            if i % 2 == 0:
                u.append(uest[i] * M[i] * sp.sin(we[i] * t - phi[i]))
            else:
                u.append(uest[i] * M[i] * sp.cos(we[i] * t - phi[i]))


    # Erro Quadrático
    erro = []
    for i in range(len(x)):
        erro.append((y[i] - f.subs(t, x[i])) ** 2)

    erro_quad = sum(erro) / len(x)

    print(f'Erro Quadrático {N} termos: {erro_quad * 100}')

    # Plot Comparação
    plt.figure(1)
    plt.plot(xfourier, yf, label = f'N = {N}')
    plt.legend()

# Pontos Deslocamento (Suave)
u_perm = sum(u)
y_desloc = []
x_desloc = np.linspace(0, 5*T, 100)
for i in range(len(x_desloc)):
    y_desloc.append(u_perm.subs(t, x_desloc[i]))
X_Y_Spline = make_interp_spline(x_desloc, np.array(y_desloc))
X_ = np.linspace(x_desloc.min(), x_desloc.max(), 500)
Y_ = X_Y_Spline(X_)

plt.figure(2)
plt.plot(X_, Y_)
plt.xlabel('Tempo [s]')
plt.ylabel('Deslocamento [m]')
plt.axhline(0, color = 'black')
plt.legend()
plt.grid(True)

plt.figure(3)
plt.plot(x, y, color = 'blue', label = 'Função Original')
plt.plot(xfourier, yf, color = 'red', label = f'Fourier {N} termos')
plt.legend()
plt.show()