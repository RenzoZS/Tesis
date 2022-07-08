import cupy as cp
import numpy as np
from system2 import *
import time
import argparse as ap

parser = ap.ArgumentParser(description='Params')
parser.add_argument('-n', type=int, default=0, help='Index name')
parser.add_argument('-Lx', type=int, default=16384, help='Large of the system')
parser.add_argument('-Ly', type=int, default=2048, help='Width of the system')
parser.add_argument('-p', type=float, default=0, help='p value for beta distribution')
parser.add_argument('-beta', type=float, default=1, help='beta value for beta distribution')
parser.add_argument('-dt', type=float, default=.01, help='Time step')
parser.add_argument('-it_max', type=int, default=2e6, help='Max number of iterations')
parser.add_argument('-delta_m', type=int, default=1000, help='Number of iterations between measurements')
parser.add_argument('-gamma', type=float, default=.2, help='gamma value')
parser.add_argument('-D', type=float, default=1, help='D value')
parser.add_argument('-beta_type', type=str, default='da', help='type of beta distribution')
args = parser.parse_args()


p = args.p
Lx = args.Lx
Ly = args.Ly
it_max = args.it_max
delta_m = args.delta_m


m_max = int(it_max//delta_m+1)
m_count = 0

s = System(Lx,Ly)
s.gamma = args.gamma
s.D = args.D
s.dt = args.dt
s.set_plane_initial_conditions()

if args.beta_type == 'da':
    s.set_dic_beta(p = p, beta0 = args.beta)
    beta = cp.mean(cp.array(s.beta))
elif args.beta_type == 'h':
    s.beta = args.beta
elif args.beta_type == 's':
    s.set_smooth_beta(p = p, beta0 = args.beta)
    beta = cp.mean(cp.array(s.beta))
elif args.beta_type == 'dc':
    s.set_dic_beta(p = p, beta0 = args.beta)
    beta = cp.mean(cp.array(s.beta))

u_fft = cp.zeros((m_max,s.Ly//2+1))
u_fft1 = cp.zeros((m_max,s.Ly//2+1)) 
width = cp.zeros(m_max)
u_sigma = cp.zeros(m_max)
u_sigma1 = cp.zeros(m_max)
u_cm = cp.zeros(m_max)
u_cm1 = cp.zeros(m_max)
I_max = cp.zeros(m_max)

ti = time.time()
while s.t_it+1 < it_max and s.u_cm() < s.Lx - s.Ly:
    if s.t_it % delta_m == 0:
        width[m_count] = s.width()
        u_sigma[m_count] = s.u_sigma()
        u_sigma1[m_count] = s.u_sigma1()
        u_cm[m_count] = s.u_cm()
        u_cm1[m_count] = s.u_cm1()
        I_max[m_count] = s.I_max()
        u_fft[m_count] = s.u_fft()
        u_fft1[m_count] = s.u_fft1()
        m_count += 1

    s.numba_update()
    s.rigid_x()

width = width[:m_count]
u_sigma = u_sigma[:m_count]
u_sigma1 = u_sigma1[:m_count]
u_cm = u_cm[:m_count]
u_cm1 = u_cm1[:m_count]
I_max = I_max[:m_count]
u_fft = u_fft[:m_count]
u_fft1 = u_fft1[:m_count]
t = np.arange(m_count)*(delta_m*s.dt)

tf = time.time()
deltat = tf - ti

log = np.array([p,Lx,Ly,it_max,delta_m,s.gamma,s.D,s.dt,deltat,s.date])

cp.savez('data/1906_da_'+str(args.n), t = t, width = width, u_sigma = u_sigma, u_sigma1 = u_sigma1, u_cm = u_cm, u_cm1 = u_cm1, I_max = I_max, u_fft = u_fft, u_fft1 = u_fft1, beta = beta, log = log)

print('Tiempo de realización: {} s'.format(deltat))
