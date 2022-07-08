import cupy as cp
import numpy as np
from system import *
import time
import argparse as ap

parser = ap.ArgumentParser(description='Index manager')
parser.add_argument('-n', type=int, default=0, help='Index name')
parser.add_argument('-Lx',type=int, default = 16384, help='Large of the system')
parser.add_argument('-Ly', type=int, default=2048, help='Width of the system')
parser.add_argument('-it',type=int, default = 1,help = 'Iteraciones sobre desorden' )
args = parser.parse_args()

p = .5
Lx = args.Lx
Ly = args.Ly

it_max = int (3e6)
delta_m = 100
m_max = it_max//delta_m
m_count_min = m_max
mp2_max = 30
it = args.it

s = System(Lx,Ly)
s.gamma = .2
s.D = 1
s.dt = .05

u_fft = cp.zeros((mp2_max,s.Ly//2+1))
u_fft1 = cp.zeros((mp2_max,s.Ly//2+1)) 
width = cp.zeros(m_max)
u_sigma = cp.zeros(m_max)
u_sigma1 = cp.zeros(m_max)
u_cm = cp.zeros(m_max)
u_cm1 = cp.zeros(m_max)
I_max = cp.zeros(m_max)
beta = 0

ti = time.time()
for _ in range(it):
    m_count = 0
    p2 = 2
    mp2_count = 0

    s.set_plane_initial_conditions()
    s.set_dic_beta(p = p)
    beta += s.beta.mean()

    while s.t_it+1 < it_max and s.u_cm() < s.Lx*.9:
        if s.t_it % delta_m == 0:
            width[m_count] += s.width()
            u_sigma[m_count] += s.u_sigma()
            u_sigma1[m_count] += s.u_sigma1()
            u_cm[m_count] += s.u_cm()
            u_cm1[m_count] += s.u_cm1()
            I_max[m_count] += s.I_max()
            m_count += 1
        if s.t_it == p2:
            p2 *= 2
            u_fft[mp2_count] += s.u_fft()
            u_fft1[mp2_count] += s.u_fft1()
            mp2_count += 1
        s.numba_update()
        s.rigid_x()

    if m_count < m_count_min:
        m_count_min = m_count
    
    s.reset()

width = width[:m_count_min]/it
u_sigma = u_sigma[:m_count_min]/it
u_sigma1 = u_sigma1[:m_count_min]/it
u_cm = u_cm[:m_count_min]/it
u_cm1 = u_cm1[:m_count_min]/it
I_max = I_max[:m_count_min]/it
beta = beta/it
u_fft = u_fft[:mp2_count]/it
u_fft1 = u_fft1[:mp2_count]/it
t = cp.arange(m_count_min)*s.dt

tf = time.time()
deltat = tf - ti
log = np.array([Lx,Ly,it_max,delta_m,mp2_max,p,s.gamma,s.D,s.dt,it,deltat,s.date])

cp.savez('data/Ly_{}_it_{}_da_{}'.format(Ly,args.it,args.n),t = t, width = width, u_sigma = u_sigma, u_sigma1 = u_sigma1, u_cm = u_cm, u_cm1 = u_cm1, I_max = I_max, u_fft = u_fft, u_fft1 = u_fft1, beta = beta, log = log)
print('Tiempo de realización: {} s'.format(deltat))