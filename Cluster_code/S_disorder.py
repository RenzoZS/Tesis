import cupy as cp
from system import *
import time
import argparse as ap

parser = ap.ArgumentParser(description='Index manager')
parser.add_argument('-n', type=int, default=0, help='Index name')
args = parser.parse_args()


p = .5
Lx = 32000
Ly = 2048

it_max = int (5e6)
delta_m = 100
m_max = it_max//delta_m
m_count = 0
mp2_count = 0
mp2_max = 16

s = System(Lx,Ly)
S0 = cp.random.choice([0,1.],(Ly,Lx),p=[p,1-p])
s.S = S0
s.set_plane_initial_conditions()
s.gamma = .2
s.D = 1
s.dt = .01
s.beta = 1
S0 = s.S.mean()
p2 = 2

t = cp.zeros(m_max)
u_fft = cp.zeros((mp2_max,s.Ly//2+1))
u_fft1 = cp.zeros((mp2_max,s.Ly//2+1)) 
width = cp.zeros(m_max)
u_sigma = cp.zeros(m_max)
u_sigma1 = cp.zeros(m_max)
u_cm = cp.zeros(m_max)
u_cm1 = cp.zeros(m_max)
I_max = cp.zeros(m_max)

ti = time.time()
while s.t_it+1 < it_max and s.u_cm() < s.Lx*.9:
    if s.t_it % delta_m == 0:
        width[m_count] = s.width()
        u_sigma[m_count] = s.u_sigma()
        u_sigma1[m_count] = s.u_sigma1()
        u_cm[m_count] = s.u_cm()
        u_cm1[m_count] = s.u_cm1()
        I_max[m_count] = s.I_max()
        t[m_count] = s.t
        m_count += 1
    if int(s.t) == p2:
        p2 *= 2
        u_fft[mp2_count] = s.u_fft()
        u_fft1[mp2_count] = s.u_fft1()
        mp2_count += 1
    s.update()
    s.rigid_x()

width = width[:m_count]
u_sigma = u_sigma[:m_count]
u_sigma1 = u_sigma1[:m_count]
u_cm = u_cm[:m_count]
u_cm1 = u_cm1[:m_count]
I_max = I_max[:m_count]
u_fft = u_fft[:mp2_count]
u_fft1 = u_fft1[:mp2_count]
t = t[:m_count]

tf = time.time()
deltat = tf - ti
log = cp.array([p,Lx,Ly,it_max,delta_m,mp2_max,s.gamma,s.D,s.dt,deltat])

cp.savez('data/S0_da_1206_'+str(args.n),t=t,width = width, u_sigma = u_sigma, u_sigma1 = u_sigma1, u_cm = u_cm,u_cm1 = u_cm1, I_max = I_max, u_fft = u_fft, u_fft1 = u_fft1, S0 = S0)
print('Tiempo de realización: {} s'.format(deltat))
