import cupy as cp
from datetime import date
from numba import cuda
import numpy as np


@cuda.jit
def numba_forces(S,I,beta,gamma,D,fS,fI):
    y,x = cuda.grid(2)
    Ly,Lx = S.shape
    if x < Lx and y < Ly :
        fS[y,x] = -beta[y,x]*S[y,x]*I[y,x]
        fI[y,x] = beta[y,x]*S[y,x]*I[y,x] - gamma*I[y,x] + D*(I[(y+1)%Ly,x] + I[(y-1+Ly)%Ly,x] + I[y,(x+1)%Lx] + I[y,(x-1+Lx)%Lx] - 4*I[y,x])

@cuda.jit
def euler(X,Y,dt):
    y,x = cuda.grid(2)
    Ly,Lx = X.shape
    if x < Lx and y < Ly :
        X[y,x] = X[y,x] + dt*Y[y,x]

@cuda.jit
def rigid_kernel(X,Y):
    i = cuda.grid(1)
    s = X.size
    if i < s :
        X[i] = 0
        Y[i] = 0  

forces = cp.ElementwiseKernel(
    'raw float64 S, raw float64 I, raw float64 H, float64 beta, float64 gamma, float64 D, float64 vx, float64 vy, int16 Lx, int16 Ly',
    'float64 fS, float64 fI',
    '''
    int x = i % Lx;
    int y = (int) i / Lx;

    fS = - beta * S[i] * I[i];

    float reaction = beta * S[i] * I[i] - gamma * I[i];
    float diffusion = D * (I[(x+1)%Lx + Lx*y] + I[(x-1+Lx)%Lx+Lx*y] + I[x + Lx*((y+1)%Ly)] + I[x + Lx*((y-1+Ly)%Ly)] - 4*I[i]);
    float convective_height = -((H[(x+1)%Lx + Lx*y] - H[(x-1+Lx)%Lx+Lx*y])*(I[(x+1)%Lx + Lx*y] - I[(x-1+Lx)%Lx+Lx*y]) + (H[x + Lx*((y+1)%Ly)] - H[x + Lx*((y-1+Ly)%Ly)])*(I[x + Lx*((y+1)%Ly)] - I[x + Lx*((y-1+Ly)%Ly)]))/4;
    float convective_wind = -(vx*(I[(x+1)%Lx + Lx*y] - I[(x-1+Lx)%Lx+Lx*y]) + vy*(I[x + Lx*((y+1)%Ly)] - I[x + Lx*((y-1+Ly)%Ly)]))/2;

    fI = reaction + diffusion + convective_height + convective_wind;
    ''',
    'forces')


forces_beta = cp.ElementwiseKernel(
    'raw float64 S, raw float64 I, float64 beta, float64 gamma, float64 D, int16 Lx, int16 Ly',
    'float64 fS, float64 fI',
    '''
    int x = i % Lx;
    int y = (int) i/Lx;

    fS = -beta*S[i]*I[i];
    fI = beta*S[i]*I[i] - gamma*I[i] + D*(I[(x+1)%Lx + Lx*y] + I[(x-1+Lx)%Lx + Lx*y] + I[x + Lx*((y+1)%Ly)] + I[x + Lx*((y-1+Ly)%Ly)] - 4*I[i]);
    ''',
    'forces_beta')


forces_v = cp.ElementwiseKernel(
    'raw float64 S, raw float64 I, float64 beta, float64 gamma, float64 D, float64 vx, float64 vy, int16 Lx, int16 Ly',
    'float64 fS, float64 fI',
    '''
    int x = i % Lx;
    int y = (int) i / Lx;

    fS = - beta * S[i] * I[i];

    float reaction = beta * S[i] * I[i] - gamma * I[i];
    float diffusion = D * (I[(x+1)%Lx + Lx*y] + I[(x-1+Lx)%Lx+Lx*y] + I[x + Lx*((y+1)%Ly)] + I[x + Lx*((y-1+Ly)%Ly)] - 4*I[i]);
    float convective_wind = -(vx*(I[(x+1)%Lx + Lx*y] - I[(x-1+Lx)%Lx+Lx*y]) + vy*(I[x + Lx*((y+1)%Ly)] - I[x + Lx*((y-1+Ly)%Ly)]))/2;

    fI = reaction + diffusion + convective_wind;
    ''',
    'forces')


class System:

    def __init__(self,Lx=1024,Ly=1024):
        self.Lx = Lx
        self.Ly = Ly
        self.S = cuda.to_device(np.ones((Ly,Lx)))
        self.I = cuda.to_device(np.zeros((Ly,Lx)))
        self.fS = cuda.to_device(np.zeros((Ly,Lx)))
        self.fI = cuda.to_device(np.zeros((Ly,Lx)))
        self.beta = 0
        self.gamma = 0
        self.D = 0
        self.H = 0
        self.vx = 0 
        self.vy = 0
        self.t = 0
        self.t_it = 0
        self.dt = .01
        self.date = str(date.today())
        self.threadsperblock = (8,64)
        self.blockspergrid = (self.Ly//self.threadsperblock[0],self.Lx//self.threadsperblock[1])

    def __del__(self):
        return None

    def reset(self):
        self.S = cuda.to_device(np.ones((self.Ly,self.Lx)))
        self.I = cuda.to_device(np.zeros((self.Ly,self.Lx)))
        self.fS = cuda.to_device(np.zeros((self.Ly,self.Lx)))
        self.fI = cuda.to_device(np.zeros((self.Ly,self.Lx)))
        self.t = 0
        self.t_it = 0
        self.date = str(date.today())
    
    def set_initial_conditions(self,S0,I0):
        self.S = cuda.to_device(S0)
        self.I = cuda.to_device(I0)

    def set_plane_initial_conditions(self,x0 = 0):
        @cuda.jit
        def set_plane_initial_conditions_kernel(S,I,x0):
            i = cuda.grid(1)
            if i < S.shape[0]:
                S[i,x0] = 0
                I[i,x0] = 1
        set_plane_initial_conditions_kernel[self.blockspergrid,self.threadsperblock](self.S,self.I,x0)
        

    def set_point_initial_conditions(self,x0,y0):
        self.I[x0,y0] = 1
        self.S[x0,y0] = 0
    
    def set_dic_beta(self, beta0 = 1., p = 0):
        self.beta = np.random.rand(self.Ly,self.Lx)
        self.beta[self.beta>p] = 1
        self.beta[self.beta<p] = 0
        self.beta = self.beta*beta0
        self.beta = cuda.to_device(self.beta)

    #def set_smooth_beta(self,beta0 = 1., p = 0 , n = 1):
    #    self.set_dic_beta(beta0,p)
    #    for _ in range(n):
    #        beta_aux = cp.copy(self.beta)
    #        smooth(self.Lx,self.Ly,beta_aux,self.beta)
#
    #def set_dic_smooth_beta(self,beta0 = 1., p = 0 , n = 1):
    #    self.set_dic_beta(beta0,p)
    #    for _ in range(n):
    #        beta_aux = cp.copy(self.beta)
    #        smooth(self.Lx,self.Ly,beta_aux,self.beta)
    #    
    #    beta_mean = cp.mean(self.beta)
    #    self.beta[self.beta>beta_mean] = beta0
    #    self.beta[self.beta<beta_mean] = 0
    #
    #def set_cahn_hilliard(self,beta_mean = 1.,e0 = 2/3):
    #    D = .01
    #    gamma = .5
#
    #    laplacian = cp.ElementwiseKernel(
    #        'raw float32 X, int16 Lx, int16 Ly','float32 Y',
    #        '''
    #        int x = i % Lx;
    #        int y = (int) i/Lx;
    #        Y = X[(x+1)%Lx + Lx*y] + X[(x-1+Lx)%Lx+Lx*y] + X[x + Lx*((y+1)%Ly)] + X[x + Lx*((y-1+Ly)%Ly)] - 4*X[i];
    #        ''',
    #        'laplacian')
#
    #    A = .0001
    #    c = cp.random.choice([-A,A],size=(self.Ly,self.Lx)).astype('float32')
#
    #    t_max = 10000
    #    a = cp.zeros_like(c)
    #    b = cp.zeros_like(c)
    #    d = cp.zeros_like(c)
#
    #    for _ in range(t_max):
    #        laplacian(4*e0*c*(c**2-1),self.Lx,self.Ly,a)
    #        laplacian(c,self.Lx,self.Ly,b)
    #        laplacian(b,self.Lx,self.Ly,d)
    #        c = c + D*(a-gamma*d)
    #    c = (c - c.min())
    #    c = c/c.max()
    #    c = c/c.mean()*beta_mean
#
    #    self.beta = c.astype('float64')

    
    #def set_egg_H(self,k=1,h=1):
    #    Y,X = cp.mgrid[0:self.Ly:1,0:self.Lx:1]
    #    self.H = h*(cp.cos(k*2*cp.pi/(self.Lx-1)*X) + cp.cos(k*2*cp.pi/(self.Ly-1)*Y))/2

    def fft_beta(self):
        return cp.fft.fftshift(cp.fft.fft2(self.beta))
    
    def update(self):
        if isinstance(self.H,cp.ndarray):
            forces(self.S,self.I,self.H,self.beta,self.gamma,self.D,self.vx,self.vy,self.Lx,self.Ly,self.fS,self.fI)
        elif self.vx != 0 or self.vy != 0:
            forces_v(self.S,self.I,self.beta,self.gamma,self.D,self.vx,self.vy,self.Lx,self.Ly,self.fS,self.fI)
        else:
            forces_beta(self.S,self.I,self.beta,self.gamma,self.D,self.Lx,self.Ly,self.fS,self.fI)
        euler[self.blockspergrid,self.threadsperblock](self.S,self.fS,self.dt)
        euler[self.blockspergrid,self.threadsperblock](self.I,self.fI,self.dt)
        self.t += self.dt
        self.t_it += 1
    
    def numba_update(self):
        numba_forces[self.blockspergrid, self.threadsperblock](self.S,self.I,self.beta,self.gamma,self.D,self.fS,self.fI)
        euler[self.blockspergrid,self.threadsperblock](self.S,self.fS,self.dt)
        euler[self.blockspergrid,self.threadsperblock](self.I,self.fI,self.dt)
        self.t += self.dt
        self.t_it += 1
    
    def numba_update2(self):
        x1 = int(self.u_cm() - self.Ly/2)
        x2 = int(self.u_cm() + self.Ly/2)
        if x1 < 0:
            x1 = 0
        if x2 > self.Lx - 1:
            x2 = self.Lx - 1

        numba_forces[self.blockspergrid, self.threadsperblock](self.S[:,x1:x2],self.I[:,x1:x2],self.beta[:,x1:x2],self.gamma,self.D,self.fS[:,x1:x2],self.fI[:,x1:x2])
        self.fS[:,x1] = self.fS[:,x2-1] = self.fI[:,x1] = self.fI[:,x2-1] = 0
        euler[self.blockspergrid,self.threadsperblock](self.S[:,x1:x2],self.fS[:,x1:x2],self.dt)
        euler[self.blockspergrid,self.threadsperblock](self.I[:,x1:x2],self.fI[:,x1:x2],self.dt)
        self.t += self.dt
        self.t_it += 1

    def update2(self):
        x1 = int(self.u_cm() - self.Lx/4)
        x2 = int(self.u_cm() + self.Lx/4)
        if x1 < 0:
            x1 = 0
        if x2 > self.Lx - 1:
            x2 = self.Lx - 1
        
        forces_beta(self.S[:,x1:x2],self.I[:,x1:x2],self.beta[:,x1:x2],self.gamma,self.D,x2-x1,self.Ly,self.fS[:,x1:x2],self.fI[:,x1:x2])
        self.fS[:,x1] = self.fS[:,x2-1] = self.fI[:,x1] = self.fI[:,x2-1] = 0
        euler[self.blockspergrid,self.threadsperblock](self.S[:,x1:x2],self.fS[:,x1:x2],self.dt)
        euler[self.blockspergrid,self.threadsperblock](self.I[:,x1:x2],self.fI[:,x1:x2],self.dt)
        self.t += self.dt
        self.t_it += 1

    
    def get_S1(self):
        return cp.asnumpy(self.S[:,int(self.Lx*.2):int(self.u_cm())-int(self.Lx*.2)].mean())
    
    def get_p(self):
        return cp.asnumpy(cp.count_nonzero(self.beta)/(self.Lx*self.Ly))

    def rigid_x(self):
        rigid_kernel[self.blockspergrid,self.threadsperblock](self.S[:,-1],self.I[:,-1])
    
    def rigid_y(self):
        rigid_kernel[self.blockspergrid,self.threadsperblock](self.S[-1,:],self.I[-1,:])
    
    def u(self):
        return cp.argmax(cp.array(self.I),axis=1)
    
    def u1(self):
        return cp.dot(cp.array(self.I),cp.arange(self.Lx))/(cp.sum(cp.array(self.I),axis=1))
        
    
    def u2(self):
        return cp.dot(cp.array(self.I),cp.arange(self.Lx)**2)/(cp.sum(cp.array(self.I),axis=1))
    
    def u_cm(self):
        return cp.mean(self.u())

    def u_cm1(self):
        return cp.mean(self.u1())
    
    def u_sigma(self):
        return cp.std(self.u())
    
    def u_sigma1(self):
        return cp.std(self.u1())
    
    def width(self):
        return cp.sqrt(self.u2()-self.u1()**2).mean()
    
    def u_fft(self):
        return cp.square(cp.abs(cp.fft.rfft(self.u())))

    def u_fft1(self):
        return cp.square(cp.abs(cp.fft.rfft(self.u1())))
    
    def f_I(self):
        return cp.roll(cp.mean(self.I,axis=0),int(-self.u_cm())+self.Lx//2)
    
    def ff_I(self):
        return cp.mean(self.I,axis=0)
    
    def I_max(self):
        return cp.mean(cp.max(cp.array(self.I),axis=1))