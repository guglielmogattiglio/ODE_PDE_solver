import numpy as np

try:
    import jax 
    import jax.numpy as jnp
    from jax.config import config
    config.update("jax_enable_x64", True)
    is_jax_installed = True
except ImportError as e:
    print('Jax not installed. Falling back to numpy')
    print(e)
    jax = None
    jnp = np
    is_jax_installed = False

from utils import Normalize

      
    
class ODE():
    def get_vector_field(self, *args, **kwargs):
        raise NotImplementedError('This is an abstract class')
    
    def get_init_cond(self, *args, **kwargs):
        raise NotImplementedError('This is an abstract class')
    
class FHN_PDE(ODE):
    def __init__(self, steps, tspan, d_x, d_y, normalization=None):
        self.steps = steps
        self.tspan = tspan
        self.d_x = d_x
        self.d_y = d_y

        d = 2 * (d_x * d_y)  
        mn, mx = np.array([[-1] * d, [1] * d])
        self.normalizer = Normalize(mn, mx, normalization)
        self.d = d
        self.DXX, self.DYY = self._calc_matrices(d_x, d_y)

    @staticmethod
    def _calc_matrices(d_x, d_y):                            
        xspan = [-1,1]                                                  

        dx = (xspan[1]-xspan[0])/(d_x-1)
        dy = (xspan[1]-xspan[0])/(d_y-1)

        z1 = np.ones(d_x)
        Txx = np.diag(-2*z1)
        idxs = np.arange(d_x-1)
        Txx[idxs, idxs+1] = z1[:d_x-1]
        Txx[idxs+1, idxs] = z1[:d_x-1]
        Dxx = (1/(dx**2))*Txx

        z1 = np.ones(d_y)
        Tyy = np.diag(-2*z1)
        idxs = np.arange(d_y-1)
        Tyy[idxs, idxs+1] = z1[:d_y-1]
        Tyy[idxs+1, idxs] = z1[:d_y-1]
        Dyy = (1/(dy**2))*Tyy


        # boundary conditions (periodic)
        Dxx[0,-1] = 1/(dx**2)
        Dxx[-1,0] = 1/(dx**2)
        Dyy[0, -1] = 1/(dy**2)
        Dyy[-1,0] = 1/(dy**2)

        # construct differentiation matrices (using kronecker products)
        DXX = np.kron(np.eye(d_y,d_y),Dxx)
        DYY = np.kron(Dyy,np.eye(d_x,d_x))
        
        return DXX, DYY
    
    @staticmethod
    def _f_jax(t,u,DXX,DYY):
        d = int(u.shape[0]/2)
        u1 = u[:d]
        u2 = u[d:]
        
        a = 2.8E-4
        b = 5E-3
        k = -5E-3
        tau = 0.1
        U = a*(DXX + DYY)@u1 + u1 - (u1**3) - u2 + k*jnp.ones(d)
        V = (1/tau)*( b*(DXX + DYY)@u2 + u1 - u2 )

        return jnp.hstack([U, V])
    
    @staticmethod
    def _f_np(t,u,DXX,DYY):
        d = int(u.shape[0]/2)
        u1 = u[:d]
        u2 = u[d:]
        
        a = 2.8E-4
        b = 5E-3
        k = -5E-3
        tau = 0.1
        U = a*(DXX + DYY)@u1 + u1 - (u1**3) - u2 + k*np.ones(d)
        V = (1/tau)*( b*(DXX + DYY)@u2 + u1 - u2 )

        return np.hstack([U, V])
    
    
    def _get_f(self, use_jax=True):
        if use_jax and is_jax_installed:
            f_jit = jax.jit(self._f_jax)
            f = lambda t, u: f_jit(t, u, self.DXX, self.DYY)
        else:
            f = lambda t, u: self._f_np(t, u, self.DXX, self.DYY)
        return f
    
    
    def get_vector_field(self, use_jax=True):
        f_orig = self._get_f(use_jax)

        def f_normalized(t, u):
            u = self.normalizer.inverse(u)
            out = f_orig(t, u)
            out = out * self.normalizer.get_scale()
            return out
        
        return f_normalized

    def get_init_cond(self, seed=45):
        np.random.seed(seed)
        u0 = np.random.rand(self.d)
        return self.normalizer.fit(u0)
    