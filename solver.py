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

from systems import FHN_PDE
    
class RK():
    def __init__(self, t, u0, f, method):
        if method == 'RK1':
            a = np.array([[0]]);
            b = np.array([[1]]); 
            c = np.array([0]);
        elif method == 'RK2':
            a = np.array([[0,0],[0.5,0]])
            b = np.array([[0,1]])
            c = np.array([0,0.5])
        elif method == 'RK4':  #classic fourth-order method
            a = np.array([[0,0,0,0],[0.5,0,0,0],[0,0.5,0,0],[0,0,1,0]])
            b = np.array([[1/6,1/3,1/3,1/6]])
            c = np.array([0,0.5,0.5,1])
        elif method == 'RK8': #Cooper-Verner eigth-order method (again there are many)
            s = np.sqrt(21);
            a = np.array([[0,0,0,0,0,0,0,0,0,0,0],[1/2,0,0,0,0,0,0,0,0,0,0],[1/4,1/4,0,0,0,0,0,0,0,0,0],[1/7,(-7-3*s)/98,(21+5*s)/49,0,0,0,0,0,0,0,0],[(11+s)/84,0,(18+4*s)/63,(21-s)/252,0,0,0,0,0,0,0],[(5+s)/48,0,(9+s)/36,(-231+14*s)/360,(63-7*s)/80,0,0,0,0,0,0],[(10-s)/42,0,(-432+92*s)/315,(633-145*s)/90,(-504+115*s)/70,(63-13*s)/35,0,0,0,0,0],[1/14,0,0,0,(14-3*s)/126,(13-3*s)/63,1/9,0,0,0,0],[1/32,0,0,0,(91-21*s)/576,11/72,(-385-75*s)/1152,(63+13*s)/128,0,0,0],[1/14,0,0,0,1/9,(-733-147*s)/2205,(515+111*s)/504,(-51-11*s)/56,(132+28*s)/245,0,0],[0,0,0,0,(-42+7*s)/18,(-18+28*s)/45,(-273-53*s)/72,(301+53*s)/72,(28-28*s)/45,(49-7*s)/18,0]])
            b = np.array([[1/20,0,0,0,0,0,0,49/180,16/45,49/180,1/20]])
            c = np.array([0,1/2,1/2,(7+s)/14,(7+s)/14,1/2,(7-s)/14,(7-s)/14,1/2,(7+s)/14,1])
        else:
            raise NotImplementedError('Only RK1, RK2, RK4 and RK8 are implemented')
        
        self.a = a
        self.b = b
        self.c = c
        self.t = t
        self.u0 = u0
        self.f = f

    def run(self, use_jax=True):
        if use_jax and is_jax_installed:
            res = self._RK_jax_(jnp.array(self.t), self.u0, self.f, 
                                jnp.array(self.a), jnp.array(self.b), jnp.array(self.c))
        else:
            res = self._RK_numpy_(self.t, self.u0, self.f, self.a, self.b, self.c)
        return np.array(res)
            

    @staticmethod
    def _RK_jax_(t, u0, f, a, b, c):
        u = jnp.zeros((u0.shape[0], t.shape[0]))
        u = u.at[:,0].set(u0)
        dim = u0.shape[0]
        S = b.shape[-1]
        
        def inner_inn_loop(j, carry):
            temp, i, k = carry
            return [temp + a[i,j] * k[:,j], i, k]
        
        def inner_loop(i, carry):
            n, k, u, h = carry
            temp = jnp.zeros(dim)
            temp, _, _ = jax.lax.fori_loop(0, i, inner_inn_loop, [temp, i, k])
            return [n, k.at[:,i].set(h*f(t[n]+c[i]*h, u[:,n]+temp)), u, h]
        
        def outer_loop(n, u):
            h = t[n+1] - t[n]
            k = jnp.zeros((dim,S))
            k = k.at[:,0].set(h*f(t[n], u[:,n]))
            _, k, _, _ = jax.lax.fori_loop(1, S, inner_loop, [n, k, u, h])
            return u.at[:, n+1].set(u[:,n] + jnp.sum(b*k, 1))
            
        u = jax.lax.fori_loop(0, t.shape[0]-1, outer_loop, u)
        return u.T
        # return temp, k, f1
    # RK_jax_ = jax.jit(RK_jax_, static_argnums=(2,)) 

    @staticmethod
    def _RK_numpy_(t, u0, f, a, b, c):        
        u = np.zeros((len(u0), len(t)))
        u[:,0] = u0
        
        for n in range(len(t)-1):
            
            # iterate over runge kutta 
            h = t[n+1] - t[n]
            dim = len(u0)
            S = b.shape[-1]
            k = np.zeros((dim,S))
            k[:,0] = h*f(t[n], u[:,n])
            
            # calculate the coefficients k
            for i in range(1,S):
                temp = np.zeros(dim)
                for j in range(0, i):
                    temp = temp + a[i,j] * k[:,j]
                k[:,i] = h*f(t[n]+c[i]*h, u[:,n]+temp)
                
            # calculate the final solution
            u[:,n+1] = u[:,n] + np.sum(b*k, 1)
            
        return u.T

class Solver():
    '''
    Class for intergating initial value problems for implemented ODEs and PDEs.'''

    # TODO: pass ODE/PDE specific parameters as *args **kwargs. Update relevant ODE classes
    def __init__(self, ode='FHN', rk='RK8', steps=1e5, tspan=[0, 100], d_x=15, d_y=15, normalization='-11', use_jax=True):
        '''
        ode: str, name of the ODE system
        rk: str, order of the Runge-Kutta method. Either 'RK1', 'RK2', 'RK4' or 'RK8'.
        steps: int, number of discretization steps for the RK solve.r.
        tspan: list, time range for integration, given as [t_0, T].
        d_x: int, number of discretization points in the x direction.
        d_y: int, number of discretization points in the y direction.
        normalization: str, normalization type. Either 'identity' or '-11'.
        use_jax: bool, whether to use Google JAX for vector field evaluation.'''
        
        if ode != 'FHN':
            raise NotImplementedError('Only FHN is implemented')
        
        self.Nf = int(steps)
        self.tspan = tspan
        self.d_x = d_x  
        self.d_y = d_y

        ode = FHN_PDE(steps, tspan, d_x, d_y, normalization)
        self.f = ode.get_vector_field(use_jax)
        u0 = ode.get_init_cond()
        self.rk_solver = RK(np.linspace(tspan[0], tspan[-1], num=self.Nf+1), u0, self.f, rk)
        self.use_jax = use_jax
        self.u0 = u0

    def run(self):
        return self.rk_solver.run(self.use_jax)
    

