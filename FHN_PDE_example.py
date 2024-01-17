import os
base_dir = r'' # Insert path to directory containing solver.py
os.chdir(base_dir)

from solver import Solver
import numpy as np  

s = Solver(steps=1e5, tspan=[0, 5000], d_x=10, d_y=10, use_jax=True)
out = s.run()

# Save output
path = f'fhn_pde_traj_{out.shape[0]}.csv'
np.savetxt(os.path.join(base_dir, path), out)
