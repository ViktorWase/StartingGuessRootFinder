import numpy as np
from random import random

def pso(dims, inds, max_iter, err_func, w=0.95, theta_p=0.1, theta_g=0.1):
   positions = np.random.rand(inds, dims)

   best_known_positions = np.copy(positions)
   errors = [err_func(best_known_positions[i,:]) for i in range(inds)]
   best_err = min(errors)
   best_err_idx = errors.index(best_err)
   global_best_pos = np.copy(best_known_positions[best_err_idx,:])
   for i in range(inds):
      best_known_positions[i,:] = best_known_positions[best_err_idx,:]

   velocities = np.random.rand(inds, dims)*2.0 - 1.0


   for itr in range(max_iter):
      for ind in range(inds):
         for d in range(dims):
            x_id = positions[ind, d]
            rp = random()
            rg = random()
            v = velocities[ind, d]

            velocities[ind, d] = w*v +theta_p*rp*(best_known_positions[ind, d]-x_id) +theta_g*rg*(global_best_pos[d]-x_id)

         positions[ind, :] = positions[ind, :] + velocities[ind, :]

         err = err_func(positions[ind, :])
         if err < errors[ind]:
            errors[ind] = err
            best_known_positions[ind,:] = positions[ind, :]
            if err < best_err:
               global_best_pos = positions[ind, :]
               best_err = err
               #print(itr, err)
   return (best_err, global_best_pos)
"""
def test(x):
   from math import sin, cos
   return (x[0]-2)*(x[0]-2) + sin(x[1]) + cos(x[2]*x[3])

pso(4, 20, 50, test)
"""