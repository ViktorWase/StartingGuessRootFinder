from copy import deepcopy
from random import gauss
from math import sqrt, exp
#from optimization import acceptance_prob
def acceptance_prob(new_error, current_error, temp):
	"""
	The acceptance probability depends on the difference in the objective
	function as well as the "temperature" of the system. Higher temperature means that anything could happen.
	"""
	diff = new_error - current_error
	return exp(-(diff)/temp)

class TaylorApprox():
	"""
	Approximates the function by a linear Taylor approximation in the point that
	is the mean of all roots. This approximation can be used to find a starting guess by
	solving for 0.
	"""
	def get_starting_guess_with_pnt(self, pars, pnt):
		der = self.func_der([pnt], pars)
		val = self.func([pnt], pars)

		# Using a Taylor approximation in the point mean_root the
		# function turns in to
		# val + der*(x-mean)=0

		# solving this for x gives
		try:
			x = -val/der + pnt
			return x
		except ValueError:
			return pnt

	def get_starting_guess(self, pars):
		return self.get_starting_guess_with_pnt(pars, self.expansion_point)

	def optimize_expansion_point(self, func, func_der, all_roots, all_parameters, max_iter=2000):
		def error_l2(expansion_pnt):
			err = 0.0
			n = len(all_roots)
			assert n == len(all_parameters)
			assert n>0
			for i in range(n):
				diff = self.get_starting_guess_with_pnt(all_parameters[i], expansion_pnt)-all_roots[i]
				err += diff*diff
			return sqrt(err/float(n))

		best_of_all_itr = float('inf')
		best_of_all_pnt = 0
		for _ in range(10):
			temperature_itr = 0

			root_span = max(all_roots) - min(all_roots)
			current_exp_pnt = min(all_roots)+root_span*random()
			current_err = error_l2(current_exp_pnt)
			best_err = current_err
			best_exp_pnt = current_exp_pnt

			for itr in range(max_iter):
				if itr % 1000 == 0:
					print("iter:", itr," of", max_iter)
				temp = float(max_iter-temperature_itr)/max_iter
				# Do a small mutation to create a new function (aka solution)
				new_exp_pnt = current_exp_pnt + gauss(0, root_span/40.0)
				#cgp = CGP(dims, op_table, new_sol, nr_of_parameters=nr_of_pars)
				new_err = error_l2(new_exp_pnt)

				temperature_itr += 1
				if new_err < current_err or acceptance_prob(new_err, current_err, temp)<random():
					current_exp_pnt = new_exp_pnt
					current_err = new_err

					if new_err < best_err:
						print("best yet:", new_err)
						best_err = new_err
						best_exp_pnt = new_exp_pnt
			if best_err<best_of_all_itr:
				best_of_all_itr = best_err
				best_of_all_pnt = best_exp_pnt
		return best_of_all_pnt

	def __init__(self, func, func_der, all_roots, all_parameters):
		
		self.func = deepcopy(func)
		self.func_der = deepcopy(func_der)
		self.nr_of_parameters = nr_of_parameters
		#self.mean_root = sum(float(x)/len(all_roots) for x in all_roots)
		self.expansion_point = self.optimize_expansion_point(func, func_der, all_roots, all_parameters)

if __name__ == '__main__':
	from math import acos, sqrt, pi
	from random import random

	from newton_raphson import newton_raphson
	func = lambda x, P: acos(1.0/ ( x[0]/P[0] - 1.0))+acos(1.0/ ( x[0]/P[1] - 1.0))-P[2] if x[0] >= 0 else 1.0e10
	func_der = lambda x, P: P[0] /( (x[0]-P[0])*(x[0]-P[0])*sqrt(1.0 - (P[0]/(x[0]-P[0]))**2))+P[1]/( (x[0]-P[1])*(x[0]-P[1])*sqrt(1.0 - (P[1]/(x[0]-P[1]))**2)) if x[0] >= 0 else 1.0


	parameter_generator = lambda :  [-0.9*random()-0.05, -0.9*random()-0.05 ,pi+pi*random()]

	nr_of_parameters = 3

	# Get parameter samples
	n = 1000
	parameter_samples_full = [parameter_generator() for _ in range(n)]
	for tmp in parameter_samples_full:
		assert len(tmp) == nr_of_parameters

	# Find the roots for each parameter sample
	newt_rap_results = [newton_raphson(func, func_der, parameter, max_iter=10000, convg_lim=1.0e-14, x0=1.0e-8) for parameter in parameter_samples_full]
	roots = []
	parameter_samples = []
	i = 0
	for res in newt_rap_results:
		root, error = res
		# Remove all roots that didn't converge.
		if error<1.0e-12:
			roots.append(root)
			parameter_samples.append(parameter_samples_full[i])
		i += 1
	

	tay = TaylorApprox(func, func_der, roots, parameter_samples)
	from math import fabs

	const_mean = 0.0
	tay_mean = 0.0
	mean = sum(r for r in roots)/len(roots)
	assert len(roots)==len(parameter_samples)
	for i in range(len(roots)):
		diff = fabs(roots[i]-tay.get_starting_guess(parameter_samples[i]))
		tay_mean += diff/len(roots)
		const_mean += (fabs(roots[i]-mean))/len(roots)
	print(tay_mean)
	print(const_mean)
