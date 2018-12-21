from copy import deepcopy

class TaylorApprox():
	"""
	Approximates the function by a linear Taylor approximation in the point that
	is the mean of all roots. This approximation can be used to find a starting guess by
	solving for 0.
	"""
	def __init__(self, func, func_der, all_roots):
		
		self.func = deepcopy(func)
		self.func_der = deepcopy(func_der)
		self.nr_of_parameters = nr_of_parameters
		self.mean_root = sum(float(x)/len(all_roots) for x in all_roots)

	def get_starting_guess(self, pars):
		der = self.func_der([self.mean_root], pars)
		val = self.func([self.mean_root], pars)

		# Using a Taylor approximation in the point mean_root the
		# function turns in to
		# val + der*(x-mean)=0

		# solving this for x gives
		try:
			x = -val/der + self.mean_root
			return x
		except ValueError:
			return self.mean_root


if __name__ == '__main__':
	from math import acos, sqrt, pi
	from random import random

	from newton_raphson import newton_raphson
	func = lambda x, P: acos(1.0/ ( x[0]/P[0] - 1.0))+acos(1.0/ ( x[0]/P[1] - 1.0))-P[2] if x[0] >= 0 else 1.0e10
	func_der = lambda x, P: P[0] /( (x[0]-P[0])*(x[0]-P[0])*sqrt(1.0 - (P[0]/(x[0]-P[0]))**2))+P[1]/( (x[0]-P[1])*(x[0]-P[1])*sqrt(1.0 - (P[1]/(x[0]-P[1]))**2)) if x[0] >= 0 else 1.0


	parameter_generator = lambda :  [-0.9*random()-0.05, -0.9*random()-0.05 ,pi+pi*random()]

	nr_of_parameters = 3

	# Get parameter samples
	n = 400
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
	

	tay = TaylorApprox(func, func_der, roots)

	print(tay.get_starting_guess(parameter_samples[0]))
	print(roots[0])




