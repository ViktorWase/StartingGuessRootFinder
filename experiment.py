# Create 2000 random samples. 
# The first half is for training, the other half for verification.

from math import fabs
from time import time, perf_counter
from random import random

from newton_raphson import newton_raphson
from taylor_approx import TaylorApprox

def get_constant_solution(roots):
	n = len(roots)

	return sum(roots)/float(n)

def get_all_roots(func, func_der, parameter_generator, nr_of_parameters, n=10000):
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
	assert len(parameter_samples) == len(roots)

	return roots, parameter_samples

def taylor_experiment(func, func_der, roots, parameter_samples):
	tay = TaylorApprox(func, func_der, roots, parameter_samples)

	tay_mean = 0.0
	mean = sum(r for r in roots)/len(roots)
	assert len(roots)==len(parameter_samples)
	for i in range(len(roots)):
		diff = fabs(roots[i]-tay.get_starting_guess(parameter_samples[i]))
		tay_mean += diff/len(roots)
	print(tay_mean)

if __name__ == '__main__':
	#func = lambda x, beta: x[0]*x[0]*x[0] - beta[0]
	#func_der = lambda x, beta: 3*x[0]*x[0]
	#parameter_generator = lambda: [random()]
	#nr_of_parameters = 1

	from math import sin, cos
	func = lambda x, beta: x[0] - beta[0]*sin(x[0]) - beta[1]
	func_der = lambda x, beta: 1.0 - beta[0]*cos(x[0])
	nr_of_parameters = 2
	parameter_generator = lambda: [0.4*random(), random()]

	roots, par_samples = get_all_roots(func, func_der, parameter_generator, nr_of_parameters)
	#taylor_experiment(func, func_der, roots, par_samples)

	tay = TaylorApprox(func, func_der, roots, par_samples)
	con = get_constant_solution(roots)
	n = len(roots)
	print("Con:", con)
	print("Taylor expansion pnt:", tay.expansion_point)
	print("CGP function:")
	assert False
	all_tay_times = [0.0]*n
	for i in range(n):
		t0 = perf_counter()
		parameters = par_samples[i]
		newton_raphson(func, func_der, parameters, max_iter=10000, convg_lim=1.0e-5, x0=con)
		t = perf_counter()-t0
		all_tay_times[i] = t

	all_const_times = [0.0]*n
	for i in range(n):
		t0 = perf_counter()
		pars = par_samples[i]
		c = tay.get_starting_guess(pars)
		newton_raphson(func, func_der, parameters, max_iter=10000, convg_lim=1.0e-5, x0=c)
		t = perf_counter()-t0
		all_const_times[i] = t

	all_const_times.sort()
	all_tay_times.sort()

	import matplotlib.pyplot as plt
	plt.plot(all_const_times, 'r')
	plt.plot(all_tay_times)

	plt.show()
