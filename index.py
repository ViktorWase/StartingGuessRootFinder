from random import gauss
from time import time
from math import sqrt, fabs
from newton_raphson import newton_raphson
from cgp import cgp
from pso import pso

def calc_computational_baseline_operations(n, f, is_binary):
	numbers = [fabs(gauss(0,1))+1.0e8 for _ in range(n)]

	if is_binary:
		second_numbers = [fabs(gauss(0,1))+1.0e8 for _ in range(n)]

	t0 = time()
	if is_binary:
		for i in range(n):
			temp = f(numbers[i], second_numbers[i])
	else:
		for i in range(n):
			temp = f(numbers[i])
	t = time() - t0

	return t/float(n)

def objective_func_cont(x, data):
	"""
	The error function of the continous optimization.
	the data object contains:
	- a CGP-object describing the mathematical function.
	- conv_factors
	- NR running time
	- ops running times
	- parameter samples
	- the corresponding roots.
	"""
	cgp = data[0]
	conv_factors = data[1]
	running_time_NR = data[2]
	running_times_ops = data[3]
	par_samples = data[4]
	roots = data[5]

	n = len(roots)
	assert len(par_samples) == n

	total_error = 0.0
	total_time_error = 0.0
	for i in range(n):
		root = roots[i]
		par_sample = par_samples[i]

		root_guess = cgp.eval(par_sample, x) # Yes, I know that par_sample goes to cgp.x, and x goes to cgp.parameters. The naming is terrible, but it is the way it should be.

		total_error += (root_guess-root)*(root_guess-root)

		# TODO: Add the error from the timeing baselines
	return sqrt(total_error)

if __name__ == '__main__':
	from random import random
	from math import pi, sin, cos
	func = lambda x, beta: x[0] - beta[0]*sin(x[0]) - beta[1]
	func_der = lambda x, beta: 1.0 - beta[0]*cos(x[0])
	parameter_generator = lambda : [random(), random()*2*pi]
	nr_of_parameters = 2

	# Get parameter samples
	n = 100
	parameter_samples_full = [parameter_generator() for _ in range(n)]

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
	
	# Get computational baseline for NR
	total_time = 0.0
	nr_of_comp_iters = 10000
	for i in range(len(roots)):
		# Create a few perturbances
		par_sample = parameter_samples[i]
		perturbances = [gauss(0, 1.0) for _ in range(nr_of_comp_iters)]

		t0 = time()
		for j in range(nr_of_comp_iters):
			x = roots[i] + perturbances[j]

			temp1 = func([x], par_sample)
			temp2 = func_der([x], par_sample)

			temp3 = -temp1/temp2
		t = (time()-t0)/float(nr_of_comp_iters)
		total_time += t
	total_time /= float(len(roots))
	print("The mean time for an iter of NR is", total_time, "sec.")

	# Subtract a baseline that calls two identity lambda funcs and calcs x as root+pert. It's hopefully a good way to get a fair comparison.
	id_func_1 = lambda x, beta: x[0]
	id_func_2 = lambda x, beta: x[0]
	t0 = time()

	for j in range(nr_of_comp_iters):
		x = roots[i] + perturbances[j]
		temp1 = id_func_1([x], par_sample)
		temp2 = id_func_2([x], par_sample)
	t_baseline_NR = (time()-t0)/float(nr_of_comp_iters)
	print(t_baseline_NR)
	total_time -= t_baseline_NR
	print("The adjusted mean time for an iter of NR is", total_time, "sec.")

	# Get computational baseline for all operations
	ops = [lambda x,y: x+y, lambda x,y: x*y, lambda x: sqrt(x), lambda x,y: x-y,  lambda x,y: x/y, lambda x: x]
	is_binary_list = [True,True,False,True,True,False]
	assert len(is_binary_list) == len(ops)

	ops_running_times = [calc_computational_baseline_operations(nr_of_comp_iters, ops[i], is_binary_list[i]) for i in range(len(ops))]
	# Since the last is id, we can subtract that as a baseline.
	for i in range(len(ops_running_times)):
		ops_running_times[i] -= ops_running_times[-1]
	print("The other running times are:", ops_running_times)

	# Calculate the convergance factor of NR for each
	# of the found roots.
	# TODO: The following only works if the derivative is non-zero at the root. Generalize somehow!
	# Calc 0.5*f''(r)/f'(r) for each root r. This defines the rate of the
	# local convergence. I mean, it's quadratic, but this is the coefficient.
	conv_factors = [0.0]*len(roots)
	der_approx = lambda f, x: 0.5*(f([x+1.0e-8])-f([x-1.0e-8]))*1.0e8
	for i in range(len(roots)):
		root = roots[i]
		pars = parameter_samples[i]
		func_der_curry = lambda x: func_der(x, pars)

		conv_factors[i] = 0.5*fabs(der_approx(func_der_curry, root))

	# Start the optimization.
	nr_of_nodes = 7
	from operation_table import op_table
	nr_of_funcs = len(op_table)
	nr_of_parameters_for_cgp = 3
	multistart_opt(roots, parameter_samples, nr_of_parameters, nr_of_funcs, nr_of_nodes, error_func, op_table, 'sa', nr_of_pars=nr_of_parameters_for_cgp, max_time=60*10)
