from random import gauss
from copy import copy
from time import time
from math import sqrt, fabs, inf
from newton_raphson import newton_raphson
from cgp import CGP
from optimization import multistart_opt
from pso import pso

def calc_alpha_for_root(func, par_vals, x_root, h=1.0e-10):
	# Calcs 2*f''(x)/f'(x)
	f_left = func([x_root - h], par_vals)
	f_mid = func([x_root], par_vals)
	f_right = func([x_root + h], par_vals)

	f_der = (f_right - f_left)/(2*h)
	f_second_der = (f_right - 2.0*f_mid + f_left)/(h*h)

	return 2.0*f_second_der/f_der

def projected_new_error(nr_of_newton_iters, start_err, alpha):
	# So yeah, this is hard to explain. TODO: Write a good explanation.
	exponent = pow(2.0, nr_of_newton_iters)
	return pow(alpha, exponent-1)*pow(start_err, exponent)

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

	# TODO: Check if the following 2 lines are slow. They don't need to be recalculated everytime.
	is_par_used = cgp.which_parameters_are_used()
	nr_of_pars_used = sum(is_par_used)

	

	# One could imagine a function f, with one variable x, and two parameters a & b,
	# where the function is f(x) = b*x. Which means that the parameter a is unused. This means 
	# that it doesn't have to be tuned, but the function still needs two parameters (at least it 
	# thinks that it does). In these cases we just set a:=0 and tune only b.

	def func(X, parameters=[], derivative=False, der_dir=0):
		# TODO: This shouldn't have to be done every time.
		# Insert zeroes in the non-used parameters
		pars = [0.0 for _ in range(cgp.nr_of_parameters)]
		#print(x, parameters, nr_of_pars_used, cgp.nr_of_parameters)
		assert len(parameters) == nr_of_pars_used

		counter = 0
		for i in range(cgp.nr_of_parameters):
			if is_par_used[i]:
				pars[i] = parameters[counter]
				counter += 1

		try:
			return cgp.eval(X, parameters=pars, derivative=derivative, der_dir=der_dir)
		except (ValueError, ZeroDivisionError): # Sometimes x gets really big and this can cause problems.
			print("Math domain error in start error func.")
			return 1.0e20

	n = len(roots)
	assert len(par_samples) == n

	total_error = 0.0
	total_time_error = 0.0
	for i in range(n):
		root = roots[i]
		par_sample = par_samples[i]

		root_guess = func(par_sample, x) # Yes, I know that par_sample goes to cgp.x, and x goes to cgp.parameters. The naming is terrible, but it is the way it should be.
		err = fabs(root_guess-root)

		nr_of_newton_iters = 0 # TODO: SET THIS!
		proj_err = projected_new_error(nr_of_newton_iters, fabs(root_guess-root), conv_factors[i])
		projected_lost_decrease_in_error = err - proj_err

		total_error += (err+projected_lost_decrease_in_error)*(err+projected_lost_decrease_in_error)

	return sqrt(total_error)

def objective_func_disc(f_vals, pnts, dims, new_cgp, nr_of_pars, op_table, data):
	cgp = copy(new_cgp) # TODO: Is this copy really needed?
	# Then we find out which parameters that are used.
	is_par_used = cgp.which_parameters_are_used()
	nr_of_pars_used = sum(is_par_used)
	data = [new_cgp] + data
	
	# We ignore the constant CGPs, because they are dumb and treated separately.
	if cgp.is_constant:
		return (inf, [])

	if nr_of_pars_used > 0:
		# If some parameters are used in the model, then these will have to be tuned.
		
		# Then we do a curve-fitting of the numerical constants/parameters.
		nr_of_pnts = len(pnts)

		err_func = lambda x: objective_func_cont(x, data)
		inds = 20
		max_iter = 50
		(error, best_par_vals) = pso(nr_of_pars_used, inds, max_iter, err_func)

		#(best_par_vals, error) = combo_curve_fitting(residual_func, f_vals, nr_of_pars_used, nr_of_pnts, pnts, jacobian_func, func)

		# If some parameters are unused, then we'll just set them to 0.
		if len(best_par_vals) != nr_of_pars:
			assert len(best_par_vals) < nr_of_pars
			best_par_vals_padded = [0.0 for _ in range(cgp.nr_of_parameters)]

			counter = 0
			for i in range(cgp.nr_of_parameters):
				if is_par_used[i]:
					best_par_vals_padded[i] = best_par_vals[counter]
					counter += 1
			best_par_vals = best_par_vals_padded
	else:
		# If no parameter is actually used, then there is no need for any curve-fitting.
		func = cgp.eval

		# Okay, so no parameters are actually used, so we'll just use some dummy values.
		pars = [0.0 for _ in range(cgp.nr_of_parameters)]
		error = objective_func_cont([], data)
		best_par_vals = pars

	return (error, best_par_vals)

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
	# TODO: If the operation has 2 operands, then so should the lambda that is used as a baseline.
	for i in range(len(ops_running_times)):
		ops_running_times[i] -= ops_running_times[-1]
	print("The other running times are:", ops_running_times)

	# Calculate the convergance factor of NR for each
	# of the found roots.
	# TODO: The following only works if the derivative is non-zero at the root. Generalize somehow!
	# Calc 0.5*f''(r)/f'(r) for each root r. This defines the rate of the
	# local convergence. I mean, it's quadratic, but this is the coefficient.
	conv_factors = [0.0]*len(roots)
	for i in range(len(roots)):
		root = roots[i]
		pars = parameter_samples[i]

		conv_factors[i] = calc_alpha_for_root(func, pars, root)

	# Collect all the data objects the optimization needs.
	optimization_data = [conv_factors, total_time, ops_running_times, parameter_samples, roots]

	# Start the optimization.
	nr_of_nodes = 7
	from operation_table import op_table
	nr_of_funcs = len(op_table)
	nr_of_parameters_for_cgp = 3
	multistart_opt(roots, parameter_samples, nr_of_parameters, nr_of_funcs, nr_of_nodes, objective_func_disc, op_table, 'sa', optimization_data, nr_of_pars=nr_of_parameters_for_cgp, max_time=60*10)
