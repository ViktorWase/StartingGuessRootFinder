from random import gauss
from copy import copy
from time import time, perf_counter
from math import sqrt, fabs
from newton_raphson import newton_raphson
from cgpy.cgp import CGP
from optimization import multistart_opt
from pso import pso
from taylor_approx import TaylorApprox
inf = float('inf')


def calc_alpha_for_root(func, par_vals, x_root, h=1.0e-10):
	# Calcs 2*f''(x)/f'(x)
	f_left = func([x_root - h], par_vals)
	f_mid = func([x_root], par_vals)
	f_right = func([x_root + h], par_vals)

	f_der = (f_right - f_left)/(2*h)
	f_second_der = (f_right - 2.0*f_mid + f_left)/(h*h)

	return min(fabs(2.0*f_second_der/(f_der+1.0e-10)), 1.0/(x_root*x_root+1.0e-10))


def projected_new_error(nr_of_newton_iters, start_err, alpha):
	# So yeah, this is hard to explain. TODO: Write a good explanation.
	exponent = pow(2.0, nr_of_newton_iters)
	if alpha == 0:
		return 0
	return pow(alpha, exponent-1)*pow(start_err, exponent)


def calc_computational_baseline_operations(n, f, is_binary):
	numbers = [fabs(gauss(0, 1))+1.0e-8 for _ in range(n)]

	if is_binary:
		second_numbers = [fabs(gauss(0, 1))+1.0e-8 for _ in range(n)]

	t0 = time()
	if is_binary:
		for i in range(n):
			temp = f(numbers[i], second_numbers[i])
	else:
		for i in range(n):
			temp = f(numbers[i])
	t = time() - t0

	return t/float(n)


def calc_the_number_of_newton_iters_that_are_lost(cgp, running_times_ops, running_time_NR):
	total_time = 0.0
	nr_of_nodes = int((len(cgp.gene)-1)/3)
	for node in range(nr_of_nodes):
		if cgp.used_nodes[node]:
			op_nr = cgp.gene[3*node]
			assert op_nr < len(cgp.op_table)

			total_time += running_times_ops[op_nr]
	newton_iters = total_time / running_time_NR
	return newton_iters


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

	nr_of_newton_iters = calc_the_number_of_newton_iters_that_are_lost(cgp, running_times_ops, running_time_NR)

	# One could imagine a function f, with one variable x, and two parameters a & b,
	# where the function is f(x) = b*x. Which means that the parameter a is unused. This means
	# that it doesn't have to be tuned, but the function still needs two parameters (at least it
	# thinks that it does). In these cases we just set a:=0 and tune only b.

	def func(X, parameters=[]):
		# TODO: This shouldn't have to be done every time.
		# Insert zeroes in the non-used parameters
		pars = [0.0 for _ in range(cgp.nr_of_parameters)]
		assert len(parameters) == nr_of_pars_used

		counter = 0
		for i in range(cgp.nr_of_parameters):
			if is_par_used[i]:
				pars[i] = parameters[counter]
				counter += 1

		try:
			return cgp.eval(X, parameters=pars)
		except (ValueError, ZeroDivisionError):  # Sometimes x gets really big and this can cause problems.
			print("Math domain error in start error func.")
			return 1.0e20

	n = len(roots)
	assert len(par_samples) == n

	total_error = 0.0
	for i in range(n):
		root = roots[i]
		par_sample = par_samples[i]

		root_guess = func(par_sample, x)  # Yes, I know that par_sample goes to cgp.x, and x goes to cgp.parameters. The naming is terrible, but it is the way it should be.
		err = fabs(root_guess-root)

		proj_err = projected_new_error(nr_of_newton_iters, fabs(root_guess-root), conv_factors[i])
		projected_lost_decrease_in_error = err - proj_err

		# clamp the error? TODO: Think about this. Is it wise?
		projected_lost_decrease_in_error = max(projected_lost_decrease_in_error, 0.0)

		total_error += (err+projected_lost_decrease_in_error)*(err+projected_lost_decrease_in_error)

	return sqrt(total_error/n)


def objective_func_disc(f_vals, pnts, dims, new_cgp, nr_of_pars, op_table, data):
	assert len(data) == 5
	cgp = copy(new_cgp)  # TODO: Is this copy really needed?
	# Then we find out which parameters that are used.
	is_par_used = cgp.which_parameters_are_used()
	nr_of_pars_used = sum(is_par_used)
	data = [new_cgp] + data

	# We ignore the constant CGPs, because they are dumb and treated separately.
	if cgp.is_constant:
		return (inf, [None]*nr_of_pars)

	if nr_of_pars_used > 0:
		# If some parameters are used in the model, then these will have to be tuned.
		# Then we do a curve-fitting of the numerical constants/parameters.
		err_func = lambda x: objective_func_cont(x, data)
		inds = 20
		max_iter = 50
		(error, best_par_vals) = pso(nr_of_pars_used, inds, max_iter, err_func)

		# (best_par_vals, error) = combo_curve_fitting(residual_func, f_vals, nr_of_pars_used, nr_of_pnts, pnts, jacobian_func, func)

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

		# Okay, so no parameters are actually used, so we'll just use some dummy values.
		pars = [0.0 for _ in range(cgp.nr_of_parameters)]
		error = objective_func_cont([], data)
		best_par_vals = pars

	return (error, best_par_vals)


def how_many_genes_exists(dims, nr_of_nodes, ignore_id=True):
	from operation_table import op_table

	nr_of_ops = len(op_table)
	if ignore_id:
		if 'id' in [op.op_name for op in op_table]:
			nr_of_ops -= 1

	nr_of_binary_ops = sum([1 if op.is_binary else 0 for op in op_table])
	nr_of_unary_ops = nr_of_ops - nr_of_binary_ops

	counter = 1
	for i in range(nr_of_nodes):
		prev_nodes = dims + i

		counter *= nr_of_binary_ops*prev_nodes*prev_nodes + nr_of_unary_ops*prev_nodes

	# And lastly the choice of output node
	counter *= nr_of_nodes+dims

	return counter


def generate_parameters(nr_of_parameters, parameter_generator, n=250):
	assert n > 0
	assert nr_of_parameters > 0
	parameter_samples_full = [parameter_generator() for _ in range(n)]
	for tmp in parameter_samples_full:
		assert len(tmp) == nr_of_parameters
	return parameter_samples_full


def binary_search_for_root(func, trials=1000, binary_search_iters=1000, sd=100, thresh=1.0e-12):
	has_found_pos = False
	has_found_neg = False
	# Randomly search for a negative and positive pnt.
	for _ in range(trials):
		if has_found_neg and has_found_pos:
			break
		x = gauss(0, 100)
		val = func([x])
		if val is not None:
			if not has_found_pos and val > 0:
				pos_x = x
				has_found_pos = True
			elif not has_found_neg and val < 0:
				has_found_neg = True
				neg_x = x
	if has_found_neg is False or has_found_pos is False:
		return False

	# Do binary search to find a root.
	# This assumes that the function is continous.
	for i in range(binary_search_iters):
		x_mid = (pos_x+neg_x)*0.5
		val = func([x_mid])
		if fabs(val) <= thresh:
			return x_mid, val
		if val > 0:
			pos_x = x_mid
		else:
			neg_x = x_mid

	return False


def calc_roots(func, func_der, parameter_samples_full):
	newt_rap_results = [newton_raphson(func, func_der, parameter, max_iter=10000, convg_lim=1.0e-12, x0=1.0e-8) for parameter in parameter_samples_full]
	roots = []
	parameter_samples = []
	i = 0

	nr_of_non_converged = 0
	for res in newt_rap_results:
		root, error = res
		# Remove all roots that didn't converge.
		if error < 1.0e-12:
			roots.append(root)
			parameter_samples.append(parameter_samples_full[i])
		else:
			parameter = parameter_samples_full[i]
			func_curry = lambda x: func(x, parameter)
			binary_search_res = binary_search_for_root(func_curry)  # Don't forget to rerun this with exp 1-3 since there was a bug.
			if binary_search_res is False:
				nr_of_non_converged += 1
				pass
			else:
				roots.append(binary_search_res[0])
				parameter_samples.append(binary_search_res[0])
		i += 1
	return roots, parameter_samples, nr_of_non_converged


def calc_baseline(roots, parameter_samples, nr_of_comp_iters=10000):
	# Get computational baseline for NR
	assert len(roots) > 0

	assert len(roots) == len(parameter_samples)
	total_time = 0.0
	for i in range(len(roots)):
		# Create a few perturbances
		par_sample = parameter_samples[i]
		perturbances = [gauss(0, 1.0) for _ in range(nr_of_comp_iters)]

		if X_NEEDS_TO_BE_POS:
			for j in range(nr_of_comp_iters):
				while roots[i] + perturbances[j] < 0:
					perturbances[j] = gauss(0, 1.0)

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
	# TODO: Create these automatically
	ops = [lambda x: sin(x), lambda x: cos(x), lambda x, y: x+y, lambda x, y: x*y, lambda x: x*x, lambda x, y: x-y, lambda x, y: x/y, lambda x: x]
	is_binary_list = [False, False, True, True, False, True, True, False]
	assert len(is_binary_list) == len(ops)

	nr_of_comp_iters_ops = nr_of_comp_iters*100
	ops_running_times = [calc_computational_baseline_operations(nr_of_comp_iters_ops, ops[i], is_binary_list[i]) for i in range(len(ops))]
	# Since the last is id, we can subtract that as a baseline.
	# TODO: If the operation has 2 operands, then so should the lambda that is used as a baseline.
	for i in range(len(ops_running_times)):
		ops_running_times[i] -= ops_running_times[-1]
	print("The other running times are:", ops_running_times)
	return ops_running_times, total_time


def calc_baselines_and_roots(func, func_der, parameter_generator, nr_of_parameters):
	# Get parameter samples
	parameter_samples_full = generate_parameters(nr_of_parameters, parameter_generator)
	print("1")
	# Find the roots for each parameter sample
	roots, parameter_samples, nr_of_non_converged = calc_roots(func, func_der, parameter_samples_full)
	assert len(roots) > 0
	print("2")
	# Get computational baseline for NR
	ops_running_times, total_time = calc_baseline(roots, parameter_samples)
	print("3")

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

	return optimization_data


def get_constant_solution(roots):
	n = len(roots)

	return sum(roots)/float(n)


def cgp_find_starting_guess(optimization_data, func, func_der,	parameter_generator, nr_of_parameters, max_time_sec=8*60*60):
	print("Starting cgp optimization algo.")
	[conv_factors, total_time, ops_running_times, parameter_samples, roots] = optimization_data

	# Start the optimization.
	nr_of_nodes = 7
	from operation_table import op_table
	nr_of_funcs = len(op_table)
	nr_of_parameters_for_cgp = 3
	(best_sol, best_err, best_pars) = multistart_opt(roots, parameter_samples, nr_of_parameters, nr_of_funcs, nr_of_nodes, objective_func_disc, op_table, 'sa', optimization_data, nr_of_pars=nr_of_parameters_for_cgp, max_time=max_time_sec)

	return (best_sol, best_err, best_pars)


if __name__ == '__main__':
	from random import random
	from math import pi, sin, cos, acos, sqrt, asin
	from scipy.special import erf
	X_NEEDS_TO_BE_POS = False
	print("Starting, time:", time())

	# EXP 1 - Planetary ellipse position
	func = lambda x, beta: x[0] - beta[0]*sin(x[0]) - beta[1]
	func_der = lambda x, beta: 1.0 - beta[0]*cos(x[0])
	parameter_generator = lambda : [random()*0.1, random()*2*pi]
	nr_of_parameters = 2

	# EXP 2 - sqrt in range [0, 1]
	# func = lambda x, beta: x[0]*x[0] - beta[0]
	# func_der = lambda x, beta: 2*x[0]
	# parameter_generator = lambda: [random()]
	# nr_of_parameters = 1

	# EXP 3 - MGA Flyby calculation
	# X_NEEDS_TO_BE_POS = True
	# func = lambda x, beta: asin(beta[0]/(beta[0]+x[0]))+asin(beta[1]/(beta[1]+x[0]))-beta[2]
	# func_der = lambda x, beta: -beta[0]/((beta[0]+x[0])*sqrt(x[0]*(2.0*beta[0]+x[0]))) - beta[1]/((beta[1]+x[0])*sqrt(x[0]*(2.0*beta[1]+x[0])))
	# parameter_generator = lambda : [1.0/((random()*1.5+0.5)**2), 1.0/((random()*1.5+0.5)**2), random()*pi*0.5]
	# nr_of_parameters = 3

	# EXP 4 - Normal propabillity calculation
	## func = lambda x, beta: -0.5*erf((beta[0]-x[0])/(1.41421356237*beta[1])) - beta[2]
	# func = lambda x, beta: 0.5+erf((beta[0]-x[0])/(2.82842712475*beta[1])) - beta[2]
	## func_der = lambda x, beta: exp(-(x[0] - beta[0])*(x[0] - beta[0])/(2.0*beta[1]*beta[1]))/(2.50662827463*beta[1])
	# func_der = lambda x, beta: 0.3989422804 * exp(-(x[0] - beta[0])*(x[0] - beta[0])/(2.0*beta[1]*beta[1]))/(beta[1])
	# parameter_generator = lambda : [gauss(0,1), fabs(gauss(0,1)), random()*0.9999+(1-0.9999)/2.0]
	# nr_of_parameters = 3

	"""
	# EXP 5 - ANN for ITD
	import torch
	from ann import Net
	from scipy.interpolate import LinearNDInterpolator

	# net = torch.load("ANN")
	net = torch.load("net_weight01_nr2")
	# x is azi, beta is [ele,  head_size, itd]

	def func(x, beta):
		tmp = net(torch.DoubleTensor(x+beta[0:2])).item()
		return tmp - beta[2]

	def func_der(x, beta):
		out = (net(torch.DoubleTensor([x[0]+1.0e-6]+beta[0:2])) - net(torch.DoubleTensor([x[0]-1.0e-6]+beta[0:2]))).item() / 2.0e-6
		return out.item()

	def parameter_generator_full(cipic_data, cipic_data_itd):
		itd = float('nan')
		interpolator = LinearNDInterpolator(cipic_data, cipic_data_itd)

		while isnan(itd):  # TODO: Make sure that the cyclic variables are taken care of.
			azi = random()*pi  # TODO: I don't think this is correct
			ele = random()*(180+45)/180*pi-pi/4
			head_size = gauss(14.49, 0.95)
			X = [azi, ele, head_size]

			itd_interpol = interpolator(X)
			itd = itd_interpol[0][0]
			print("itd", itd)

		return [ele, head_size, itd]

	import scipy.io as sio
	from ann import get_data
	import numpy as np
	from math import isnan
	IDs = ['003', '008', '009', '010', '011', '012', '015', '017', '018', '019', '020', '021', '027', '028', '033', '040', '044', '048', '050', '051', '058', '059', '060', '061', '065', '119', '124', '126', '127', '131', '133', '134', '135', '137', '147', '148', '152', '153', '154', '155', '156', '158', '162', '163', '165']
	anthro_mat = sio.loadmat('itd_data/anthropometry/anthro.mat')

	# Get the IDs. ach should have 3 numbers.
	head_sizes = [x[0] for x in anthro_mat['X']]

	# Remove all subjects with missing data
	counter = 0
	while counter < len(head_sizes):
		if isnan(head_sizes[counter]):
			head_sizes.pop(counter)
			IDs.pop(counter)
		else:
			counter += 1
	assert len(IDs) == len(head_sizes)
	inps, itds = get_data(IDs, head_sizes)

	parameter_generator = lambda: parameter_generator_full(inps, itds)
	nr_of_parameters = 3
	"""

	#func = lambda x, P: acos(1.0/ ( x[0]/P[0] - 1.0))+acos(1.0/ ( x[0]/P[1] - 1.0))-P[2] if x[0] >= 0 else 1.0e10
	#func_der = lambda x, P: P[0] /( (x[0]-P[0])*(x[0]-P[0])*sqrt(1.0 - (P[0]/(x[0]-P[0]))**2))+P[1]/( (x[0]-P[1])*(x[0]-P[1])*sqrt(1.0 - (P[1]/(x[0]-P[1]))**2)) if x[0] >= 0 else 1.0
	#parameter_generator = lambda :  [-0.9*random()-0.05, -0.9*random()-0.05 ,pi+pi*random()]
	#nr_of_parameters = 3


	print("Calc baselines and roots")
	opt_data = calc_baselines_and_roots(func, func_der, parameter_generator, nr_of_parameters)
	[conv_factors, total_time, ops_running_times, parameter_samples, roots] = opt_data

	print("Calc Taylor Approx")
	tay = TaylorApprox(func, func_der, roots, parameter_samples)
	print("Calc constant solution")
	con = get_constant_solution(roots)

	(best_sol, best_err, best_pars) = cgp_find_starting_guess(opt_data, func, func_der, parameter_generator, nr_of_parameters)
	print("Cgp done")

	n = len(roots)
	print("Con:", con)
	print("Taylor expansion pnt:", tay.expansion_point)
	print("CGP function:")
	print(best_err, best_pars)
	best_sol.print_function(parameters=best_pars)
	print("---------")
	print(conv_factors)
	print(total_time)
	print(ops_running_times)
	
	print("Generating new parameter samples for testing")
	# Get parameter samples
	par_samples = generate_parameters(nr_of_parameters, parameter_generator)
	limit = 1.0e-6

	all_const_times = [0.0]*n
	for i in range(n):
		t0 = perf_counter()
		parameters = par_samples[i]
		newton_raphson(func, func_der, parameters, max_iter=10000, convg_lim=limit, x0=con)
		t = perf_counter()-t0
		all_const_times[i] = t

	all_tay_times = [0.0]*n
	for i in range(n):
		t0 = perf_counter()
		pars = par_samples[i]
		c = tay.get_starting_guess(pars)
		newton_raphson(func, func_der, parameters, max_iter=10000, convg_lim=limit, x0=c)
		t = perf_counter()-t0
		all_tay_times[i] = t

	all_cgp_times = [0.0]*n
	cgp_func = False
	for i in range(n):
		t0 = perf_counter()
		pars = par_samples[i]
		c = cgp_func(pars)
		newton_raphson(func, func_der, parameters, max_iter=10000, convg_lim=limit, x0=c)
		t = perf_counter()-t0
		all_cgp_times[i] = t

	all_const_times.sort()
	all_tay_times.sort()
	all_cgp_times.sort()

	import matplotlib.pyplot as plt
	plt.plot(all_const_times, 'r')
	plt.plot(all_tay_times)
	plt.plot(all_cgp_times)

	plt.show()
