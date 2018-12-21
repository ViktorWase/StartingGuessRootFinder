"""
The function starting_point_approximation in this file is used for finding the function that gives the starting guess of the root. 
For example if the input function is x-a*sin(x)-b = 0, where x is the variable 
and a and b are constants/parameters. Then the function starting_point_approximation 
might return the function g(a,b)=b.

The first value (x_0) in the root finding algorithm will then be set to b.
x_0 := g(a,b)
"""

from sys import version_info
if version_info >= (3,0):
	from math import sqrt, inf, fabs
else:
	from math import sqrt, fabs
	inf = 1.0e25

from random import random, seed

from simulated_anneling import multistart_opt
from cgp import Operation, CGP
from gauss_newton import gauss_newton
#from non_linear_curve_fitting import gradient_descent
from newton_raphson import root_finders
from non_linear_curve_fitting import combo_curve_fitting

def residual_func(f_vals, pnts, beta, func):
	"""
	This function returns a vector of the differences between
	the actual function values (f_vals) and the value of the 
	function that we are trying to tune ( func(pnt, beta) ).

	Each element in the output vector represents one point from
	the input vector pnts.
	"""
	nr_of_pnts = len(pnts)
	differences = [0.0] * nr_of_pnts
	for i in range(nr_of_pnts):
		#tmp = func(beta, pnts[i]) - f_vals[i]
		tmp = func(pnts[i], parameters=beta) - f_vals[i]
		differences[i] = tmp
	return differences

def jacobian_func(f_vals, pnts, beta, func):
	"""
	This is the Jacobian matrix of the residual function above.
	"""
	nr_of_pars = len(beta)
	nr_of_residuals = len(pnts)
	jacobian = [[0.0 for i in range(nr_of_pars)] for j in range(nr_of_residuals)]

	h = 1.0e-10
	for i in range(nr_of_residuals):
		pnt = pnts[i]

		tmp = func(pnt, parameters=beta) - f_vals[i]
		for j in range(nr_of_pars):

			"""
			# TODO: Take derivative using dual numbers instead
			beta_shift = list(beta)
			beta_shift[j] += h
			der = ( func(pnt, parameters=beta_shift) - func(pnt, parameters=beta) ) / h
			"""
			#val, der = func(pnt, parameters=beta, derivative=True, der_dir=j) # I think this one is wrong. It takes the derivatives wrt the points and not the parameters.
			val, der = func(pnt, parameters=beta, derivative=True, der_dir=j+len(pnt))
			jacobian[i][j] = der * tmp
	return jacobian

def starting_point_approximation_symbolic_regression(f_vals, pnts, optimizer, nr_of_parameters=3, max_iter=1000, multi_starts=5, nr_of_nodes=15, max_time=None):
	"""
	The purpose of this function is to find an analytical function 
	f such that f(pnt) is close to f_val for the pnt is pnts and 
	f_val in f_vals.

	Note that the returned function f will be an analytical function,
	and that this is called symbolic regression.
	"""

	dims = len(pnts[0])
	for pnt in pnts:
		assert dims == len(pnt)
	assert dims>0

	n = len(f_vals)
	assert len(pnts) == n

	from operation_table import op_table
	nr_of_funcs = len(op_table)

	# Make sure that the same operation isn't used several times
	for op_nr_1 in range(nr_of_funcs):
		op_name_1 = op_table[op_nr_1].op_name
		for op_nr_2 in range(nr_of_funcs):
			op_name_2 = op_table[op_nr_2].op_name

			if op_name_1==op_name_2 and op_nr_2!=op_nr_1:
				print("The same operation is used at least twice in the starting point approximation. Shutting down.")
				assert False

	def error_func(f_vals, pnts, dims, cgp, nr_of_pars, op_table):
		"""
		This is the objective function that will be minimized.

		It takes a gene and creates a CGP from it. However, the values 
		of the numerical constants are not included in the gene. This 
		means that they have to be found using a countinous 
		optimizer. 

		For now we use Gauss-Newton to find the best values of these
		numerical parameters. This is a curve-fitting problem where
		we attempt to minimize the square of the residuals.

		The function returns the error.

		NOTE: This function and all its calls will be a bottle-neck,
		and hence it's worth spending time making it more efficient.
		"""

		# First we create the CGP object without any values to the numerical constants.
		#cgp = CGP(dims, op_table, gene, nr_of_parameters=nr_of_pars)
		assert cgp.nr_of_parameters == nr_of_pars

		# Then we find out which parameters that are used.
		is_par_used = cgp.which_parameters_are_used()
		nr_of_pars_used = sum(is_par_used)
		#print(" ")
		#print("Function:")
		#cgp.print(parameters=[i for i in range(nr_of_pars)])
		
		# We ignore the constant CGPs, because they are dumb and treated separately.
		if cgp.is_constant:
			return (inf, [])

		if nr_of_pars_used > 0:
			# If some parameters are used in the model, then these will have to be tuned.
			# However, one could imagine a function f, with one variable x, and two parameters a & b,
			# where the function is f(x) = b*x. Which means that the parameter a is unused. This means 
			# that it doesn't have to be tuned, but the function still needs two parameters (at least it 
			# thinks that it does). In these cases we just set a:=0 and tune only b.

			def func(X, parameters=[], derivative=False, der_dir=0):
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
					return cgp.eval(X, parameters=pars, derivative=derivative, der_dir=der_dir)
				except (ValueError, ZeroDivisionError): # Sometimes x gets really big and this can cause problems.
					print("Math domain error in start error func.")
					return 1.0e20


			# Then we do a curve-fitting of the numerical constants/parameters.
			nr_of_pnts = len(pnts)
			(best_par_vals, error) = combo_curve_fitting(residual_func, f_vals, nr_of_pars_used, nr_of_pnts, pnts, jacobian_func, func)
			#(best_par_vals, error) = differential_evolution(pnts, f_vals, nr_of_pars_used, residual_func, func)
			#(best_par_vals, error) = gradient_descent(residual_func, f_vals, nr_of_pars_used, nr_of_pnts, pnts, jacobian_func, func)
			#(best_par_vals, error) = gauss_newton(residual_func, f_vals, nr_of_pars_used, nr_of_pnts, pnts, jacobian_func, func)

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

			return (sqrt(error/ float(len(pnts))), best_par_vals)
		else:
			# If no parameter is actually used, then there is no need for any curve-fitting.
			func = cgp.eval

			# Okay, so no parameters are actually used, so we'll just use some dummy values.
			pars = [0.0 for _ in range(cgp.nr_of_parameters)]
			res = residual_func(f_vals, pnts, pars, func)
			return (sqrt(sum(r*r for r in res) / float(len(res))), pars)

	# Run the mutistart simulated anneling optimization algorithm.
	(cgp, best_err, best_pars) = multistart_opt(f_vals, pnts, dims, nr_of_funcs, nr_of_nodes, error_func, op_table, optimizer, max_iter=max_iter, multi_starts=multi_starts, nr_of_pars=nr_of_parameters, max_time=max_time)
	func = lambda x: cgp.eval(x, parameters=best_pars) # TODO: This only works if ALL PARAMETERS ARE USED

	return (cgp, best_err, best_pars)

def starting_point_approximation(func, nr_of_parameters, parameter_ranges, optimizer, max_iter=1000, multi_starts=2, nr_of_samples_per_parameter=25, nr_of_parameters_in_cgp=3, max_time=None, symbolic_der=None):

	assert nr_of_samples_per_parameter > 1
	assert nr_of_parameters >= 0
	if nr_of_parameters == 0:
		# I guess we don't really need an approximation 
		# function for the starting point in this case. I mean, there will 
		# only be one value for the root.
		print("There are no parameters in the input function! Then there is no need for this program.")
		assert False
	else:

		# Make sure that the input data even makes sense.
		assert len(parameter_ranges) == nr_of_parameters
		for tmp in parameter_ranges:
			assert len(tmp) == 2
			assert tmp[0] < tmp[1]

		# Calculate the total number of samples.
		nr_of_samples = 1
		for _ in range(nr_of_parameters):
			nr_of_samples *= nr_of_samples_per_parameter

		# Generate random parameter points in the given range. 
		# TODO: this is stupid, change later! We really should sample on a nice
		# Cartesian grid.
		parameter_samples = [[0.0 for _ in range(nr_of_parameters)] for _ in range(nr_of_samples)]
		for i in range(nr_of_samples):
			for d in range(nr_of_parameters):
				parameter_samples[i][d] = random()*(parameter_ranges[d][1]-parameter_ranges[d][0])+parameter_ranges[d][0]


		# Let's get the derivative as well. 
		if symbolic_der == None:
			func_der = lambda x, a: (func([x[0]+1.0e-11] , a) - func([x[0]],a))/1.0e-11
		else:
			func_der = symbolic_der
		# Step 1 
		# For each point, find the x val that is the (or a) root.
		# Do this using Newton-Raphson.
		root_samples_and_errors = [root_finders(func, func_der, parameter_samples[i]) for i in range(nr_of_samples)]
		
		# Remove all points that didn't converge
		converge_thresh = 1.0e-8
		remove_idxs = []
		counter = 0
		for i in range(nr_of_samples):
			err = root_samples_and_errors[counter][1]
			if err > converge_thresh:
				tmp = root_samples_and_errors.pop(counter)
				parameter_samples.pop(counter)
				assert tmp[1] > converge_thresh
				counter -= 1
			counter += 1

		root_samples = [tmp[0] for tmp in root_samples_and_errors]
		errors_samples = [tmp[1] for tmp in root_samples_and_errors]

		assert max(errors_samples) <= converge_thresh

		filtered_quota = 1.0-len(root_samples)/float(nr_of_samples)
		print("How many were filtered:", 100*(1.0-len(root_samples)/float(nr_of_samples)),"%")

		if filtered_quota > 5:
			# TODO: Do something in this case
			assert False

		# Step 2
		# Run a symbolic regression to find a good approximation for the root.
		# This is used as a starting point.
		(cgp, best_err, parameters) = starting_point_approximation_symbolic_regression(root_samples, parameter_samples, optimizer, max_iter=max_iter, nr_of_parameters=nr_of_parameters_in_cgp, max_time=max_time)

		# Step 2 and a half
		# The symbolic regression (tries to) ignore all constant solutions, so those should be checked as well.
		mean = sum(root_samples)/float(len(root_samples))
		error_from_mean = sqrt(sum((r-mean)*(r-mean) for r in root_samples) / float(len(root_samples)))
		if error_from_mean < best_err:
			print("DOING THE CONST THING:", error_from_mean)
			# Create a new gene that represents a constant func
			new_gene = [0] * len(cgp.gene)
			new_gene[-1] = cgp.dims+0

			cgp = CGP(cgp.dims, cgp.op_table, new_gene, nr_of_parameters=1)
			parameters = [mean]

			for _ in range(10):
				assert cgp.eval([random() for _ in range(cgp.dims)], parameters=parameters) == mean
			best_err = error_from_mean

		return (cgp, best_err, parameters)
