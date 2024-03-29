"""
Simulated anneling is an optimization method that we use to find suitable
functions. Note that it is used for discrete optimization only. 
The countinous optimization (that is used to find the correct values of the
numerical parameters/constants in the function) is Differential Evolution, and it is in
another file.
"""

from copy import copy, deepcopy
from random import random, randint
from time import time

from math import sqrt, exp
from cgpy.cgp import Operation, CGP

inf = float('inf')


def create_random_gene(dims, nr_of_funcs, nr_of_nodes):
	"""
	Creates a random gene that represents a random cgp function.
	Assumes that the operations are unary or binary.
	"""
	n = 3*nr_of_nodes + 1
	counter = 0
	gene = [None for _ in range(n)]
	for i in range(nr_of_nodes):
		gene[counter] = randint(0, nr_of_funcs - 1)
		counter += 1

		for _ in range(2):
			gene[counter] = randint(0, i+dims - 1)
			counter += 1
	gene[counter] = randint(0, nr_of_nodes-1+dims)
	return gene


def get_gene_max_values(dims, nr_of_parameters, len_of_op_table, nr_of_nodes, nodes_per_layer=1):
	"""
	A gene is a list of n ints, that define the CGP. Each such number has
	a minimum value, and a maximum value. The minimum value is always zero.
	This function will return a list of the n maximum values.
	"""
	# Check the inputdata
	assert nodes_per_layer >= 1
	assert nr_of_nodes > 0
	assert dims > 0
	assert nr_of_parameters >= 0

	# The number of nodes has to be divisible by nodes_per_layer.
	# Otherwise the number of layers won't be an int, and that is strange.
	assert nr_of_nodes % nodes_per_layer == 0

	dim_and_pars = dims + nr_of_parameters

	# Each node has 3 ints: the two inputs and the operation.
	len_of_gene = nr_of_nodes*3 + 1
	max_vals = [-1]*len_of_gene

	layer = 0
	for node_count in range(nr_of_nodes):
		nr_of_nodes_and_inputs_of_all_prev_layers = layer*nodes_per_layer + dim_and_pars

		max_vals[3*node_count+2] = nr_of_nodes_and_inputs_of_all_prev_layers - 1
		max_vals[3*node_count+1] = nr_of_nodes_and_inputs_of_all_prev_layers - 1
		max_vals[3*node_count] = len_of_op_table-1

		if node_count%nodes_per_layer == nodes_per_layer-1:
			layer += 1

	# The last int of the gene just points to one of the inputs, parameters or nodes and
	# calls it the outout.
	max_vals[-1] = dim_and_pars + nr_of_nodes - 1
	assert min(max_vals)>=0
	return max_vals


def mutate(cgp_in, dims, nr_of_funcs):
	"""
	Mutates the cgp. Doesn't affect the input CGP.

	Mutates parts randomly until it mutates a part of
	the gene that is used.
	"""
	assert cgp_in.has_setup_used_nodes
	new_gene = list(cgp_in.gene)
	nr_of_nondim_and_non_par_nodes = cgp_in.nr_of_nodes - cgp_in.dims - cgp_in.nr_of_parameters
	max_vals = get_gene_max_values(cgp_in.dims, cgp_in.nr_of_parameters, len(cgp_in.op_table), nr_of_nondim_and_non_par_nodes, nodes_per_layer=cgp_in.nodes_per_layer)

	assert int((len(cgp_in.gene)-1)/3) == len(cgp_in.used_nodes)

	# Mutate the gene.
	has_mutated_used_node = False
	n = len(new_gene)
	while not has_mutated_used_node:
		i = randint(0, n-1)
		new_gene[i] = randint(0, max_vals[i])

		# Check if the mutated part is used

		# The last element of the gene is always used
		if i == n-1:
			if new_gene[i] != cgp_in.gene[i]:
				has_mutated_used_node = True
		else:
			is_active = cgp_in.used_nodes[int(i/3)]

			if is_active and new_gene[i] != cgp_in.gene[i]:
				# If the second input is changed then we need to check
				# that it is a binary operation
				if i % 3 == 2:
					op = cgp_in.op_table[new_gene[i-2]]
					if op.is_binary:
						has_mutated_used_node = True
				else:
					has_mutated_used_node = True

	# Create the new CGP object
	new_cgp = CGP(cgp_in.dims, cgp_in.op_table, new_gene, nr_of_parameters=cgp_in.nr_of_parameters, fast_setup=not cgp_in.has_setup_used_nodes)
	return new_cgp


def mutate_old(cgp_in, dims, nr_of_funcs, mute_rate=0.4):
	"""
	Mutates the cgp. Doesn't affect the input CGP.

	One of the USED parts of the gene is always mutated.
	"""
	gene = list(cgp_in.gene)
	nodes = int((len(gene)-1)/3)

	nr_of_used_parts = sum(cgp_in.used_nodes)+1  # The +1 is for the last number in the gene, which decides which node that is the output. THe -1 is for indexing.
	used_part_2_mutate = 0 if nr_of_used_parts<=0 else randint(0, nr_of_used_parts-1)
	used_part_counter = 0
	counter = 0
	has_forced_mutated = False
	for i in range(nodes):

		# Make sure at least some part of the USED gene is mutated...
		# TODO: REMOVE THE WHOLE THING WHER EIT MUTATES INTO SOMETHING NEW. It can mutate into itself.
		if cgp_in.used_nodes[i]:
			if used_part_counter == used_part_2_mutate:
				assert has_forced_mutated == False
				has_forced_mutated = True
				is_binary = cgp_in.op_table[gene[counter]].is_binary

				random_node = randint(0, 2 + (1 if is_binary else 0)-1)
				if random_node == 0:
					has_changed = False
					while not has_changed:
						old_val = gene[counter]
						gene[counter] = randint(0, nr_of_funcs-1)
						if gene[counter] != old_val:
							has_changed = True
				else:
					assert random_node==1 or random_node==2
					has_changed = False
					while not has_changed:
						old_val = gene[counter+random_node]
						gene[counter+random_node] = randint(0,i+dims - 1)
						if gene[counter+random_node] != old_val:
							has_changed = True
			#counter += 3
			used_part_counter += 1

		
		#... the other parts don't have to mutate
		if random()<mute_rate:
			gene[counter] = randint(0, nr_of_funcs-1)
		counter += 1

		for _ in range(2):
			if random()<mute_rate:
				gene[counter] = randint(0,i+dims - 1)
			counter += 1
	if random() < mute_rate or nr_of_used_parts-1 == used_part_2_mutate:
		assert counter == len(gene)-1
		if nr_of_used_parts-1 == used_part_2_mutate:
			assert has_forced_mutated == False
		has_forced_mutated = True
		old_val = gene[counter]
		gene[counter] = randint(0, nodes-1+dims)
	assert has_forced_mutated

	# This tmp is just to make sure it does not cause a seg-fault.
	start_node = gene[-1]
	tmp = gene[3*(start_node-dims)]

	assert counter == len(gene)-1
	assert dims == cgp_in.dims+cgp_in.nr_of_parameters
	return CGP(cgp_in.dims, cgp_in.op_table, gene, nr_of_parameters=cgp_in.nr_of_parameters)


def acceptance_prob(new_error, current_error, temp):
	"""
	The acceptance probability depends on the difference in the objective
	function as well as the "temperature" of the system. Higher temperature means that anything could happen.
	"""
	diff = new_error - current_error
	return exp(-(diff)/temp)


def sa(f_vals, pnts, dims, nr_of_funcs, nr_of_nodes, error_func, op_table, optimization_data, max_iter=500, nr_of_pars=0, reheat_iter=-1, remaining_time=None):
	"""
	Simulated anneling is a simple way of doing compinatorial optimization without getting stuck in local minima.
	It basically works like this:

	1) take the current solution and apply a small change to it

	2) If this new solution is better, keep it.

	3) There is a chance that the new solution is kept even if it is worse.
	   This chance decreases as the iteration number grows, and the chance
	   is small if the new solution is much worse than the old solution.

	4) Repeate the process a bit, and return the best solution.

	The wiki page is rather good as well.
	"""

	start_time = time()

	assert nr_of_funcs == len(op_table)
	# Create a starting function (solution) at random.
	current_sol = create_random_gene(dims+nr_of_pars, nr_of_funcs, nr_of_nodes)

	current_cgp = CGP(dims, op_table, current_sol, nr_of_parameters=nr_of_pars)

	(current_error, best_pars) = error_func(f_vals, pnts, dims, current_cgp, nr_of_pars, op_table, optimization_data)

	print(current_error)
	#assert False

	best_cgp = deepcopy(current_cgp)
	best_error = current_error

	iterations_since_update = 0

	temperature_itr = 0

	for itr in range(max_iter):
		if itr % 50 == 0:
			print("iter:", itr, " of", max_iter)
		temp = float(max_iter-temperature_itr)/max_iter
		# Do a small mutation to create a new function (aka solution)
		new_cgp = mutate(current_cgp, dims+nr_of_pars, nr_of_funcs)
		assert current_cgp.nr_of_parameters == new_cgp.nr_of_parameters
		#cgp = CGP(dims, op_table, new_sol, nr_of_parameters=nr_of_pars)
		(new_error, new_pars) = error_func(f_vals, pnts, dims, new_cgp, nr_of_pars, op_table, optimization_data)

		temperature_itr += 1

		if new_error < current_error or acceptance_prob(new_error, current_error, temp) < random():
			#current_sol = new_sol
			current_cgp = new_cgp
			current_error = new_error

			if new_error < best_error:
				print("best yet:", new_error, " at time:", time())
				#write_2_file(new_pars, itr, current_error, 'sa',current_cgp.convert2str(parameters=new_pars))
				new_cgp.print_function(parameters=new_pars)
				best_cgp = deepcopy(new_cgp)
				best_error = new_error
				best_pars = list(new_pars)
		else:
			iterations_since_update += 1

			# If no change has been made in a while, then we set the temp to max again!
			if iterations_since_update == reheat_iter:
				temperature_itr = 0
				iterations_since_update = 0
				print("Reheating.")

		if remaining_time is not None and time()-start_time >= remaining_time:
			break

	return (best_cgp, best_error, best_pars)


def multistart_opt(f_vals, pnts, dims, nr_of_funcs, nr_of_nodes, error_func, op_table, optimizer, optimization_data, max_iter=1000, multi_starts=10, nr_of_pars=0, max_time=None):
	"""
	A multistart version of simulated anneling/es. Returns the best found solution.
	"""
	best_err = inf
	best_sol = None
	best_pars = None

	if max_time is not None:
		multi_starts = None

	counter = 0

	start_time = time()
	remaining_time = None

	while True:
		print("STARTING NEW:", counter+1, "of", multi_starts)

		if max_time is not None:
			passed_time = time() - start_time
			remaining_time = max_time - passed_time
		if optimizer == "sa":
			(sol, err, pars) = sa(f_vals, pnts, dims, nr_of_funcs, nr_of_nodes, error_func, op_table, optimization_data, max_iter=max_iter, nr_of_pars=nr_of_pars, remaining_time=remaining_time)
		else:
			print(optimizer, "is not a used optimizer.")
			assert False

		if err < best_err:
			best_err = err
			best_sol = sol
			best_pars = pars
		print(pars, nr_of_pars)
		assert len(pars) == nr_of_pars

		counter += 1

		if multi_starts is not None:
			if counter == multi_starts:
				break
		else:
			if time() - start_time >= max_time:
				print("STOPPING")
				break
			else:
				print(time() - start_time, max_time)
	return (best_sol, best_err, best_pars)
