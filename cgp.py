"""
CGP stands for Cartesian Genetic Programming, and is a network that 
represents a function. It's really nice. Google it!
"""

from math import sin, cos, sqrt, log, pow, exp, fabs, asin, acos, pi
from random import randint, random
from copy import copy

from dual_numbers import sind, cosd, expd, logd, sqrtd, powd, asind, acosd, DualNumber

class Operation():
	"""
	A mathematical operation. We have a couple of the elementary ones, but 
	more might be added later. 

	Each object has:
		- a name by which it is identified
		- the function itself
		- a dual version of the function. This means 
		  that dual numbers are used.
	"""
	def __init__(self, op_name):
		self.op_name = op_name
		if op_name == "sin":
			self.func = lambda x: sin(x)
			self.dual_func = lambda x: sind(x)
			self.is_binary = False
		elif op_name == "cos":
			self.func = lambda x: cos(x)
			self.dual_func = lambda x: cosd(x)
			self.is_binary = False
		elif op_name == "acos":
			self.func = lambda x: acos(x) if x<1 and x>-1 else 0.0
			self.dual_func = lambda x: acosd(x) if x.a<1 and x.a>-1 else DualNumber(0.0,0.0)
			self.is_binary = False
		elif op_name == "asin":
			self.func = lambda x: asin(x) if x<1 and x>-1 else 0.0
			self.dual_func = lambda x: asind(x) if x.a<1 and x.a>-1 else DualNumber(0.0,0.0)
			self.is_binary = False
		elif op_name == "+":
			self.func = lambda x,y: x+y
			self.dual_func = lambda x,y: x+y
			self.is_binary = True
		elif op_name == "-":
			self.func = lambda x,y: x-y
			self.dual_func = lambda x,y: x-y
			self.is_binary = True
		elif op_name == "*":
			self.func = lambda x,y: x*y
			self.dual_func = lambda x,y: x*y
			self.is_binary = True
		elif op_name == "sqr":
			self.func = lambda x: x*x
			self.dual_func = lambda x: x*x
			self.is_binary = False
		elif op_name == "log":
			self.func = lambda x: log(x) if x>0 else 0.0
			self.dual_func = lambda x: logd(x) if x.a>0 else DualNumber(0.0, 0.0)
			self.is_binary = False
		elif op_name == "/":
			self.func = lambda x, y: x/y if y!=0 else 0.0
			self.dual_func = lambda x, y: x/y if y.a!=0 else DualNumber(0.0, 0.0)
			self.is_binary = True
		elif op_name == "id":
			self.func = lambda x: x
			self.dual_func = lambda x: x
			self.is_binary = False
		else:
			assert(False)
		self.str = copy(op_name)


# Functions for converting a CGP function to a string
def convert(op_table, gene, variable_names, nr_of_nodes, dims, parameters=[]):
	# We start at the end node and recursively work our way back.
	assert ( nr_of_nodes > 0 )
	assert ( dims > 0 )
	assert ( len(variable_names) >= dims )

	current_node_nr = gene[-1]
	return convert_rec(op_table, gene, variable_names, nr_of_nodes, dims + len(parameters), len(parameters), current_node_nr, parameters=parameters)

def convert_rec(op_table, gene, variable_names, nr_of_nodes, total_dims, nr_of_parameters, current_node_nr, parameters=[]):
	assert(nr_of_parameters == len(parameters))
	if(current_node_nr < total_dims):
		nr_of_vars = total_dims-nr_of_parameters
		if(current_node_nr < nr_of_vars):
			return variable_names[current_node_nr]
		else:
			return str(parameters[current_node_nr - nr_of_vars])

	op = op_table[gene[3 * (current_node_nr-total_dims) + 0]]
	nr_of_vars = total_dims-nr_of_parameters

	if op.is_binary:
		left_str = None
		right_str = None
		if( gene[3 * (current_node_nr-total_dims) + 1] < total_dims ):

			if gene[3 * (current_node_nr-total_dims) + 1] < nr_of_vars:
				left_str = variable_names[gene[3 * (current_node_nr-total_dims) + 1]]
			else:
				left_str = str(parameters[gene[3 * (current_node_nr-total_dims) + 1] - nr_of_vars])

		else:
			assert( gene[3 * (current_node_nr-total_dims) + 1] < current_node_nr )
			left_str = "("+convert_rec(op_table, gene, variable_names, nr_of_nodes, total_dims, nr_of_parameters, gene[3 * (current_node_nr-total_dims) + 1], parameters=parameters)+")"

		if( gene[3 * (current_node_nr-total_dims) + 2] < total_dims ):
			if gene[3 * (current_node_nr-total_dims) + 2] < nr_of_vars:
				right_str = variable_names[gene[3 * (current_node_nr-total_dims) + 2]]
			else:
				right_str = str(parameters[gene[3 * (current_node_nr-total_dims) + 2] - nr_of_vars])

		else:
			assert( gene[3 * (current_node_nr-total_dims) + 2] < current_node_nr )
			right_str = "("+convert_rec(op_table, gene, variable_names, nr_of_nodes, total_dims, nr_of_parameters, gene[3 * (current_node_nr-total_dims) + 2], parameters=parameters)+")"
		return left_str+op.str+right_str
	else:

		middle_str = None
		if( gene[3 * (current_node_nr-total_dims) + 1] < total_dims ):

			if gene[3 * (current_node_nr-total_dims) + 1] < nr_of_vars:
				middle_str = variable_names[gene[3 * (current_node_nr-total_dims) + 1]]
			else:
				middle_str = str(parameters[gene[3 * (current_node_nr-total_dims) + 1] - nr_of_vars])

		else:
			middle_str = convert_rec(op_table, gene, variable_names, nr_of_nodes, total_dims, nr_of_parameters, gene[3 * (current_node_nr-total_dims) + 1], parameters=parameters)
		if op.str == "sqr":
			return "("+middle_str+")^{2}"
			
		return op.str+"("+middle_str+")"

class CGP():
	"""
	A Cartesian Genetic Programming object.
	This is a way of denoting a mathematical function as
	a "gene" that can be used in evolutionary optimizations.
	"""
	def __init__(self, dims, op_table, gene, nr_of_parameters=0):
		assert len(gene)>0
		assert dims > 0
		assert len(op_table) > 0
		assert nr_of_parameters >= 0
		
		self.op_table = op_table #NOTE: We only copy by reference here to speed it up a little.
		self.gene = list(gene)
		self.nr_of_parameters = nr_of_parameters
		self.dims = dims

		self.is_constant = None

		self.gene_sanity_check()

		self.setup_used_genes_list()

		self.nr_of_nodes = int((len(self.gene)-1)/3)+self.dims+self.nr_of_parameters

		assert(dims > 0)

	def gene_sanity_check(self):
		nr_of_ins = self.dims + self.nr_of_parameters

		gene_counter = 0
		for i in range(int((len(self.gene)-1)/3)):
			assert self.gene[gene_counter] < len(self.op_table)
			gene_counter += 1
			assert self.gene[gene_counter] < i+nr_of_ins
			gene_counter += 1
			assert self.gene[gene_counter] < i+nr_of_ins
			gene_counter += 1

	def setup_used_genes_list(self):
		# Nr of nodes excluding the output node
		nr_of_nodes = int((len(self.gene)-1)/3)+self.dims+self.nr_of_parameters

		class CGPNode():
			"""Temporary node object"""
			def __init__(self, upstream1, upstream2, is_used=False):
				self.upstream1 = upstream1
				self.upstream2 = upstream2
				self.is_used = is_used

			def update_is_used(self):
				self.is_used = True
				if self.upstream1 != None:
					self.upstream1.update_is_used()
				if self.upstream2 != None:
					assert self.upstream1 != None
					self.upstream2.update_is_used()

		nodes = [None for _ in range(nr_of_nodes)]

		gene_counter = 0
		for i in range(nr_of_nodes):
			if i < self.dims + self.nr_of_parameters:
				nodes[i] = CGPNode(None, None)
			else:
				op = self.op_table[self.gene[gene_counter]]
				gene_counter += 1

				if op.is_binary:
					nodes[i] = CGPNode(nodes[self.gene[gene_counter]], nodes[self.gene[gene_counter+1]])
					assert nodes[self.gene[gene_counter]] != None
					assert nodes[self.gene[gene_counter+1]] != None
				else:
					nodes[i] = CGPNode(nodes[self.gene[gene_counter]], None)
					assert nodes[self.gene[gene_counter]] != None

				gene_counter += 2
		assert gene_counter == len(self.gene)-1

		nodes[self.gene[gene_counter]].update_is_used()

		#See if any variables are used
		is_any_varible_used = False
		for d in range(self.dims):
			if nodes[d].is_used:
				is_any_varible_used = True
		self.is_constant = not is_any_varible_used

		# Remove the parameters and variables 
		nodes = nodes[self.dims+self.nr_of_parameters:]

		assert len(nodes) == int((len(self.gene)-1)/3)
		self.used_genes = [node.is_used for node in nodes]


	def eval(self, X, parameters = [], derivative=False, der_dir=0):
		"""
		Evaluates the function at point X using the parameters
		in parameters.

		If derivative is true, then it takes the derivative in the point as well.
		It return the tuple (function_val, derivative_val) in that case.

		der_dir is the direction of the derivative. Which means that der_dir=0 => derivative
		with respect to X[0], and so on. If der_dir >= len(X), then we will start deriving 
		with respect to the parameters
		"""
		if derivative:
			assert der_dir < len(X) + len(parameters)
			assert der_dir >= 0
		assert(self.nr_of_parameters == len(parameters))
		if self.dims != len(X):
			print("There is a mismatch in dimensions in CGP. There should be", self.dims, " but the input is ", len(X))
		assert(self.dims == len(X))

		# Combined dimensionality of the variables and parameters.
		total_dims = len(X) + len(parameters)

		# Okay, so this is a litte weird. But n is the total number of 
		# nodes used. A node is something that has a value and (possibly)
		# connections to other nodes. The returned value is the value of
		# the last node.
		n = int((len(self.gene)-1)/3) + total_dims
		all_node_vals = [None] * n 

		# Convert to dual number if we want the derivative
		if derivative:
			X = [DualNumber(x, 0) for x in X]

		# The inputs (variables and parameters) are the first nodes.
		for i in range(total_dims):
			if i < len(X):
				all_node_vals[i] = X[i]
			else:
				# Convert to dual number if we want the derivative
				if derivative:
					all_node_vals[i] = DualNumber(parameters[i-len(X)], 0.0)
				else:
					all_node_vals[i] = parameters[i-len(X)]
		if derivative:
			all_node_vals[der_dir].b = 1.0

		# Okay, so let's step thru all the other nodes.
		node_nr = total_dims
		gene_counter = 0
		for node_nr in range(total_dims, n):
			if self.used_genes[node_nr-total_dims]:
				assert(gene_counter<len(self.gene))
				assert(self.gene[gene_counter]<len(self.op_table))

				# All the nodes (except for the inputs) have an
				# operation (such as +, - cos()) and connections
				# to 1 or 2 (depending on the operation) "older"
				# nodes. "Older" means that the nodes are found
				# earlier in the list.
				op = self.op_table[self.gene[gene_counter]]
				gene_counter += 1
				node_val = None

				# The node has 2 connections if the operation is binary,
				# and 1 connection otherwise.
				if op.is_binary:
					x1 = all_node_vals[self.gene[gene_counter]]
					gene_counter += 1
					x2 = all_node_vals[self.gene[gene_counter]]
					gene_counter += 1

					if derivative:
						node_val = op.dual_func(x1, x2)
					else:
						node_val = op.func(x1, x2)
				else:
					x = all_node_vals[self.gene[gene_counter]]
					gene_counter += 1
					gene_counter += 1

					if derivative:
						node_val = op.dual_func(x)
					else:
						node_val = op.func(x)
				assert( all_node_vals[node_nr] == None)
				all_node_vals[node_nr] = node_val
			else:
				gene_counter += 3
		#assert(sum([x==None for x in all_node_vals])==0)
		assert(sum([x==None for x in all_node_vals]) == sum(x==False for x in self.used_genes))
		assert all_node_vals[self.gene[gene_counter]] != None
		assert gene_counter == len(self.gene)-1

		# Return the value of the last node.
		if not derivative:
			return all_node_vals[self.gene[gene_counter]]
		else:
			return (all_node_vals[self.gene[gene_counter]].a, all_node_vals[self.gene[gene_counter]].b)

	def convert2str(self, parameters=[], var_names=None):
		if len(parameters) != self.nr_of_parameters:
			print("Wrong number of parameters in the convert2str function.")

		assert len(parameters) == self.nr_of_parameters

		if var_names == None:
			var_names = ["x"+str(i+1) for i in range(self.dims)]

		total_dims = len(parameters) + self.dims
		nr_of_nodes = int((len(self.gene)-1)/3) + total_dims

		return convert(self.op_table, self.gene, var_names, nr_of_nodes, self.dims, parameters=parameters)

	def print_function(self, parameters=[], var_names=None):
		print(self.convert2str(parameters=parameters, var_names=var_names))

	def merge_lists(self, x1, x2):
		out = []

		for x in x1:
			if x not in out:
				out.append(x)
		for x in x2:
			if x not in out:
				out.append(x)
		return out

	def which_variables_and_parameters_are_used(self):
		# Combined dimensionality of the variables and parameters.
		total_dims = self.dims + self.nr_of_parameters
		n = int((len(self.gene)-1)/3) + total_dims
		node_depends_on_which_nodes = [[] for _ in range(n)]

		gene_counter = 0

		for i in range(total_dims):
			if i < self.dims:
				node_depends_on_which_nodes[i].append(i)
			else:
				node_depends_on_which_nodes[i].append(i)

		for node_nr in range(total_dims, n):
			op = self.op_table[self.gene[gene_counter]]
			gene_counter += 1

			# The node has 2 connections if the operation is binary,
			# and 1 connection otherwise.
			if op.is_binary:
				x1 = node_depends_on_which_nodes[self.gene[gene_counter]]
				gene_counter += 1
				x2 = node_depends_on_which_nodes[self.gene[gene_counter]]
				gene_counter += 1

				x = self.merge_lists(x1, x2)
			else:
				x = list(node_depends_on_which_nodes[self.gene[gene_counter]])
				gene_counter += 1
				gene_counter += 1

			node_depends_on_which_nodes[node_nr] = x

		assert gene_counter == len(self.gene)-1
		return sorted(node_depends_on_which_nodes[self.gene[gene_counter]])

	def which_parameters_are_used(self):
		# TODO: This shouldn't be re-calculated every time.
		var_and_par = self.which_variables_and_parameters_are_used()

		is_parameter_used = [False for _ in range(self.nr_of_parameters)]

		for x in var_and_par:
			# Ignore the variables, and only focus on the parameters.
			if x >= self.dims:
				par_nr = x - self.dims
				is_parameter_used[par_nr] = True
		return is_parameter_used

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


if __name__ == '__main__':
	"""
	# This are all operations that the CGP is allowed to use. These are not set in stone.
	op_table = [Operation("+"), Operation("*"), Operation("sin"), Operation("cos"), Operation("sqr"), Operation("-"), Operation("log"), Operation("/")]
	nr_of_funcs = len(op_table)

	from simulated_anneling import create_random_gene

	dims = 2
	nr_of_nodes = 5
	cgp_gene = create_random_gene(dims, nr_of_funcs, nr_of_nodes)
	cgp = CGP(dims, op_table, cgp_gene)
	print(cgp.convert2str())
	print(cgp.which_variables_and_parameters_are_used())
	"""

	dims = 3
	nr_of_nodes = 3
	print(how_many_genes_exists(dims, nr_of_nodes))