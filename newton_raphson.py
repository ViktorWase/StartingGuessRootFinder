from math import fabs
from random import gauss

def newton_raphson(func, func_der, parameters, max_iter=1000, convg_lim=1.0e-12, x0=1.0e-8):
	"""
	A basic version of the Newton-Raphson root finding method. Note that both input and output
	are only allowed to be one dimensional.
	"""

	convg_lim_sqr = convg_lim*convg_lim
	x = x0
	for i in range(max_iter):
		f = func([x], parameters)
		f_dx = func_der([x], parameters)
		if f_dx == 0:
			break
		x = x - f/f_dx
		if f*f < convg_lim_sqr:
			break
	error = fabs(f)
	return (x, error)

def bisection_method(x_neg, x_pos, func, parameters, max_iter=1000, convg_lim=1.0e-12, x0=1.0e-8):
	assert func([x_neg], parameters) <= 0
	assert func([x_pos], parameters) >= 0

	for i in range(max_iter):
		x = 0.5*(x_neg+x_pos)
		f = func([x], parameters)

		if fabs(f) < convg_lim:
			break

		if f < 0:
			x_neg = x
		else:
			x_pos = x
	return (x, fabs(f))

def find_pos_and_negative(func, parameters):
	x_1 = gauss(0.0, 1.0)
	first_f = func([x_1], parameters)
	f_second = None

	for _ in range(1000):
		x_2 = gauss(0.0, 1.0)
		f = func([x_2], parameters)

		if f*first_f < 0.0:
			f_second = f
			break

	if f_second is None:
		return (None, None)

	if f_second < first_f:
		return (x_2, x_1)
	else:
		return (x_1, x_2)

def bisection_root_finder(func, parameters, max_iter=1000, convg_lim=1.0e-12):
	(x_neg, x_pos) = find_pos_and_negative(func, parameters)

	if x_neg is None:
		return (0.0, 1.0e10)

	return bisection_method(x_neg, x_pos, func, parameters, max_iter=max_iter, convg_lim=convg_lim)

def root_finders(func, func_der, parameters):
	(x, err) = newton_raphson(func, func_der, parameters)

	
	(x_b, err_b) = bisection_root_finder(func, parameters)
	if err_b < err:
		err = err_b
		x=x_b
	
	return (x, err)

