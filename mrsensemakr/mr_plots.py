# Produce contour plots by calling the equivalent sensemakr function

import sys
import numpy as np
from scipy.stats import t
from . import mrsensemakr
from .PySensemakr.sensemakr import ovb_plots, sensitivity_stats


def plot_mr_sensemakr(x, var_type, benchmark_covariates=None, k=1, alpha=None, lim_x=None, lim_y=None):
	"""
	Creates an omitted variable bias contour plot using the Sensemakr functions, depicting the sensitivity either of
		the first stage (instrument-exposure) or reduced form (instrument-outcome) regressions.

	:param x: an MRSensemakr object (see mrsensemakr.py)
	:param var_type: a string, either "outcome" or "exposure", indicating which is being shown in the plot
	:param benchmark_covariates: a string or list of strings with names of the variables to use for benchmark bounding
	:param k: a float or list of floats with each being a multiple of the strength of association between a
					benchmark variable and the treatment variable to test with benchmark bounding
	:param alpha: a float with the significance level for the robustness value RV_qa to render the
					estimate not significant
	:param lim_x: the upper limit of the plot's x-axis (lower limit is 0)
	:param lim_y: the upper limit of the plot's y-axis (lower limit is 0)
	"""
	if var_type == 'outcome':
		model = x.out['outcome']['model']
	elif var_type == 'exposure':
		model = x.out['exposure']['model']
	else:
		sys.exit("Error: 'var_type' must be 'outcome' or 'exposure'.")

	if alpha is None:
		alpha = x.out['info']['alpha']
	if alpha < 0 or alpha > 1:
		sys.exit('Error: alpha must be between 0 and 1.')
	dof = model.df_resid
	t_value = model.tvalues[x.out['info']['instrument']]
	t_thr = abs(t.ppf(alpha / 2.0, df = dof - 1)) * np.sign(t_value)

	model_data = sensitivity_stats.model_helper(model, covariates=x.out['mr']['instrument'])
	estimate = float(model_data['estimate'])
	se = float(model_data['se'])
	dof = float(model_data['dof'])
	rv = float(x.out[var_type]['sensitivity']['rv'])

	bounds = None
	if benchmark_covariates is not None:
		benchmark_covariates = [mrsensemakr.clean_benchmarks(i, model) for i in benchmark_covariates]
		bounds = mrsensemakr.multiple_bounds(model=model, covariate=x.out['mr']['instrument'],
			benchmark_covariates=benchmark_covariates, k=k, alpha=alpha)

		max_r2dz_x = max(bounds['r2dz_x'] * 1.5, rv * 1.5)
		max_r2yz_dx = max(bounds['r2yz_dx'] * 1.5, rv * 1.5)
		if lim_x is None:
			lim_x = max_r2dz_x
		if lim_y is None:
			lim_y = max_r2yz_dx

	if lim_x is None:
		lim_x = rv * 1.5
	if lim_y is None:
		lim_y = rv * 1.5
	xlab = "Partial R^2 of unobservables with genetic instrument"
	if var_type == 'outcome':
		ylab = "Partial R^2 of unobservables with outcome trait"
	else:
		ylab = "Partial R^2 of unobservables with exposure trait"

	ovb_plots.ovb_contour_plot(model=model, treatment=x.out['mr']['instrument'], sensitivity_of='t-value', t_threshold=t_thr, xlab=xlab,
		ylab=ylab, lim=lim_x, lim_y=lim_y)
	#ovb_plots.ovb_contour_plot(estimate=estimate, se=se, dof=dof, treatment=x.out['mr']['instrument'],
	#							sensitivity_of='t-value', xlab=xlab, ylab=ylab, lim=lim_x, lim_y=lim_y)  #,
								#r2yz_dx=[0.1], r2dz_x=[0.1])
	if benchmark_covariates is not None:
		ovb_plots.add_bound_to_contour(bounds=bounds, treatment=x.out['mr']['instrument'])
#
