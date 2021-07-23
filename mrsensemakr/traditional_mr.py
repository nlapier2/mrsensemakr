import math
import sys
from scipy.stats import pearsonr, t
import pandas as pd
from sensemakr import sensitivity_stats


def ils(fs, rf, instrument):
	"""
	Given first stage and reduced form regression models, produces the Indirect Least Squares (ILS) estimates.
		Also known as the "ratio estimate".

	:param fs: a statsmodels OLSResults object containing the "first stage" regression (exposure on instrument)
	:param rf: a statsmodels OLSResults object containing the "reduced form" regression (outcome on instrument)
	:param instrument: a string with the instrument variable name
	:return: a Pandas DataFrame containing the ILS effect and standard error
	"""
	summ_fs = sensitivity_stats.model_helper(model=fs, covariates=instrument)
	summ_rf = sensitivity_stats.model_helper(model=rf, covariates=instrument)
	rho = calc_rho(fs, rf)
	estimates = ils_estimates(summ_fs['estimate'], summ_rf['estimate'], summ_fs['se'], summ_rf['se'], rho)
	return estimates


def ils_estimates(fs_coef, rf_coef, fs_se, rf_se, rho):
	"""
	Given first state and reduced form coefficients and standard errors, produce the ILS estimates.

	:param fs_coef: Coefficient (float) from the first stage regression (exposure on instrument)
	:param rf_coef: Coefficient (float) from the reduced form regression (outcome on instrument)
	:param fs_se: Standard error (float) of the first stage regression
	:param rf_se: Standard error (float) of the reduced form regression
	:param rho: (float) Correlation between first stage residuals and reduced form residuals
	:return: a Pandas DataFrame containing the ILS effect and standard error
	"""
	iv_estimate = ils_coef(fs_coef, rf_coef)
	iv_se = ils_se(fs_coef, rf_coef, fs_se, rf_se, rho)
	return pd.DataFrame({'estimate': iv_estimate, 'se': iv_se})


def ils_coef(fs_coef, rf_coef):
	"""
	Given the first stage and reduced form coefficients, calculate the ratio-estimated ILS (MR) coefficient.

	:param fs_coef: Coefficient (float) from the first stage regression (exposure on instrument)
	:param rf_coef: Coefficient (float) from the reduced form regression (outcome on instrument)
	:return: a float, tau, representing the estimated coefficient for the ILS regression, e.g. the MR effect estimate
	"""
	tau = rf_coef / fs_coef
	return tau


def ils_var(fs_coef, rf_coef, fs_se, rf_se, rho):
	"""
	Given the first stage and reduced form coefficients and standard errors, compute the ILS variance.

	:param fs_coef: Coefficient (float) from the first stage regression (exposure on instrument)
	:param rf_coef: Coefficient (float) from the reduced form regression (outcome on instrument)
	:param fs_se: Standard error (float) of the first stage regression
	:param rf_se: Standard error (float) of the reduced form regression
	:param rho: (float) Correlation between first stage residuals and reduced form residuals
	:return: {...}
	"""
	tau = ils_coef(fs_coef, rf_coef)
	fs_var = fs_se ** 2
	rf_var = rf_se ** 2
	var_tau = (1 / (fs_coef**2)) * (rf_var + (tau ** 2) * fs_var - 2 * tau * rho * math.sqrt(fs_var * rf_var))
	return var_tau


def ils_se(fs_coef, rf_coef, fs_se, rf_se, rho):
	""" Returns the square root of ILS Variance (see above) """
	se_tau = math.sqrt(ils_var(fs_coef, rf_coef, fs_se, rf_se, rho))
	return se_tau


# Get summary statistics from RF and FS regressions

def calc_rho(fs, rf):
	"""
	Calculate the correlation between the first stage and reduced form residuals.

	:param fs: a statsmodels OLSResults object containing the "first stage" regression (exposure on instrument)
	:param rf: a statsmodels OLSResults object containing the "reduced form" regression (outcome on instrument)
	:return: (float) Correlation between first stage residuals and reduced form residuals
	"""
	return pearsonr(fs.resid, rf.resid)[0]


def iv_model_helper(fs, rf, instrument):
	"""
	A helper function for extracting information from the first stage (fs) and reduced form (rf) regressions.

	:param fs: a statsmodels OLSResults object containing the "first stage" regression (exposure on instrument)
	:param rf: a statsmodels OLSResults object containing the "reduced form" regression (outcome on instrument)
	:param instrument: a string with the instrument variable name
	:return: a dictionary containing the coefficients, standard errors, and DOF of fs and rf regressions
	"""
	summ_fs = sensitivity_stats.model_helper(model=fs, covariates=instrument)
	summ_rf = sensitivity_stats.model_helper(model=rf, covariates=instrument)
	rho = calc_rho(fs, rf)
	if summ_fs['dof'] != summ_rf['dof']:
		sys.exit("Degrees of freedom of first-stage and reduced form regressions differ.")
	iv_data = {'fs_coef': summ_fs['estimate'], 'fs_se': summ_fs['se'],
				'rf_coef': summ_rf['estimate'], 'rf_se': summ_rf['se'], 'rho': rho, 'dof': summ_rf['dof']}
	return iv_data


def mr_estimates(fs, rf, exposure, outcome, instrument, alpha):
	"""
	Computes the "traditional" Mendelian Randomization (MR) estimate, standard error, t_value, p_value, and
		confidence interval.

	:param fs: a statsmodels OLSResults object containing the "first stage" regression (exposure on instrument)
	:param rf: a statsmodels OLSResults object containing the "reduced form" regression (outcome on instrument)
	:param exposure: a string with the exposure variable name
	:param outcome: a string with the outcome variable name
	:param instrument: a string with the instrument variable name
	:param alpha: a float with the significance level to use for hypothesis testing
	:return:
	"""
	trad_mr = ils(fs=fs, rf=rf, instrument=instrument)
	dof = rf.df_resid
	t_value = trad_mr['estimate'] / trad_mr['se']
	crit_value = t.ppf(1-(alpha/2), dof)
	ci_low = trad_mr['estimate'] - crit_value * trad_mr['se']
	ci_up = trad_mr['estimate'] + crit_value * trad_mr['se']
	p_value = 2 * (t.pdf(-abs(t_value), dof))
	# out = pd.DataFrame({'exposure': exposure, 'outcome': outcome, 'instrument': instrument,
	out = {'exposure': exposure, 'outcome': outcome, 'instrument': instrument,
			'estimate': float(trad_mr['estimate']), 'se': float(trad_mr['se']), 't_value': float(t_value),
			'ci_low': ci_low, 'ci_up': ci_up, 'p_value': float(p_value), 'dof': dof, 'alpha': alpha}  # )
	return out


def print_trad_mr(x, digits=3):
	"""
	Prints the results of a traditional MR analysis.

	:param x: a dictionary returned by the mr_estimates method above containing the traditional MR results
	:param digits: number of digits to show for printed floats
	"""
	print("Traditional MR results (2SLS)")
	print("  MR Estimate (95% CI):", str(round(float(x['estimate']), digits)),
			"(" + str(float(round(x['ci_low'], digits))) + " - " + str(round(float(x['ci_up']), digits)) + ")")
	if float(x['p_value']) < 2e-16:
		print("  P-value:  < 2e-16")
	else:
		print("  P-value:", str(float(x['p_value'])))
#
