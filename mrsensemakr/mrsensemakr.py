# import pandas as pd
import statsmodels.formula.api as smf
from . import traditional_mr
from .PySensemakr.sensemakr import ovb_bounds
from .PySensemakr.sensemakr import sensitivity_stats


class MRSensemakr:

	def __init__(self, outcome, exposure, instrument, data, covariates=None, 
	benchmark_covariates=None, k=1, alpha=0.05):
		"""MRSensemakr

		:param outcome: name of the outcome trait.
		:type outcome: str
		:param exposure: name of the exposure trait.
		:type exposure: str
		:param instrument: name of the genetic instrument.
		:type instrument: str
		:param data: numpy array containing the variables used in the analysis.
		:type data: np.array
		:param covariates: name of the control covariates, such as age, sex, genomic principal components, batch effect dummies and putative pleiotropic pathways., defaults to None
		:type covariates: list(str), optional
		:param benchmark_covariates: The user has two options: (i) list of the names of covariates that will be used to bound the plausible strength of the unobserved confounders. Each variable will be considered separately; (ii) a list of lists of covariates that will be used, as a group, to bound the plausible strength of the unobserved confounders. The names of the list will be used for the benchmark labels., defaults to None
		:type benchmark_covariates: list(str), optional
		:param k: Parameterizes how many times stronger residual biases are related to the treatment and the outcome in comparison to the observed benchmark covariates., defaults to 1
		:type k: int, optional
		:param alpha: significance level, defaults to 0.05
		:type alpha: float, optional
		"""
		self.outcome = outcome
		self.exposure = exposure
		self.instrument = instrument
		self.data = data
		self.covariates = covariates
		self.benchmark_covariates = benchmark_covariates
		self.k = k
		self.alpha = alpha

		self.out = {}
		self.out['info'] = {'outcome': self.outcome, 'exposure': self.exposure,
			'instrument': self.instrument, 'covariates': self.covariates, 'alpha': self.alpha}

		# first stage
		if self.covariates is None:
			fs_form = make_formula(self.exposure, [self.instrument])
		else:
			fs_form = make_formula(self.exposure, [self.instrument] + self.covariates)
		first_stage = smf.ols(formula=fs_form, data = self.data)
		fitted_fs = first_stage.fit()

		# reduced form
		if self.covariates is None:
			rf_form = make_formula(self.outcome, [self.instrument])
		else:
			rf_form = make_formula(self.outcome, [self.instrument] + self.covariates)
		reduced_form = smf.ols(formula=rf_form, data = self.data)
		fitted_rf = reduced_form.fit()

		# traditional mr
		trad_mr = traditional_mr.mr_estimates(fitted_fs, fitted_rf,
			self.exposure, self.outcome, self.instrument, self.alpha)
		self.out['mr'] = trad_mr

		# first stage sensitivity
		fs_sense = sense_trait(fitted_fs, self.instrument, self.exposure, self.alpha, 'exposure')
		self.out['exposure'] = {'model': fitted_fs, 'sensitivity': fs_sense}

		# reduced form sensitivity
		rf_sense = sense_trait(fitted_rf, self.instrument, self.outcome, self.alpha, 'outcome')
		self.out['outcome'] = {'model': fitted_rf, 'sensitivity': rf_sense}

		# bounds
		if self.benchmark_covariates is not None:
			self.benchmark_covariates = [clean_benchmarks(i, fitted_fs)
				for i in self.benchmark_covariates]

			fs_bounds = ovb_bounds.ovb_partial_r2_bound(model=fitted_fs, treatment=self.instrument,
				benchmark_covariates=self.benchmark_covariates, kd=self.k)
			# names(fs.bounds)[2:3] <- c("r2zw.x", "r2dw.zx")
			self.out['exposure']['bounds'] = fs_bounds

			rf_bounds = ovb_bounds.ovb_partial_r2_bound(model=fitted_rf, treatment=self.instrument,
				benchmark_covariates=self.benchmark_covariates, kd=self.k)
			# names(rf.bounds)[2:3] <- c("r2zw.x", "r2yw.zx")
			self.out['outcome']['bounds'] = rf_bounds


def print_mr_sensemakr(x, digits=3):
	"""
	Print the sensitivity statistics for an MRSensemakr object.

	:param x: an MRSensemakr object (see above)
	:type x: MRSensemakr object
	:param digits: number of digits to show for printed floats, defaults to 3
	:type digits: int, optional 
	:return: None
	"""
	print("Sensitivity Analysis for Mendelian Randomization (MR)")
	print(" Exposure: " + str(x.out['mr']['exposure']))
	print(" Outcome: " + str(x.out['mr']['outcome']))
	print(" Genetic instrument: " + str(x.out['mr']['instrument']) + "\n")
	# print(str(x.out['mr']))
	traditional_mr.print_trad_mr(x.out['mr'], digits=digits)
	print()
	print_sense_trait(x.out['exposure']['sensitivity'], digits=digits)
	print()
	print_sense_trait(x.out['outcome']['sensitivity'], digits=digits)
	if 'bounds' in x.out['exposure'] and 'bounds' in x.out['outcome']:
		print("Bounds on the maximum explanatory power of omitted variables W, if it were as strong as:")
		# bounds <- cbind(x$exposure$bounds, x$outcome$bounds[,3,drop = F])
		# bounds[,2:4] <-  lapply(
		#    bounds[,2:4], function(x) paste0(round(x*100, digits = digits), "%"))
		print(x.out['exposure']['bounds'])
		print(x.out['outcome']['bounds'] + "\n")


def make_formula(y, x):
	"""
	Given independent and dependent variables, make into a string resembling an R-style formula.

	:param y: the dependent variable in the regression
	:type y: str
	:param x: the independent variables in the regression
	:type x: list(str)
	:return: a string containing an R-style formula describing the regression of y on x
	"""
	return y + ' ~ ' + ' + '.join(x)


def sense_trait(model, instrument, trait, alpha, type_trait):
	"""
	Given a trait (either exposure or outcome), compute the sensitivity statistics (partial R2, robustness value).

	:param model: a statsmodels OLSResults object containing either the "first stage" or "reduced form" regression
	:type model: statsmodels OLSResults
	:param instrument: a string with the instrument variable name
	:type instrument: str
	:param trait: a string with the name of the (exposure or outcome) trait
	:type trait: str
	:param alpha: significance level for the robustness value RV_qa to render the
					estimate not significant
	:type alpha: float
	:param type_trait: either "exposure" or "outcome", specifying which type of trait
	:type type_trait: str
	:return: a dictionary containing the partial R2 and robustness value (rv), as well as the input variable names
	"""
	r2 = sensitivity_stats.partial_r2(model=model, covariates=instrument)
	rv = sensitivity_stats.robustness_value(model=model, covariates=instrument, q=1, alpha=alpha)
	out = {'instrument': instrument, 'trait': trait, 'type_trait': type_trait, 'partial_r2': r2,
			'rv': rv, 'alpha': alpha}
	return out


def print_sense_trait(x, digits=4):
	"""
	Print the results obtained from the sense_trait method above.

	:param x: the dictionary returned by the sense_trait method above
	:type x: dict
	:param digits: number of digits to show for printed floats
	:type digits: int, optional
	:return: None
	"""
	print("Sensitivity generic instrument (" + str(x['instrument']) + ") -> " +
			str(x['type_trait']) + " (" + str(x['trait']) + ")")
	print("  Partial R2: " + str(round(x['partial_r2']*100, digits)) + "%")
	print("  RV (alpha = " + str(x['alpha']) + "): " + str(round(float(x['rv']) * 100, digits)) + "%")


def clean_benchmarks(bench, model):
	# not yet implemented
	return bench


def multiple_bounds(model, covariate, benchmark_covariates, k, alpha=0.05):
	# not yet implemented
	pass
#
