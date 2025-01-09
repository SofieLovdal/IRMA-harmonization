# IRMA-harmonization
This repo showcases how to apply the IRMA method to harmonize your feature vectors.
IRMA was especially developed for harmonization of site effects in brain FDG PET features.

To use the method you in particular need sets of feature vectors that can be considered comparable, were it not for a site/scanner effect (or other types of bias you'd like to disregard).
For brain PET data, these comparable groups could for example be age and gender-matched healthy control cohorts.

We provide a minimal set of data which can be used to run our code examples. These are not the PET feature vectors from our paper, but instead simply points drawn from arbitrary multivariate Gaussian distributions.

Further information can be found in, and if you use this code, please cite Lovdal et al. (2025) "IRMA: Machine learning-based harmonization of FDG PET brain
scans in multi-center studies"
