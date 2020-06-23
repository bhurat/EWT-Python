This Python package allows for the use of the 1D and 2D Empirical wavelet transforms, as described in the following papers:

- J.Gilles, "Empirical wavelet transform" to appear in IEEE Trans. Signal Processing, Vol.61, No.16, 3999--4010, August 2013.
Preprint available at ftp://ftp.math.ucla.edu/pub/camreport/cam13-33.pdf
- J.Gilles, G.Tran, S.Osher "2D Empirical transforms. Wavelets, Ridgelets and Curvelets Revisited", SIAM Journal on Imaging Sciences, Vol.7, No.1, 157--186, January 2014. Preprint available at ftp://ftp.math.ucla.edu/pub/camreport/cam13-35.pdf
- J.Gilles, K. Heal, "A parameterless scale-space approach to find meaningful modes in histograms - Application to image and spectrum segmentation", International Journal of Wavelets, Multiresolution and Information Processing, Vol.12, No.6, 1450044-1--1450044-17, December 2014.
Preprint available at ftp://ftp.math.ucla.edu/pub/camreport/cam14-05.pdf

This toolbox is freely distributed and can be used without any charges for research purposes. 
For commercial purposes, please contact me before. 

For any questions, comments, bug reports, email basile.hurat@gmail.com

=====================
Currently Implemented
=====================
| 1D EWT Jupiter Notebook example
|
|ewt (package)
	|-1d EWT functions
	|-2D EWT functions
		|-Tensor ewt and iewt
		|-Curvelet-1 ewt and iewt
	|-Boundaries
		|-scale-space option with all options
	|-Utilities
		|-parameter struct
		|-Spectrum Regularization

=======================
To come in next version
=======================
|-Curvelet option 2
|-Curvelet option 3
|-Comments
===============
Needed Packages
===============
- Numpy		:Most functionality is done using numpy arrays
- Scipy		:Needed for bessel function for scale-space and erf function for ppfft
============
Organization
============
EWT
|
|-EWT1D		: 1D EWT functions
|-EWT2D		: 2D EWT functions
|-Boundaries	: Functions needed for boundary detection (1d or 2d)
|-Tests		: Test functions for testing 
|-Utilities	: Includes parameter struct and various utility functions