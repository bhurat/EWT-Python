
# The Empirical Wavelet Transform for Python
### By Basile Hurat

## Introduction
This is documentation for the empirical wavelet transform package in Python. Empirical wavelets are a generalization of wavelets. A family of empirical wavelets can be formed from the translation, scaling, and modulation of a mother wavelet. This allows them to be more adaptive than traditional wavelets, and algorithms have been used to construct a family of wavelets that are based on the information of the signal they decompose. The following papers go into more detail on empirical wavelets:
1. ewt1d
2. ewt2d
3. scalespace
4. ewwt
5. continuous

This package is originally based on the MATLAB package by Jerome Gilles (https://www.mathworks.com/matlabcentral/fileexchange/42141-empirical-wavelet-transforms), and includes python implementations of some functions (ppfft,ippfft,appfft,fracfft) based on the MATLAB code by Michael Elad (https://elad.cs.technion.ac.il/software/) . 

The primary functions in this package are: 
- ewt1d.py
	- **ewt1d**: 1D empirical wavelet transform
	- **iewt1d**: 1D inverse empirical wavelet transform
- ewt2d.py
	- **ewt2dTensor**: 2D empirical tensor transform
	- **iewt2dTensor**:  2D inverse empirical tensor transform
	- **ewt2dLP**: 2D empirical Littlewood-Paley transform
	- **iewt2dLP**: 2D inverse empirical Littlewood-Paley transform
	- **ewt2dRidgelet**: the 2D empirical ridgelet transform
	- **iewt2dRidgelet**: 2D inverse empirical ridgelet transform
	- **ewt2dCurvelet**: 2D empirical curvelet transform
	- **iewt2dCurvelet**: 2D inverse empirical curvelet transform
- utilities.py
	- **ewt_params**: Class which contains all empirical wavelet transform parameters
	- **showewt1dBoundaries**:  Shows boundaries superimposed on spectrum for 1D
	- **show2DTensorBoundaries**: Shows boundaries superimposed on spectrum
	- **show2DLPBoundaries**: Shows boundaries superimposed on spectrum
	- **show2DCurveletBoundaries**: Shows boundaries superimposed on spectrum
	- **showEWT1DCoefficients**: Shows resulting coefficients of ewt for 1D
	- **ShowEWT2DCoefficients**: Shows resulting coefficients of ewt for 1D


## EWT Parameter Guide
There are many parameters for the empirical wavelet transform functions, all of which are held within the ewt_params() class. This section goes over what each parameter does and the options within that parameter, if appropriate.
First, to set parameters to default, you call the class with no input. Then, you may change parameters as you please 
Ex:

	import ewt
	params = ewt.utilities.ewt_params()
	params.log = 1

Here are the parameters based on their use

#### Signal Pre-processing Parameters:
 * **log**: Boolean. If 1, performs log of magnitude spectrum of signal before detecting boundaries
 * **removeTrends**: Preprocesses by removing global trends within signal. Options include:
	 * 'none': no changes 
	 * 'plaw': power law fit
	 *  'poly': polynomial fit
		 *	**degree**: polynomial degree value if you choose 'poly' in removeTrends
	 * 'morpho': Average of opening and closing operators
	 *  'tophat': Morphological tophat operator 
	 *  'opening': opening morphological operator
* **spectrumRegularize**: Preprocesses by regularizing the original signal. Options include
	* 'gaussian': Gaussian smoothing
		* **lengthFilter**: Gaussian filter size
		* **sigmaFilter**: Gaussian standard deviation 
	*    'average':  box filter smoothing
			* **lengthFilter**: box filter size
	* 'closing': morphological closing operator
		* **lengthFilter**: operator filter size

#### Boundary Parameters
* **detect**: Method for detecting boundaries. Options include:
	* 'scalespace': Use scalespace representations to determine boundaries,.
		* **typeDetect**: Method for scalespace to use to determine meaningful extrema
	* 'locmax': Returns midpoints between N highest maxima.
	* 'locmaxmin': Returns lowest minima between N highest maxima.
	* 'locmaxminf': Same as above except function finds minima from original signal, not pre-processed signal.
	* 'adaptive': Returns lowest minima between a set of points init_bounds
		* **init_bounds**: array of initial bounds to look for lowest minima between
	* 'adaptivereg': Same as above except function finds minima from original signal, not pre-processed signal.


- **N**: Determines number of boundaries for all boundary detection methods other than 'scalespace'

#### Empirical Wavelet Parameters
* **option**: (1,2,3) Option desired when using empirical curvelet transform


## Using EWT for 1D Signals

Begin by loading ewt package and other necessary packages 

    import numpy as np
    import scipy as sp
    import matplotlib.pyplot as plt
Load a signal and ewt parameter class. Then alter defaults as you please

    x = np.arange(0,1,.001)
    f = 6*x + np.cos(8*np.pi*x) + 0.5*np.cos(40*np.pi*x)
    params = ewt.utilities.ewt_params()
    params.log = 1
	params.removeTrends = 'opening'
Then, perform the empirical wavelet transform. If you are working in 1d, then it simply is 

    [ewtc,mfb,bounds] = ewt.ewt1d.ewt1d(f,params)

The ewtc variable holds your empirical wavelet coefficients, the mfb variable holds your empirical wavelets, and the bounds variable gives you the detected bounds that defined your filterbank.  

To perform an inversed empirical wavelet transform, you use

	reconstruction = ewt.ewt1d.iewt1d(ewtc,mfb)

## Using EWT for 2D Signals
In 2D, you have multiple options,  which differ in numbers of outputs. Your options include:
|function name|Output #  | Description | support shape |
|--|--|--|--|
| ewt2dTensor | 5 | Empirical Tensor Transform | Rectangular in Fourier Domain|
| ewt2dLP | 3 | Empirical Littlewood-Paley Transform | Ring in Fourier Domain|
| ewt2dRidgelet | 3 | Empirical Ridgelet Transform | Rectangular in Pseudo-Polar Fourier Domain|
| ewt2dCurvelet* | 4 | Empirical Curvelet Transform | Polar wedges in Fourier Domain|
*The empirical Curvelet transform has multiple options for detecting the boundaries that defined the polar wedges. They are described below
* Option 1: Detects scales and angles independently
* Option 2: Detects scales first, and then detects angles for each scale
* Option 3: Detects angles first, and then detects scales for each angle 

With this, we walk you through an example with 2D EWT.

Begin by loading ewt package and other necessary packages 

    import numpy as np
    import scipy as sp
    import matplotlib.pyplot as plt
Load an image and ewt parameter class. Then alter defaults as you please

    x = np.arange(0,1,.001)
    f = 6*x + np.cos(8*np.pi*x) + 0.5*np.cos(40*np.pi*x)
    params = ewt.utilities.ewt_params()
    params.log = 1
	params.removeTrends = 'opening'
	params.option = 2
Then, to perform the 2D empirical wavelet transform, it is straightforward. Here, we perform the empirical curvelet transform but other 2D transforms follow a similar structure. 

    [ewtc,mfb,scale_bound,angle_bound] = ewt.ewt2d.ewt2dCurvelet(f,params)
Once again, ewtc holds the empirical curvelet coefficients and mfb holds the empirical curvelet filters. The detected scale bounds and angular bounds are held in separate variables. 

To perform the inverse 2D empirical wavelet transform, you use the relevant inverse function:

	reconstruction = iewt2dCurvelet(ewtc,mfb)

## Displaying And Viewing Results
Once you have performed the empirical wavelet transform on your signal, there are two things that useful to visualize: the empirical wavelet coefficients and the detected boundaries that define the empirical wavelet supports. There is a family of functions for each type of visualization. 

We begin with the visualization of empirical wavelet coefficients. There is a function for each dimension, with the following parameters:

	showEWT1DCoefficients(ewtc)
	showEWT2DCoefficients(ewtc,ewt_type,option = 1)

where ewtc is the empirical wavelet coefficients, ewt_type is a string dictating which type you want ('tensor', 'lp', 'ridgelet', 'curvelet'), and option dictates the curvelet option you used. This creates a figure which shows each empirical wavelet coefficient is shown in a different subfigure. 


As for the the visualization of boundaries, the following functions exist:

	showewt1dBoundaries(f,bounds):
	show2DTensorBoundaries(f,bounds_row,bounds_col)
	show2DLPBoundaries(f,bounds_scales)
	show2DCurveletBoundaries(f,option,bounds_scales,bounds_angles)

where f is the original signal, 'bounds' variables contain the bounds, and option dictates the option you used for the empirical curvelet transform. *Note: For ridgelets, you use the function for LP Empirical Wavelets!* These functions show a figure with the magnitude spectrum of the image and superimposed on it lines which show the detected boundary. 
## Glossary of Functions
This section names each function contained within this package. We present them based on the files they are found in. Documentation for the individual functions can be found in block comments at the top of each function.

### ewt1d
ewt1d
iewt1d
ewt_LP_Filterbank
ewt_LP_Scaling
ewt_LP_Wavelet
ewt_LP_Scaling_Complex
ewt_LP_Wavelet_ComplexLow
ewt_LP_Wavelet_CompelxHigh
ewt_beta

### ewt2d

### Boundaries
ewt_boundariesDetect
ewt_localMaxBounds
ewt_localMaxMinBounds
ewt_adaptiveBounds
ewt_GSSDetect
GSS
lengthScaleCurve
localmin
removePlateaus
otsu
empiricalLaw
halfNormal
ewtkmeans
### Utilities
ewt_params (Class)
spectrumRegularize
removeTrends
ewt_opening
ewt_closing
ewt_erosion
ewt_dilation
showewt1dBoundaries
show2DTensorBoundaries
show2DLPBoundaries
show2DCurveletBoundaries
showEWT1DCoefficients
showEWT2DCoefficients