/* Header for common metrics for image comparison */
#include <iostream>
#include <vector>

/* 
	MSE - Mean Square Error
		From MATLAB docs:
		MSE measures the average squared difference between actual and ideal pixel values. 
		This metric is simple to calculate but might not align well with the human perception of quality.
*/
double _mse(std::vector<int> im, unsigned int im_w, unsigned int im_h, std::vector<int> ref);


/* 
	pSNR - Peak Signal-to-Noise Ratio 
		From MATLAB docs:
			pSNR is derived from the mean square error, and indicates the ratio of the maximum pixel intensity to the power of the distortion. 
			Like MSE, the pSNR metric is simple to calculate but might not align well with perceived quality.

			pSNR = 10*log10(R^2 / MSE), where R is the maximum fluctuation in the input image data type
*/
double psnr(std::vector<int> im, unsigned int im_w, unsigned int im_h, std::vector<int> ref);
