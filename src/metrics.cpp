/* Implementation of metrics */

#include "metrics.hpp"
#include <iostream>
#include <cmath>
#include <limits>

/* MSE - Mean Square Error */
double _mse(std::vector<int> im, unsigned int im_w, unsigned int im_h, std::vector<int> ref)
{
	if(im.size() != ref.size())
	{
		std::cerr << "Image sizes are not the same!  Cannot calculate MSE" << std::endl;
		exit(-1);
	}

	double sq_err = 0;

	for(unsigned long i = 0; i < im.size(); i+=3)
	{
		/* Get the RGB values for the image and reference */
		double im_rgb_r = im[i]; 
		double im_rgb_g = im[i+1]; 
		double im_rgb_b = im[i+2];

		double ref_rgb_r = ref[i]; 
		double ref_rgb_g = ref[i+1]; 
		double ref_rgb_b = ref[i+2];

		/* Find square error for each RGB value and then append to total square error */
		double r_sq_err = pow( (im_rgb_r - ref_rgb_r) , 2);
		double g_sq_err = pow( (im_rgb_g - ref_rgb_g) , 2);
		double b_sq_err = pow( (im_rgb_b - ref_rgb_b) , 2);

		sq_err += r_sq_err + g_sq_err + b_sq_err;
	}

	double mean_sq_err = sq_err/(im_w * im_h);

	return mean_sq_err;
}


/* pSNR - Peak Signal-to-Noise Ratio */
double psnr(std::vector<int> im, unsigned int im_w, unsigned int im_h, std::vector<int> ref)
{
	/* Find MSE first */
	double mean_sq_err = _mse(im, im_w, im_h, ref);

	/* Plug into pSNR formula */
	double R = 255;
	double peak_snr = 10*log10( pow(R,2) / mean_sq_err);

	return peak_snr;
}



