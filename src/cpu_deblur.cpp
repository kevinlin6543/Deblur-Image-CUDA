#include "./lodepng/lodepng.h"
#include "./metrics.hpp"
#include <iostream>
#include <math.h>
#include <algorithm>

const double pi = 3.14159265358979323846;
//unsigned width, height, width_o, height_o;

std::vector<std::vector<double> > calculatePSF() {
	int psf_size = 5;
	double mean_row = 0.0;
	double mean_col = psf_size/2.0;

	double sigma_row = 4.0;
	double sigma_col = 3.0;

	double sum = 0.0;
	double temp;

	std::vector<std::vector<double> > psf(psf_size, std::vector<double> (psf_size));

	for (int j = 0; j< psf.size(); j++) {
		for (int k = 0; k< psf[0].size(); k++) {
			temp = exp(-0.5 * (pow((j - mean_row) / sigma_row, 2.0) + pow((k - mean_col) / sigma_col, 2.0))) / (2* pi * sigma_row * sigma_col);
			sum += temp;
			psf[j][k] = temp;
		}
	}

	for (int row = 0; row<psf.size(); row++) {
		for (int col = 0; col<psf[0].size(); col++) {
			psf[row][col] /= sum;
		}
	}
	return psf;
}

std::vector<int> decodePNG(const char* filename, unsigned &w, unsigned &h) {
    std::vector<unsigned char> image;
    //unsigned width, height;

    lodepng::decode(image, w, h, filename);

    std::vector<int> image_without_alpha;
    for(unsigned int i = 0; i < image.size(); i++) {
        if (i % 4 != 3) {
            image_without_alpha.push_back((int)image[i]);
        }
    }

    return image_without_alpha;
}


std::vector<std::vector<int> > convert2D(std::vector<int> &vec1D, unsigned w, unsigned h) {
    //std::cerr << "vec1d size = " << vec1D.size() << '\n';
    std::vector<std::vector<int> > vec2D;
    vec2D.resize(h);
    for (int i = 0; i < h; i++){
        vec2D[i].resize(w);
    }
    for (int i = 0; i < vec1D.size(); i++) {
        // std::cerr << "i = " << i << '\n';
        int row = i / w;
        int col = i % w;
        /*
        std::cerr << "row = " << row << '\n';
        std::cerr << "col = " << col << '\n';
        std::cerr << "vec2D size = [" << vec2D.size() << ", " << vec2D[0].size() << "]" << '\n';
        */
        vec2D[row][col] = vec1D[i];
    }
    return vec2D;
}


void elementWiseMul(std::vector<std::vector<std::vector<double> > > &a,
	std::vector<std::vector<std::vector<double> > > &b,
	std::vector<std::vector<std::vector<double> > > &c) {

	if (a.size() != b.size() || a[0].size() != b[0].size() || a[0][0].size() != b[0][0].size()) {
		std::cerr << "Error in Element Wise Multiplication. Dimensions do not agree." << std::endl;
		std::cerr << "Dimension of A: [" << a.size() << ", " << a[0].size() << ", " << a[0][0].size() << "]" << std::endl;
		std::cerr << "Dimension of B: [" << b.size() << ", " << b[0].size() << ", " << b[0][0].size() << "]" << std::endl;
		exit(-1);
	}
	for(int i = 0; i < a.size(); i++) {
		for(int j = 0; j < a[0].size(); j++) {
			std::transform(a[i][j].begin(), a[i][j].end(), b[i][j].begin(), c[i][j].begin(), std::multiplies<double>());
		}
	}
}

void convolve(std::vector<std::vector<std::vector<double> > > &src,
	std::vector<std::vector<std::vector<double> > > &dest,
	std::vector<std::vector<std::vector<double> > > &kernel) {

	int kernel_centerx = kernel[0].size() / 2;
	int kernel_centery = kernel.size() / 2;
	int rows = src.size();
	int cols = src[0].size();
	for(int i = 0; i < rows; ++i) {
	    for(int j = 0; j < cols; ++j) {
	        for(int m = 0; m < kernel.size(); ++m) {
	            for(int n = 0; n < kernel[0].size(); ++n) {
	                int ii = i + (m - kernel_centery);
	                int jj = j + (n - kernel_centerx);
	                if(ii >= 0 && ii < rows && jj >= 0 && jj < cols) {
						std::transform(src[ii][jj].begin(), src[ii][jj].end(), kernel[m][n].begin(), dest[i][j].begin(), std::multiplies<double>());
	                  	//dest[i][j] += src[ii][jj] * kernel[m][n];
				  	}
	            }
	        }
	    }
	}

}

void print3D(std::vector<std::vector<std::vector<double> > > a) {
	std::cout << "[";
	for(int i = 0; i < a.size(); i++) {
		std::cout << "[";
		for(int j = 0; j < a[0].size(); j++) {
			std::cout << "[";
			for(int k = 0; k < a[0][0].size(); k++) {
				std::cout << a[i][j][k] << ' ';
			}
			std::cout << "] ";
		}
		std::cout << "]" << std::endl;
	}
	std::cout << "]" << std::endl;
}

int main(int argc, char *argv[])
{
    unsigned w_blurry, h_blurry;
    unsigned w_orig, h_orig;
    std::vector<int> image = decodePNG("./blurry.png", w_blurry, h_blurry);
    std::vector<int> ref = decodePNG("./orig.png", w_orig, h_orig);

    std::vector<std::vector<std::vector<int> > > final_RGB_img;
    final_RGB_img.resize(h_blurry);
    std::vector<std::vector<int> > initial = convert2D(image, w_blurry * 3, h_blurry);

    for (int i = 0; i < initial.size(); i++) {
        final_RGB_img[i] = convert2D(initial[i], 3, w_blurry);
    }

	std::vector<std::vector<double> > psf = calculatePSF();

    std::cout << "Height = " << final_RGB_img.size() << std::endl;
    std::cout << "Width = " << final_RGB_img[0].size() << std::endl;
    std::cout << "Depth = " << final_RGB_img[0][0].size() << std::endl;
	/*
	std::vector<std::vector<std::vector<double> > > vect1;
	std::vector<std::vector<std::vector<double> > > vect2;

	std::vector<std::vector<double> > temp1;
	std::vector<double> temp2 {1, 2, 3};
	temp1.push_back(temp2);
	std::vector<double> temp3 {3, 4, 5};
	temp1.push_back(temp3);
	std::vector<double> temp4 {5, 6, 7};
	temp1.push_back(temp4);
	vect1.push_back(temp1);

	std::vector<std::vector<double> > temp8;
	std::vector<double> temp5 {9, 8, 7};
	temp8.push_back(temp5);
	std::vector<double> temp6 {6, 5, 4};
	temp8.push_back(temp6);
	std::vector<double> temp7 {3, 2, 1};
	temp8.push_back(temp7);
	vect1.push_back(temp8);

	std::vector<std::vector<double> > temp9;
	std::vector<double> temp10 {100, 3.5, 33.3};
	temp9.push_back(temp10);
	std::vector<double> temp11 {4, -2.1, 21.9};
	temp9.push_back(temp11);
	std::vector<double> temp12 {16, 2.3, 11.7};
	temp9.push_back(temp12);
	vect2.push_back(temp9);

	std::vector<std::vector<double> > temp13;
	std::vector<double> temp14 {2, 45, 5};
	temp13.push_back(temp14);
	std::vector<double> temp15 {6, 11, 8};
	temp13.push_back(temp15);
	std::vector<double> temp16 {21, 30, 17};
	temp13.push_back(temp16);
	vect2.push_back(temp13);

	print3D(vect2);

	std::vector<std::vector<std::vector<double> > > result(2, std::vector<std::vector<double> > (3, std::vector<double> (3)));
	elementWiseMul(vect1, vect2, result);
	print3D(result);
	*/
    std::cout << "Blurry Width is " << w_blurry << std::endl;
    std::cout << "Blurry Height is " << h_blurry << std::endl;
    std::cout << "Orig Width is " << w_orig << std::endl;
    std::cout << "Orig Height is " << h_orig << std::endl;
    std::cout << "The MSE is " << _mse(image, w_blurry, h_blurry, ref) << std::endl;
    std::cout << "The pSNR is " << psnr(image, w_blurry, h_blurry, ref) << std::endl;

    return 0;
}
