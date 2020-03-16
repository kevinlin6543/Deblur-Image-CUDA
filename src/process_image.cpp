#include "./lodepng/lodepng.h"
#include "./metrics.hpp"
#include <iostream>


//unsigned width, height, width_o, height_o;

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


int main(int argc, char *argv[])
{
    unsigned w_blurry, h_blurry;
    unsigned w_orig, h_orig;
    std::vector<int> image = decodePNG("./blurry.png", w_blurry, h_blurry);
    std::vector<int> ref = decodePNG("./orig.png", w_orig, h_orig);

    std::cout << "Blurry Width is " << w_blurry << std::endl;
    std::cout << "Blurry Height is " << h_blurry << std::endl;
    std::cout << "Orig Width is " << w_orig << std::endl;
    std::cout << "Orig Height is " << h_orig << std::endl;
    std::cout << "The MSE is " << _mse(image, w_blurry, h_blurry, ref) << std::endl;
    std::cout << "The pSNR is " << psnr(image, w_blurry, h_blurry, ref) << std::endl;

    return 0;
}
