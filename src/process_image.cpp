#include "./lodepng/lodepng.h"
#include <iostream>

std::vector<int> decodePNG(const char* filename) {
    std::vector<unsigned char> image;
    unsigned width, height;

    lodepng::decode(image, width, height, filename);

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
    std::vector<int> image = decodePNG("./test.png");
}
