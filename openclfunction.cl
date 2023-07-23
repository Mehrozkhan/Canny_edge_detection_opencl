#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work
#endif

int getIndexGlobal(size_t countX, int i, int j) {
	return j * countX + i;
}
// Read value from global array a, return 0 if outside image
float getValueGlobal(__global const float* a, size_t countX, size_t countY, int i, int j) {
	if (i < 0 || i >= countX || j < 0 || j >= countY)
		return 0;
	else
		return a[getIndexGlobal(countX, i, j)];
}



__kernel void gaussianKernel(__global float* d_input, __global float* d_output) {

    uint i = get_global_id(0);
    uint j = get_global_id(1);

    float weights[5][5] = {
        {1, 4, 7, 4, 1},
        {4, 16, 26, 16, 4},
        {7, 26, 41, 26, 7},
        {4, 16, 26, 16, 4},
        {1, 4, 7, 4, 1}
    };

    uint countX = get_global_size(0);
	uint countY = get_global_size(1);

    float sum = 0.0;

    for (int x = -2; x <= 2; x++) {
        for (int y = -2; y <= 2; y++) {
            float pixelValue = getValueGlobal(d_input, countX, countY, i + x, j + y);
            sum += weights[x + 2][y + 2] * pixelValue;
        }
    }

    d_output[getIndexGlobal(countX, i, j)] = sum / 273; // Normalize by the sum of the kernel elements
}

__kernel void nonMaxSuppressionKernel(__global const float* d_input2, __global float* d_output2, __global const int* d_in_segment1) {

    uint i = get_global_id(0);
    uint j = get_global_id(1);

    uint countX = get_global_size(0);
	uint countY = get_global_size(1);

    int segment = d_in_segment1[getIndexGlobal(countX, i, j)];
    float currentPixel = d_input2[getIndexGlobal(countX, i, j)];
    float leftPixel, rightPixel, upperPixel, lowerPixel;

    switch (segment) {
    case 1:
        leftPixel = d_input2[getIndexGlobal(countX, i - 1, j)];
        rightPixel = d_input2[getIndexGlobal(countX, i + 1, j)];
        if (leftPixel >= currentPixel || rightPixel > currentPixel)
            d_output2[getIndexGlobal(countX, i, j)] = 0;
        else
            d_output2[getIndexGlobal(countX, i, j)] = currentPixel;
        break;
    case 2:
        leftPixel = d_input2[getIndexGlobal(countX, i - 1, j)];
        rightPixel = d_input2[getIndexGlobal(countX, i + 1, j)];
        if (leftPixel >= currentPixel || rightPixel > currentPixel)
            d_output2[getIndexGlobal(countX, i, j)] = 0;
        else
            d_output2[getIndexGlobal(countX, i, j)] = currentPixel;
        break;
    case 3:
        upperPixel = d_input2[getIndexGlobal(countX, i, j - 1)];
        lowerPixel = d_input2[getIndexGlobal(countX, i, j + 1)];
        if (upperPixel >= currentPixel || lowerPixel > currentPixel)
            d_output2[getIndexGlobal(countX, i, j)] = 0;
        else
            d_output2[getIndexGlobal(countX, i, j)] = currentPixel;
        break;
    case 4:
        upperPixel = d_input2[getIndexGlobal(countX, i, j - 1)];
        lowerPixel = d_input2[getIndexGlobal(countX, i, j + 1)];
        if (upperPixel >= currentPixel || lowerPixel > currentPixel)
            d_output2[getIndexGlobal(countX, i, j)] = 0;
        else
            d_output2[getIndexGlobal(countX, i, j)] = currentPixel;
        break;
    default:
        d_output2[getIndexGlobal(countX, i, j)] = 0;
        break;
    }
}