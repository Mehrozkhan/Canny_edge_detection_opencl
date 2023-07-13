#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work
#endif

//TODO
__kernel void sobelKernel1(__global const float* d_input, __global float* d_output) {

	/*size_t i = get_global_id(0);
	size_t j = get_global_id(1);
	size_t countX = get_global_size(0);
	size_t countY = get_global_size(1);

	int getIndexGlobal(std::size_t countX, int i, int j) {
		return j * countX + i;
	}
	// Read value from global array a, return 0 if outside image
	float getValueGlobal(__global const float* a, std::size_t countX, std::size_t countY, int i, int j) {
		if (i < 0 || (size_t)i >= countX || j < 0 || (size_t)j >= countY)
			return 0;
		else
			return a[getIndexGlobal(countX, i, j)];
	}
	float Gx = getValueGlobal(d_input, countX, countY, i - 1, j - 1) + 2 * getValueGlobal(d_input, countX, countY, i - 1, j) + getValueGlobal(d_input, countX, countY, i - 1, j + 1)
		- getValueGlobal(d_input, countX, countY, i + 1, j - 1) - 2 * getValueGlobal(d_input, countX, countY, i + 1, j) - getValueGlobal(d_input, countX, countY, i + 1, j + 1);
	float Gy = getValueGlobal(d_input, countX, countY, i - 1, j - 1) + 2 * getValueGlobal(d_input, countX, countY, i, j - 1) + getValueGlobal(d_input, countX, countY, i + 1, j - 1)
		- getValueGlobal(d_input, countX, countY, i - 1, j + 1) - 2 * getValueGlobal(d_input, countX, countY, i, j + 1) - getValueGlobal(d_input, countX, countY, i + 1, j + 1);
	d_output[getIndexGlobal(countX, i, j)] = sqrt(Gx * Gx + Gy * Gy);*/

}

//TODO
