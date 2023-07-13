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

__kernel void sobelKernel(__global float* d_input, __global float* d_output, __global int* d_out_segment) {


	uint i = get_global_id(0);
	uint j = get_global_id(1);

	uint countX = get_global_size(0);
	uint countY = get_global_size(1);

	float Gx = getValueGlobal(d_input, countX, countY, i - 1, j - 1) + 2 * getValueGlobal(d_input, countX, countY, i - 1, j) + getValueGlobal(d_input, countX, countY, i - 1, j + 1)
		- getValueGlobal(d_input, countX, countY, i + 1, j - 1) - 2 * getValueGlobal(d_input, countX, countY, i + 1, j) - getValueGlobal(d_input, countX, countY, i + 1, j + 1);
	float Gy = getValueGlobal(d_input, countX, countY, i - 1, j - 1) + 2 * getValueGlobal(d_input, countX, countY, i, j - 1) + getValueGlobal(d_input, countX, countY, i + 1, j - 1)
		- getValueGlobal(d_input, countX, countY, i - 1, j + 1) - 2 * getValueGlobal(d_input, countX, countY, i, j + 1) - getValueGlobal(d_input, countX, countY, i + 1, j + 1);
	
	d_output[getIndexGlobal(countX, i, j)] = sqrt(Gx * Gx + Gy * Gy);
	
	float theta = atan2(Gy, Gx);

	theta = theta * (360.0 / (2.0 * 3.14159265358979323846264338327950288));

	int segment = 0;
			
	if ((theta <= 22.5 && theta >= -22.5) || (theta <= -157.5) || (theta >= 157.5))
		segment = 1;  // "-"
	else if ((theta > 22.5 && theta <= 67.5) || (theta > -157.5 && theta <= -112.5))
		segment = 2;  // "/" 
	else if ((theta > 67.5 && theta <= 112.5) || (theta >= -112.5 && theta < -67.5))
		segment = 3;  // "|"
	else if ((theta >= -67.5 && theta < -22.5) || (theta > 112.5 && theta < 157.5))
		segment = 4;  // "\"  
	else
		segment = 0;
	
	d_out_segment[getIndexGlobal(countX, i, j)] = segment;
			

}

//TODO
