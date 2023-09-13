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
		leftPixel = d_input2[getIndexGlobal(countX, i, j) - 1];
		rightPixel = d_input2[getIndexGlobal(countX, i, j) + 1];
		if (leftPixel >= currentPixel || rightPixel > currentPixel)
			d_output2[getIndexGlobal(countX, i, j)] = 0;
		else
			d_output2[getIndexGlobal(countX, i, j)] = currentPixel;
		break;
	case 2:
		leftPixel = d_input2[getIndexGlobal(countX, i, j) - (countX - 1)]; 
		rightPixel = d_input2[getIndexGlobal(countX, i, j) + (countX - 1)];
		if (leftPixel >= currentPixel || rightPixel > currentPixel)
			d_output2[getIndexGlobal(countX, i, j)] = 0;
		else
			d_output2[getIndexGlobal(countX, i, j)] = currentPixel;
		break;
	case 3:
		upperPixel = d_input2[getIndexGlobal(countX, i, j) - (countX)];
		lowerPixel = d_input2[getIndexGlobal(countX, i, j) + (countX)];
		if (upperPixel >= currentPixel || lowerPixel > currentPixel)
			d_output2[getIndexGlobal(countX, i, j)] = 0;
		else
			d_output2[getIndexGlobal(countX, i, j)] = currentPixel;
		break;
	case 4:
		upperPixel = d_input2[getIndexGlobal(countX, i, j) - (countX + 1)];
		lowerPixel = d_input2[getIndexGlobal(countX, i, j) + (countX + 1)];
		if (upperPixel >= currentPixel || lowerPixel > currentPixel)
			d_output2[getIndexGlobal(countX, i, j)] = 0;
		else
			d_output2[getIndexGlobal(countX, i, j)] = currentPixel;
		break;
	default:
		d_output2[getIndexGlobal(countX, i, j)] = 0;
		break;
	}

__kernel void DoubleThresholdKernel(__global float* d_inputDt, __global float* d_outputDt, float low_threshold, float high_threshold)
{

	uint i = get_global_id(0);
	uint j = get_global_id(1);

	uint countX = get_global_size(0);
	uint countY = get_global_size(1);

	if (getValueGlobal(d_inputDt, countX, countY, i, j) > high_threshold)
		d_outputDt[getIndexGlobal(countX, i, j)] = 255;     //absolutely edge
	else if (getValueGlobal(d_inputDt, countX, countY, i, j) > low_threshold)
	{
		d_outputDt[getIndexGlobal(countX, i, j)] = 127;      //potential edge

	}
	else
		d_outputDt[getIndexGlobal(countX, i, j)] = 0;       //absolutely not edge

}

__kernel void HysteresisKernel(__global float* d_inputHst, __global float* d_outputHst)
{
	

	uint i = get_global_id(0);
	uint j = get_global_id(1);

	uint countX = get_global_size(0);
	uint countY = get_global_size(1);
	d_outputHst[getIndexGlobal(countX, i, j)] = d_inputHst[getIndexGlobal(countX, i, j)];

	if (d_inputHst[getIndexGlobal(countX, i, j)] == 127) {
		if (d_inputHst[getIndexGlobal(countX, i, j) - 1] == 255 || d_inputHst[getIndexGlobal(countX, i, j) + 1] == 255 ||
			d_inputHst[getIndexGlobal(countX, i, j) - countX] == 255 || d_inputHst[getIndexGlobal(countX, i, j) + countX] == 255 ||
			d_inputHst[getIndexGlobal(countX, i, j) - countX - 1] == 255 || d_inputHst[getIndexGlobal(countX, i, j) - countX + 1] == 255 ||
			d_inputHst[getIndexGlobal(countX, i, j) + countX - 1] == 255 || d_inputHst[getIndexGlobal(countX, i, j) + countX + 1] == 255)
			d_outputHst[getIndexGlobal(countX, i, j)] = 255;
		else
			d_outputHst[getIndexGlobal(countX, i, j)] = 0;
	}

}


//TODO
