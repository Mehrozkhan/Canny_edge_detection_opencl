/**********************************************************************************************************************
* File name: CannzEdgeDetection.cl
* This file consists of the kernel implementations for computing the Canny edge detection on the input image using GPU.
* 
***********************************************************************************************************************
*/

/**********************************************************************************************************************
* Header files
***********************************************************************************************************************
*/
#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work
#endif
#define NUMBER_OF_BINS 256

int getIndexGlobal(size_t countX, int i, int j) {
	return j * countX + i;
}

float getValueGlobal(__global const float* a, size_t countX, size_t countY, int i, int j) {
	if (i < 0 || i >= countX || j < 0 || j >= countY)
		return 0;
	else
		return a[getIndexGlobal(countX, i, j)];
}

/* Kernel for Histogram calculation */
__kernel void calculateHistogramKernel(__global float* d_input, __global float* d_output)
{
	uint i = get_global_id(0); // global index of current pixel in X direction
	uint j = get_global_id(1); // global index of current pixel in Y direction

	uint countX = get_global_size(0); //global size in X direction
	uint countY = get_global_size(1); //global size in Y direction

	// Calculate the histogram of the image
	int Pixel_value = d_input[getIndexGlobal(countX, i, j)];
	atomic_add(&d_output[Pixel_value],1);
	
}

/* Kernel for Histogram equalization */
__kernel void histogramEqualizationKernel(__global float* d_input, __global int* d_Cdf, __global float* d_output)
{

	uint i = get_global_id(0); // global index of current pixel in X direction
	uint j = get_global_id(1); // global index of current pixel in Y direction

	uint countX = get_global_size(0); //global size in X direction
	uint countY = get_global_size(1); //global size in Y direction
	
	// Calculate the equalized image
	uint Pixel_value = d_input[getIndexGlobal(countX, i, j)];
	float x = d_Cdf[Pixel_value] * 255 / (countX * countY);
	d_output[getIndexGlobal(countX, i, j)] = round(x)/255;
			
}

/* Kernel for applying gaussian filter */
__kernel void gaussianKernel(__global float* d_input, __global float* d_output) {

	uint i = get_global_id(0); // global index of current pixel in X direction
	uint j = get_global_id(1); // global index of current pixel in Y direction

	uint countX = get_global_size(0); //global size in X direction
	uint countY = get_global_size(1); //global size in Y direction

	float weights[5][5] = {
		{1, 4, 7, 4, 1},
		{4, 16, 26, 16, 4},
		{7, 26, 41, 26, 7},
		{4, 16, 26, 16, 4},
		{1, 4, 7, 4, 1}
	};

	float sum = 0.0;

	for (int x = -2; x <= 2; x++) {
		for (int y = -2; y <= 2; y++) {
			float pixelValue = getValueGlobal(d_input, countX, countY, i + x, j + y);
			float sum_temp = sum;
			sum = sum_temp + weights[x + 2][y + 2] * pixelValue;
		}
	}

	d_output[getIndexGlobal(countX, i, j)] = sum / 273; // Normalize by the sum of the kernel elements
}

/* Kernel for applying sobel filter */
__kernel void sobelKernel(__global float* d_input, __global float* d_output, __global int* d_out_segment) {


	uint i = get_global_id(0); // global index of current pixel in X direction
	uint j = get_global_id(1); // global index of current pixel in Y direction

	uint countX = get_global_size(0); //global size in X direction
	uint countY = get_global_size(1); //global size in Y direction

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

/* Kernel for applying Non Max Supression */
__kernel void nonMaxSuppressionKernel(__global float* d_input, __global float* d_output, __global int* d_in_segment) {

	uint i = get_global_id(0); // global index of current pixel in X direction
	uint j = get_global_id(1); // global index of current pixel in Y direction


	uint countX = get_global_size(0); //global size in X direction
	uint countY = get_global_size(1); //global size in Y direction

	switch (d_in_segment[getIndexGlobal(countX, i, j)]) {

		case 1: // Horizontal "-"
			// Check if the current pixel's magnitude is greater than its neighbors in the horizontal direction
			if (getValueGlobal(d_input, countX, countY, i - 1, j) >= d_input[getIndexGlobal(countX, i, j)] || getValueGlobal(d_input, countX, countY, i + 1, j) > d_input[getIndexGlobal(countX, i, j)])
				d_output[getIndexGlobal(countX, i, j)] = 0;
			else 
				d_output[getIndexGlobal(countX, i, j)] = d_input[getIndexGlobal(countX, i, j)];
			break;
		
		case 2: // Diagonal "/"
				// Check if the current pixel's magnitude is greater than its neighbors in the diagonal direction
				if (getValueGlobal(d_input, countX, countY, i + 1, j - 1) >= getValueGlobal(d_input, countX, countY, i, j) || getValueGlobal(d_input, countX, countY, i - 1, j + 1) > getValueGlobal(d_input, countX, countY, i, j))
					d_output[getIndexGlobal(countX, i, j)] = 0;
				else 
					d_output[getIndexGlobal(countX, i, j)] = d_input[getIndexGlobal(countX, i, j)];
				break;
		case 3: // Vertical "|"
                                // Check if the current pixel's magnitude is greater than its neighbors in the vertical direction
				if (getValueGlobal(d_input, countX, countY, i, j - 1) >= getValueGlobal(d_input, countX, countY, i, j) || getValueGlobal(d_input, countX, countY, i, j + 1) > getValueGlobal(d_input, countX, countY, i, j))
					d_output[getIndexGlobal(countX, i, j)] = 0;
				else d_output[getIndexGlobal(countX, i, j)] = d_input[getIndexGlobal(countX, i, j)];
				break;
		case 4: // Diagonal "\"
							// Check if the current pixel's magnitude is greater than its neighbors in the diagonal direction
				if (getValueGlobal(d_input, countX, countY, i - 1, j - 1) >= getValueGlobal(d_input, countX, countY, i, j) || getValueGlobal(d_input, countX, countY, i + 1, j + 1) > getValueGlobal(d_input, countX, countY, i, j))
					d_output[getIndexGlobal(countX, i, j)] = 0;
				else d_output[getIndexGlobal(countX, i, j)] = d_input[getIndexGlobal(countX, i, j)];
				break;
		default:
			d_output[getIndexGlobal(countX, i, j)] = 0;
			break;
	}
	
}

/* Kernel for Double thresholding */
__kernel void DoubleThresholdKernel(__global float* d_inputDt, __global float* d_outputDt, float low_threshold, float high_threshold)
{

	uint i = get_global_id(0); // global index of current pixel in X direction
	uint j = get_global_id(1); // global index of current pixel in Y direction

	uint countX = get_global_size(0); //global size in X direction
	uint countY = get_global_size(1); //global size in Y direction

        //Checking for strong edge pixel
   	if (getValueGlobal(d_inputDt, countX, countY, i, j) > high_threshold)
		d_outputDt[getIndexGlobal(countX, i, j)] = 1;
        // Checking for weak edge pixel
	else if (getValueGlobal(d_inputDt, countX, countY, i, j) > low_threshold)
	{
		d_outputDt[getIndexGlobal(countX, i, j)] = 0.5;

	}
        // Suppress edges with gradient less than low threshold
	else
		d_outputDt[getIndexGlobal(countX, i, j)] = 0;

}

/* Kernel for applying edge hysterisis */
__kernel void HysteresisKernel(__global float* d_inputHst, __global float* d_outputHst)
{
	

	uint i = get_global_id(0); // global index of current pixel in X direction
	uint j = get_global_id(1); // global index of current pixel in Y direction


	uint countX = get_global_size(0); //global size in X direction
	uint countY = get_global_size(1); //global size in Y direction


	// Initialize the output pixel value to the same as the input.
	d_outputHst[getIndexGlobal(countX, i, j)] = d_inputHst[getIndexGlobal(countX, i, j)];

        // Check if current pixel is weak edge pixel
	if (d_inputHst[getIndexGlobal(countX, i, j)] == 0.5) {
        // Check if the neighboring pixels are strong edge pixels
		if (d_inputHst[getIndexGlobal(countX, i, j) - 1] == 1 || d_inputHst[getIndexGlobal(countX, i, j) + 1] == 1 ||
			d_inputHst[getIndexGlobal(countX, i, j) - countX] == 1 || d_inputHst[getIndexGlobal(countX, i, j) + countX] == 1 ||
			d_inputHst[getIndexGlobal(countX, i, j) - countX - 1] == 1 || d_inputHst[getIndexGlobal(countX, i, j) - countX + 1] == 1 ||
			d_inputHst[getIndexGlobal(countX, i, j) + countX - 1] == 1 || d_inputHst[getIndexGlobal(countX, i, j) + countX + 1] == 1)
			d_outputHst[getIndexGlobal(countX, i, j)] = 1; // set current ouput pixel as an edge
		else
			d_outputHst[getIndexGlobal(countX, i, j)] = 0;   // set current ouput pixel to not an edge
	}

}



