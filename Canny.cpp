/**********************************************************************************************************************
* File name: Canny.cpp
* This file consists of function definitions of the CPU implementation.
* 
***********************************************************************************************************************
*/

/**********************************************************************************************************************
* Header files
***********************************************************************************************************************
*/
#include <stdio.h>

#include <Core/Assert.hpp>
#include <Core/Time.hpp>
#include <Core/Image.hpp>
#include <OpenCL/cl-patched.hpp>
#include <OpenCL/Program.hpp>
#include <OpenCL/Event.hpp>
#include <OpenCL/Device.hpp>

#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include<string.h>

#include <boost/lexical_cast.hpp>


#define M_PI acos(-1.0)
#define NUMBER_OF_BINS 256

/**********************************************************************************************************************
* Global variables
***********************************************************************************************************************
*/
/* Threshold values for Double thresholding operation */
float high_threshold = (65 * 1.33)/255;//0.25*255;
float low_threshold = (65 * 0.66)/255;//0.05*255;

/* Variables for time benchmarking */
Core::TimeSpan cpubeginGaussian = Core::TimeSpan::fromSeconds(0);
Core::TimeSpan cpuendGaussian = Core::TimeSpan::fromSeconds(0);
Core::TimeSpan cpubeginsobel = Core::TimeSpan::fromSeconds(0);
Core::TimeSpan cpuendSobel = Core::TimeSpan::fromSeconds(0);
Core::TimeSpan cpuendNonmaxsuppression = Core::TimeSpan::fromSeconds(0);
Core::TimeSpan cpuendDoublethreshold = Core::TimeSpan::fromSeconds(0);
Core::TimeSpan cpuendHysteresis = Core::TimeSpan::fromSeconds(0);

/**********************************************************************************************************************
* Function definitions (CPU Implementation)
***********************************************************************************************************************
*/

/**********************************************************************************************************************
* Function name: getIndexGlobal
* Calculate the global index from 2D coordinates (i, j) in a 2D image array
* Parameters:
*  countX - Width of the image in pixel
*  i - X-coordinate
*  j - Y-coordinate
* Return:
*  The global index corresponding to the (i, j) coordinates
***********************************************************************************************************************
*/
int getIndexGlobal(std::size_t countX, int i, int j) {
	return j * countX + i;
}

/*********************************************************************************************************************** 
* Function name: getValueGlobal
* Read a value from a global array 'a' with bounds checking, return 0 if outside image boundaries.
* Parameters:
*  a - Input array
*  countX - Width of the image in pixels
*  countY - Height of the image in pixels
*  i - X-coordinate
*  j - Y-coordinate
* Return:
*   The value at the specified (i, j) coordinates if within bounds, otherwise returns 0.
***********************************************************************************************************************
*/
float getValueGlobal(const std::vector<float>& a, std::size_t countX, std::size_t countY, int i, int j) {
	if (i < 0 || (size_t)i >= countX || j < 0 || (size_t)j >= countY)
		return 0;
	else
		return a[getIndexGlobal(countX, i, j)];
}

/******************************************************************************************************************************
* Function name: calculateHistogram
*  This function calculates the histogram representation of the image. The intensity value for each pixel ranges from 0 - 255.
*  A histogram has 256 bins and it represents the number of pixels in the input image having the corresponding intensity value.
* Parameters:
*  h_outputCpu - Output buffer for the calculated histogram of the image
*  h_input - Input buffer for the Input image
*  countX - Width of the image in pixels
*  countY - Height of the image in pixels
* Return:
*  void
********************************************************************************************************************************
*/
void calculateHistogram(std::vector<int>& histogram, std::vector<float>& h_input, std::size_t countX, std::size_t countY)
{
	for (int i = 0; i < countX; i++)
	{
		for (int j = 0; j < countY; j++)
		{
			int Pixel_value = h_input[getIndexGlobal(countX, i, j)];
			histogram[Pixel_value]++;
		}
	}
}

/******************************************************************************************************************************
* Function name: histogramEqualization
* Distribute the intensities throughout the image.
* Parameters:
*  h_outputCpu - Output buffer for the histogram equalized output image
*  h_input - Input buffer for the Input image
*  countX - Width of the image in pixels
*  countY - Height of the image in pixels
* Return:
*  void
********************************************************************************************************************************
*/
void histogramEqualization(std::vector<float>& h_outputCpu, std::vector<float>& h_input, std::size_t countX, std::size_t countY)
{

	std::vector<int> histogram(NUMBER_OF_BINS, 0);

	// Calculate the histogram of the image
	calculateHistogram(histogram, h_input, countX, countY);

	// Calculate the cumulative distribution function (CDF)
	std::vector<int> cdf(NUMBER_OF_BINS, 0);
	cdf[0] = histogram[0];
	for (int i = 1; i < NUMBER_OF_BINS; i++)
	{
		cdf[i] = cdf[i - 1] + histogram[i];
	}

	// Calculate the equalized image
	for (int i = 0; i < countX; i++)
	{
		for (int j = 0; j < countY; j++)
		{
			h_outputCpu[getIndexGlobal(countX, i, j)] = std::round(cdf[h_input[getIndexGlobal(countX, i, j)]] * 255 / (countX * countY))/255;
		}
	}
}

/******************************************************************************************************************************
* Function name: gaussianFilter
*  Applying gaussian blur to remove noise
* Parameters:
*   h_outputCpu - Output buffer for the blurred image
*   h_input - Input buffer containing the histogram equalized image
*   countX - Width of the image in pixels
*   countY - Height of the image in pixels
* Return:
*  void
********************************************************************************************************************************
*/
void gaussianFilter(std::vector<float>& h_outputCpu, const std::vector<float>& h_input, std::size_t countX, std::size_t countY)
{
    // Gaussian kernel weights
	float weights[5][5] = {
		{1,  4,  7,  4, 1},
		{4, 16, 26, 16, 4},
		{7, 26, 41, 26, 7},
		{4, 16, 26, 16, 4},
		{1,  4,  7,  4, 1} };
    // Loop through the input image pixels, excluding a 2-pixel border
	for (int y = 2; y < countY - 2; y++)
	{
		for (int x = 2; x < countX - 2; x++)
		{
            // Convolving the input image with the Gaussian kernel
			float sum = 0.0;
			for (int j = -2; j <= 2; j++)
			{
				for (int i = -2; i <= 2; i++)
				{
					sum += weights[j + 2][i + 2] * h_input[(y + j) * countX + (x + i)];
				}
			}
            // Normalize by the sum of the kernel elements
			h_outputCpu[y * countX + x] = sum / 273; 
		}
	}
}

/******************************************************************************************************************************
* Function name: sobelEdgeDetector
* Applying the Sobel edge detection filter to the input image and determining edge segment orientations.
* Parameters:
*   h_outputCpu - Output buffer for the edge-detected image
*   h_input - Input buffer containing the gaussian blurred image
*   h_out_segment - Output buffer for the detected edge segment orientations
*   countX - Width of the image in pixels
*   countY - Height of the image in pixels
* Return:
*   void
********************************************************************************************************************************
*/
void sobelEdgeDetector(std::vector<float>& h_outputCpu, const std::vector<float>& h_input, std::vector<int>& h_out_segment, std::size_t countX, std::size_t countY)
{
    // Loop through each pixel in the image
	for (int i = 0; i < (int)countX; i++) {
		for (int j = 0; j < (int)countY; j++) {

            // Calculate gradient components using the Sobel operator
			float Gx = getValueGlobal(h_input, countX, countY, i - 1, j - 1) + 2 * getValueGlobal(h_input, countX, countY, i - 1, j) + getValueGlobal(h_input, countX, countY, i - 1, j + 1)
				- getValueGlobal(h_input, countX, countY, i + 1, j - 1) - 2 * getValueGlobal(h_input, countX, countY, i + 1, j) - getValueGlobal(h_input, countX, countY, i + 1, j + 1);
			float Gy = getValueGlobal(h_input, countX, countY, i - 1, j - 1) + 2 * getValueGlobal(h_input, countX, countY, i, j - 1) + getValueGlobal(h_input, countX, countY, i + 1, j - 1)
				- getValueGlobal(h_input, countX, countY, i - 1, j + 1) - 2 * getValueGlobal(h_input, countX, countY, i, j + 1) - getValueGlobal(h_input, countX, countY, i + 1, j + 1);
                        
            // Calculate the gradient magnitude and store it in the output buffer
			h_outputCpu[getIndexGlobal(countX, i, j)] = sqrt(Gx * Gx + Gy * Gy);
                        
            // Calculate the orientation angle of the gradient and determine the edge segment
			double theta = std::atan2(Gy, Gx);
			theta = theta * (360.0 / (2.0 * M_PI));
			int segment = 0;
			if (Gx != 0.0 || Gy != 0.0)
			{
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
 
                // Store the detected segment in the output buffer
				h_out_segment[getIndexGlobal(countX, i, j)] = segment;
			}
		}
	}
}

/******************************************************************************************************************************
* Function name: nonMaxSuppression
* Apply non-maximum suppression to the input gradient magnitude image based on gradient orientations.
* Parameters:
*   h_outputCpu - Output buffer for the non-maximum suppressed image
*   h_input - Input buffer containing the gradient magnitude image
*   h_in_segment - Input buffer containing the detected edge segment orientations
*   countX - Width of the image in pixels
*   countY - Height of the image in pixels
* Return:
*   void
********************************************************************************************************************************
*/
void nonMaxSuppression(std::vector<float>& h_outputCpu, const std::vector<float>& h_input, std::vector<int>& h_in_segment, std::size_t countX, std::size_t countY)
{
    // Loop through each pixel in the image
	for (int i = 0; i < (int)countX; i++) {
		for (int j = 0; j < (int)countY; j++)
		{
            // Determine the edge segment orientation for the current pixel
			switch (h_in_segment[getIndexGlobal(countX, i, j)]) {
			case 1: // Horizontal "-"
				// Check if the current pixel's magnitude is greater than its neighbors in the horizontal direction
				if (h_input[getIndexGlobal(countX, i, j) - 1] >= h_input[getIndexGlobal(countX, i, j)] || h_input[getIndexGlobal(countX, i, j) + 1] > h_input[getIndexGlobal(countX, i, j)])
					h_outputCpu[getIndexGlobal(countX, i, j)] = 0;
				else h_outputCpu[getIndexGlobal(countX, i, j)] = h_input[getIndexGlobal(countX, i, j)];
				break;
			case 2: // Diagonal "/"
				// Check if the current pixel's magnitude is greater than its neighbors in the diagonal direction
				if (h_input[getIndexGlobal(countX, i, j) - (countX - 1)] >= h_input[getIndexGlobal(countX, i, j)] || h_input[getIndexGlobal(countX, i, j) + (countX - 1)] > h_input[getIndexGlobal(countX, i, j)])
					h_outputCpu[getIndexGlobal(countX, i, j)] = 0;
				else h_outputCpu[getIndexGlobal(countX, i, j)] = h_input[getIndexGlobal(countX, i, j)];
				break;
			case 3: // Vertical "|"
                                // Check if the current pixel's magnitude is greater than its neighbors in the vertical direction
				if (h_input[getIndexGlobal(countX, i, j) - (countX)] >= h_input[getIndexGlobal(countX, i, j)] || h_input[getIndexGlobal(countX, i, j) + (countX)] > h_input[getIndexGlobal(countX, i, j)])
					h_outputCpu[getIndexGlobal(countX, i, j)] = 0;
				else h_outputCpu[getIndexGlobal(countX, i, j)] = h_input[getIndexGlobal(countX, i, j)];
				break;
			case 4: // Diagonal "\"
                                // Check if the current pixel's magnitude is greater than its neighbors in the diagonal direction
				if (h_input[getIndexGlobal(countX, i, j) - (countX + 1)] >= h_input[getIndexGlobal(countX, i, j)] || h_input[getIndexGlobal(countX, i, j) + (countX + 1)] > h_input[getIndexGlobal(countX, i, j)])
					h_outputCpu[getIndexGlobal(countX, i, j)] = 0;
				else h_outputCpu[getIndexGlobal(countX, i, j)] = h_input[getIndexGlobal(countX, i, j)];
				break;
			default: // Undefined segment
				// Suppress the pixel
				h_outputCpu[getIndexGlobal(countX, i, j)] = 0;
				break;
			}
		}
	}
}

/******************************************************************************************************************************
* Function name: applyDoubleThreshold
*  This function sets all pixels with intensity, 
*  - above higher threshold to 1 (absolutely edge)
*  - above lower threshold but belwo higher threshold to 0.5 (potential edge)
*  - below lower threshold to 0. (absolutely not edge)
* Parameters:
*  h_outputCpu -Output buffer for the double threshold output image
*  h_input - Input buffer containing Non Max Supression output
*  countX - Width of the image in pixels
*  countY - Height of the image in pixels
* Return:
*  void
********************************************************************************************************************************
*/
void applyDoubleThreshold(std::vector<float>& h_outputCpu, const std::vector<float>& h_input, std::size_t countX, std::size_t countY) 
{
	
	for (int i = 0; i < countX; i++) {
		for (int j = 0; j < countY; j++) {

			if (h_input[getIndexGlobal(countX, i, j)] > high_threshold)
				h_outputCpu[getIndexGlobal(countX, i, j)] = 1;      //absolutely edge
			else if (h_input[getIndexGlobal(countX, i, j)] > low_threshold)
			{
				h_outputCpu[getIndexGlobal(countX, i, j)] = 0.5;     //potential edge

			}
			else
				h_outputCpu[getIndexGlobal(countX, i, j)] = 0;       //absolutely not edge
		}
	}
}

/******************************************************************************************************************************
* Function name: applyEdgeHysteresis
*  Hysterisis edge tracking retains the potential edges(intensity 0.5) if they are connected to a strong edge (Intensity 1) else
*  they are discarded. This is the final step of the canny edge detection algorithm. 
* Parameters:
*  h_outputCpu - Output buffer for the Hysterisis output image
*  h_input - Input buffer containing double thresholded image
*  countX - Width of the image in pixels
*  countY - Height of the image in pixels
* Return:
*  void
********************************************************************************************************************************
*/
void applyEdgeHysteresis(std::vector<float>& h_outputCpu, const std::vector<float>& h_input, std::size_t countX, std::size_t countY) 
{

	memcpy(h_outputCpu.data(), h_input.data(), countX * countY * sizeof(float));
	//Retain potential edges that are linked to strong edges
	for (int i = 1; i < countX - 1; i++) {
		for (int j = 1; j < countY - 1; j++) {
			if (h_input[getIndexGlobal(countX, i, j)] == 0.5) {
				if (h_input[getIndexGlobal(countX, i, j) - 1] == 1 || h_input[getIndexGlobal(countX, i, j) + 1] == 1 ||
					h_input[getIndexGlobal(countX, i, j) - countX] == 1 || h_input[getIndexGlobal(countX, i, j) + countX] == 1 ||
					h_input[getIndexGlobal(countX, i, j) - countX - 1] == 1 || h_input[getIndexGlobal(countX, i, j) - countX + 1] == 1 ||
					h_input[getIndexGlobal(countX, i, j) + countX - 1] == 1 || h_input[getIndexGlobal(countX, i, j) + countX + 1] == 1)
					h_outputCpu[getIndexGlobal(countX, i, j)] = 1;

				else
					h_outputCpu[getIndexGlobal(countX, i, j)] = 0;
			}
		}
	}

}

/******************************************************************************************************************************
* Function name: applyCanny_CPU
* This function takes in the histogram equalized image and performs canny edge detection.
* Parameters:
*  h_outputCpu - Output buffer for the Canny edge detection output image
*  h_input - Input buffer containing the histogram equalized image
*  countX - Width of the image in pixels
*  countY - Height of the image in pixels
* Return:
*  void
********************************************************************************************************************************
*/
void applyCanny_CPU(std::vector<float>& h_outputCpu, const std::vector<float>& h_input, std::size_t countX, std::size_t countY,
	std::size_t count, std::size_t size)
{
	// Allocate space for output data from CPU and GPU on the host
	std::vector<float> h_outputCpu_Gaussian(count);
	std::vector<float> h_outputCpu_Sobel(count);
	std::vector<int>   h_out_segment(count);
	std::vector<float> h_outputCpu_NonMaxSupression(count);
	std::vector<float> h_outputCpu_DoubleThreshold(count);

	// Initialize memory to 0xff (useful for debugging because otherwise GPU memory will contain information from last execution)
	memset(h_outputCpu_Gaussian.data(), 255, size);
	memset(h_outputCpu_Sobel.data(), 255, size);
	memset(h_out_segment.data(), 255, size);
	memset(h_outputCpu_NonMaxSupression.data(), 255, size);
	memset(h_outputCpu_DoubleThreshold.data(), 255, size);

	//Perform Canny edge detection and benchmark the execution time
	cpubeginGaussian = Core::getCurrentTime();
	gaussianFilter(h_outputCpu_Gaussian, h_input, countX, countY);
	cpuendGaussian = Core::getCurrentTime();
	cpubeginsobel = Core::getCurrentTime();
	sobelEdgeDetector(h_outputCpu_Sobel, h_outputCpu_Gaussian, h_out_segment, countX, countY);
	cpuendSobel = Core::getCurrentTime();
	nonMaxSuppression(h_outputCpu_NonMaxSupression, h_outputCpu_Sobel, h_out_segment, countX, countY);
	cpuendNonmaxsuppression = Core::getCurrentTime();
	applyDoubleThreshold(h_outputCpu_DoubleThreshold, h_outputCpu_NonMaxSupression, countX, countY);
	cpuendDoublethreshold = Core::getCurrentTime();
	applyEdgeHysteresis(h_outputCpu, h_outputCpu_DoubleThreshold, countX, countY);
	cpuendHysteresis = Core::getCurrentTime();

	//Store the output Images
	Core::writeImagePGM("1_Histogram_Equalization_Cpu_Output.pgm", h_input, countX, countY);
	Core::writeImagePGM("2_Gaussian_Cpu_Output.pgm", h_outputCpu_Gaussian, countX, countY);
	Core::writeImagePGM("3_Sobel_Cpu_Output.pgm", h_outputCpu_Sobel, countX, countY);
	Core::writeImagePGM("4_NonMaxSupression_Cpu_Output.pgm", h_outputCpu_NonMaxSupression, countX, countY);
	Core::writeImagePGM("5_DoubleThreshold_Cpu_Output.pgm", h_outputCpu_DoubleThreshold, countX, countY);
	Core::writeImagePGM("6_CannyEdgeDetection_Cpu_Output.pgm", h_outputCpu, countX, countY);
	
}

