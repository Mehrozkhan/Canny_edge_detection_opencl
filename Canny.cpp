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
float high_threshold = 65 * 1.33;//0.25*255;
float low_threshold = 65 * 0.66;//0.05*255;

Core::TimeSpan cpubeginGaussian = Core::TimeSpan::fromSeconds(0);
Core::TimeSpan cpuendGaussian = Core::TimeSpan::fromSeconds(0);
Core::TimeSpan cpubeginsobel = Core::TimeSpan::fromSeconds(0);
Core::TimeSpan cpuendSobel = Core::TimeSpan::fromSeconds(0);
Core::TimeSpan cpuendNonmaxsuppression = Core::TimeSpan::fromSeconds(0);
Core::TimeSpan cpuendDoublethreshold = Core::TimeSpan::fromSeconds(0);
Core::TimeSpan cpuendHysteresis = Core::TimeSpan::fromSeconds(0);

/**********************************************************************************************************************
* CPU Implementation
***********************************************************************************************************************
*/
int getIndexGlobal(std::size_t countX, int i, int j) {
	return j * countX + i;
}
/* Read value from global array a, return 0 if outside image*/
float getValueGlobal(const std::vector<float>& a, std::size_t countX, std::size_t countY, int i, int j) {
	if (i < 0 || (size_t)i >= countX || j < 0 || (size_t)j >= countY)
		return 0;
	else
		return a[getIndexGlobal(countX, i, j)];
}

/**********************************************************************************************************************
* Function definitions
***********************************************************************************************************************
*/

/******************************************************************************************************************************
* Function name: calculateHistogram
*
* Parameters:
*  h_outputCpu -
*  h_input -
*  countX -
*  countY -
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
			//std::cout << Pixel_value << ", ";

			histogram[Pixel_value]++;
		}
	}

}

/******************************************************************************************************************************
* Function name: histogramEqualization
*
* Parameters:
*  h_outputCpu -
*  h_input -
*  countX -
*  countY -
* Return:
*  void
********************************************************************************************************************************
*/
void histogramEqualization(std::vector<float>& h_outputCpu, std::vector<float>& h_input, std::size_t countX, std::size_t countY)
{

	//std::size_t size = h_input.size();
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
	float normalizationFactor = static_cast<float>(NUMBER_OF_BINS - 1);
	for (int i = 0; i < countX; i++)
	{
		for (int j = 0; j < countY; j++)
		{
			h_outputCpu[getIndexGlobal(countX, i, j)] = std::round(cdf[h_input[getIndexGlobal(countX, i, j)]] * 255 / (countX * countY));
			//h_outputCpu[getIndexGlobal(countX, i, j)] = h_outputCpu[getIndexGlobal(countX, i, j)] ;
			//std::cout << h_outputCpu[getIndexGlobal(countX, i, j)] << ", ";
		}
	}
	calculateHistogram(histogram, h_input, countX, countY);



}

/******************************************************************************************************************************
* Function name: gaussianFilter
*  Applying gaussian blur to remove noise
* Parameters:
*  h_outputCpu -
*  h_input -
*  countX -
*  countY -
* Return:
*  void
********************************************************************************************************************************
*/
void gaussianFilter(std::vector<float>& h_outputCpu, const std::vector<float>& h_input, std::size_t countX, std::size_t countY)
{
	float weights[5][5] = {
		{1,  4,  7,  4, 1},
		{4, 16, 26, 16, 4},
		{7, 26, 41, 26, 7},
		{4, 16, 26, 16, 4},
		{1,  4,  7,  4, 1} };
	for (int y = 2; y < countY - 2; y++)
	{
		for (int x = 2; x < countX - 2; x++)
		{
			float sum = 0.0;
			for (int j = -2; j <= 2; j++)
			{
				for (int i = -2; i <= 2; i++)
				{
					sum += weights[j + 2][i + 2] * h_input[(y + j) * countX + (x + i)];
				}
			}
			h_outputCpu[y * countX + x] = sum / 273; // Normalize by the sum of the kernel elements
		}
	}
}

/******************************************************************************************************************************
* Function name: sobelEdgeDetector
*  Edge detection using sobel filter
* Parameters:
*  h_outputCpu -
*  h_input -
*  h_out_segment-
*  countX -
*  countY -
* Return:
*  void
********************************************************************************************************************************
*/
void sobelEdgeDetector(std::vector<float>& h_outputCpu, const std::vector<float>& h_input, std::vector<int>& h_out_segment, std::size_t countX, std::size_t countY)
{
	for (int i = 0; i < (int)countX; i++) {
		for (int j = 0; j < (int)countY; j++) {
			float Gx = getValueGlobal(h_input, countX, countY, i - 1, j - 1) + 2 * getValueGlobal(h_input, countX, countY, i - 1, j) + getValueGlobal(h_input, countX, countY, i - 1, j + 1)
				- getValueGlobal(h_input, countX, countY, i + 1, j - 1) - 2 * getValueGlobal(h_input, countX, countY, i + 1, j) - getValueGlobal(h_input, countX, countY, i + 1, j + 1);
			float Gy = getValueGlobal(h_input, countX, countY, i - 1, j - 1) + 2 * getValueGlobal(h_input, countX, countY, i, j - 1) + getValueGlobal(h_input, countX, countY, i + 1, j - 1)
				- getValueGlobal(h_input, countX, countY, i - 1, j + 1) - 2 * getValueGlobal(h_input, countX, countY, i, j + 1) - getValueGlobal(h_input, countX, countY, i + 1, j + 1);

			h_outputCpu[getIndexGlobal(countX, i, j)] = sqrt(Gx * Gx + Gy * Gy);

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

				h_out_segment[getIndexGlobal(countX, i, j)] = segment;
			}
		}
	}
}
/*3. Non Max Supression */
/******************************************************************************************************************************
* Function name: nonMaxSuppressio
*
* Parameters:
*  h_outputCpu -
*  h_input -
*  h_in_segment-
*  countX -
*  countY -
* Return:
*  void
********************************************************************************************************************************
*/
void nonMaxSuppression(std::vector<float>& h_outputCpu, const std::vector<float>& h_input, std::vector<int>& h_in_segment, std::size_t countX, std::size_t countY)
{
	for (int i = 0; i < (int)countX; i++) {
		for (int j = 0; j < (int)countY; j++)
		{
			switch (h_in_segment[getIndexGlobal(countX, i, j)]) {
			case 1:
				if (h_input[getIndexGlobal(countX, i, j) - 1] >= h_input[getIndexGlobal(countX, i, j)] || h_input[getIndexGlobal(countX, i, j) + 1] > h_input[getIndexGlobal(countX, i, j)])
					h_outputCpu[getIndexGlobal(countX, i, j)] = 0;
				else h_outputCpu[getIndexGlobal(countX, i, j)] = h_input[getIndexGlobal(countX, i, j)];
				break;
			case 2:
				if (h_input[getIndexGlobal(countX, i, j) - (countX - 1)] >= h_input[getIndexGlobal(countX, i, j)] || h_input[getIndexGlobal(countX, i, j) + (countX - 1)] > h_input[getIndexGlobal(countX, i, j)])
					h_outputCpu[getIndexGlobal(countX, i, j)] = 0;
				else h_outputCpu[getIndexGlobal(countX, i, j)] = h_input[getIndexGlobal(countX, i, j)];
				break;
			case 3:
				if (h_input[getIndexGlobal(countX, i, j) - (countX)] >= h_input[getIndexGlobal(countX, i, j)] || h_input[getIndexGlobal(countX, i, j) + (countX)] > h_input[getIndexGlobal(countX, i, j)])
					h_outputCpu[getIndexGlobal(countX, i, j)] = 0;
				else h_outputCpu[getIndexGlobal(countX, i, j)] = h_input[getIndexGlobal(countX, i, j)];
				break;
			case 4:
				if (h_input[getIndexGlobal(countX, i, j) - (countX + 1)] >= h_input[getIndexGlobal(countX, i, j)] || h_input[getIndexGlobal(countX, i, j) + (countX + 1)] > h_input[getIndexGlobal(countX, i, j)])
					h_outputCpu[getIndexGlobal(countX, i, j)] = 0;
				else h_outputCpu[getIndexGlobal(countX, i, j)] = h_input[getIndexGlobal(countX, i, j)];
				break;
			default:
				h_outputCpu[getIndexGlobal(countX, i, j)] = 0;
				break;
			}
			//std::cout << h_input[getIndexGlobal(countX, i, j)] << ", ";
		}

	}

}

/******************************************************************************************************************************
* Function name: applyDoubleThreshold
*
* Parameters:
*  h_outputCpu -
*  h_input -
*  median -
*  countX -
*  countY -
* Return:
*  void
********************************************************************************************************************************
*/
void applyDoubleThreshold(std::vector<float>& h_outputCpu, const std::vector<float>& h_input, std::size_t countX, std::size_t countY) {


	float sum = 0.0;
	for (int i = 0; i < (int)countX; i++) {
		for (int j = 0; j < (int)countY; j++) {
			sum += h_input[getIndexGlobal(countX, i, j)];
		}
	}

	float mean = sum / (countX * countY);

	
	std::cout << "high_threshold: " << high_threshold << std::endl;
	std::cout << "low_threshold " << low_threshold << std::endl;

	for (int i = 0; i < countX; i++) {
		for (int j = 0; j < countY; j++) {

			if (h_input[getIndexGlobal(countX, i, j)] > high_threshold)
				h_outputCpu[getIndexGlobal(countX, i, j)] = 255;      //absolutely edge
			else if (h_input[getIndexGlobal(countX, i, j)] > low_threshold)
			{
				h_outputCpu[getIndexGlobal(countX, i, j)] = 127;      //potential edge

			}
			else
				h_outputCpu[getIndexGlobal(countX, i, j)] = 0;       //absolutely not edge
		}
	}

}

/******************************************************************************************************************************
* Function name: applyEdgeHysteresis
*
* Parameters:
*  h_outputCpu -
*  h_input -
*  countX -
*  countY -
* Return:
*  void
********************************************************************************************************************************
*/
void applyEdgeHysteresis(std::vector<float>& h_outputCpu, const std::vector<float>& h_input, std::size_t countX, std::size_t countY) {
	memcpy(h_outputCpu.data(), h_input.data(), countX * countY * sizeof(float));
	for (int i = 1; i < countX - 1; i++) {
		for (int j = 1; j < countY - 1; j++) {
			if (h_input[getIndexGlobal(countX, i, j)] == 127) {
				if (h_input[getIndexGlobal(countX, i, j) - 1] == 255 || h_input[getIndexGlobal(countX, i, j) + 1] == 255 ||
					h_input[getIndexGlobal(countX, i, j) - countX] == 255 || h_input[getIndexGlobal(countX, i, j) + countX] == 255 ||
					h_input[getIndexGlobal(countX, i, j) - countX - 1] == 255 || h_input[getIndexGlobal(countX, i, j) - countX + 1] == 255 ||
					h_input[getIndexGlobal(countX, i, j) + countX - 1] == 255 || h_input[getIndexGlobal(countX, i, j) + countX + 1] == 255)
					h_outputCpu[getIndexGlobal(countX, i, j)] = 255;

				else
					h_outputCpu[getIndexGlobal(countX, i, j)] = 0;
			}
		}
	}

}



std::vector<float> applyCanny_CPU(std::vector<float>& h_outputCpu, const std::vector<float>& h_input, std::size_t countX, std::size_t countY,
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

	/*Core::TimeSpan cpubeginGaussian = Core::getCurrentTime();
	gaussianFilter(h_outputCpu_Gaussian, h_input, countX, countY);
	Core::TimeSpan cpuendGaussian = Core::getCurrentTime();
	Core::TimeSpan cpubeginsobel = Core::getCurrentTime();
	sobelEdgeDetector(h_outputCpu_Sobel, h_outputCpu_Gaussian, h_out_segment, countX, countY);
	Core::TimeSpan cpuendSobel = Core::getCurrentTime();
	nonMaxSuppression(h_outputCpu_NonMaxSupression, h_outputCpu_Sobel, h_out_segment, countX, countY);
	Core::TimeSpan cpuendNonmaxsuppression = Core::getCurrentTime();
	applyDoubleThreshold(h_outputCpu_DoubleThreshold, h_outputCpu_NonMaxSupression, countX, countY);
	Core::TimeSpan cpuendDoublethreshold = Core::getCurrentTime();
	applyEdgeHysteresis(h_outputCpu, h_outputCpu_DoubleThreshold, countX, countY);
	Core::TimeSpan cpuendHysteresis = Core::getCurrentTime();
	*/
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

	

	Core::writeImagePGM("Gaussian_Cpu_Output.pgm", h_outputCpu_Gaussian, countX, countY);
	Core::writeImagePGM("Sobel_Cpu_Output.pgm", h_outputCpu_Sobel, countX, countY);
	Core::writeImagePGM("NonMaxSupression_Cpu_Output.pgm", h_outputCpu_NonMaxSupression, countX, countY);
	Core::writeImagePGM("DoubleThreshold_Cpu_Output.pgm", h_outputCpu_DoubleThreshold, countX, countY);
	return h_outputCpu_NonMaxSupression; //to be deleted at the end.
}

