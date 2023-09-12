/*
* Canny Edge Detection
* This program implements Canny Edge detection through 5 steps:
* ........
* Project Team:
* 
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
#include <vector>
#include <cmath>

#include <boost/lexical_cast.hpp>



#define M_PI acos(-1.0)
#define NUMBER_OF_BINS 256

/**********************************************************************************************************************
* Global variables 
***********************************************************************************************************************
*/
//int median = -1;
float high_threshold = 65 * 1.33;//0.25*255;
float low_threshold = 65 * 0.66;//0.05*255;

/**********************************************************************************************************************
* CPU Implementation
***********************************************************************************************************************
*/
int getIndexGlobal(std::size_t countX, int i, int j) {
	return j * countX + i;
}
/* Read value from global array a, return 0 if outside image*/
float getValueGlobal(const std::vector<float>& a, std::size_t countX, std::size_t countY, int i, int j) {
	if (i < 0 || (size_t) i >= countX || j < 0 || (size_t) j >= countY)
		return 0;
	else
		return a[getIndexGlobal(countX, i, j)];
}

/**********************************************************************************************************************
* Function definitions
***********************************************************************************************************************
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

void histogramEqualization(std::vector<float>& h_outputCpu,  std::vector<float>& h_input, std::size_t countX, std::size_t countY)
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
void sobelEdgeDetector( std::vector<float>& h_outputCpu, const std::vector<float>& h_input, std::vector<int>& h_out_segment, std::size_t countX, std::size_t countY)
{
	for (int i = 0; i < (int) countX; i++) {
		for (int j = 0; j < (int) countY; j++) {
			float Gx = getValueGlobal(h_input, countX, countY, i-1, j-1)+2*getValueGlobal(h_input, countX, countY, i-1, j)+getValueGlobal(h_input, countX, countY, i-1, j+1)
					-getValueGlobal(h_input, countX, countY, i+1, j-1)-2*getValueGlobal(h_input, countX, countY, i+1, j)-getValueGlobal(h_input, countX, countY, i+1, j+1);
			float Gy = getValueGlobal(h_input, countX, countY, i-1, j-1)+2*getValueGlobal(h_input, countX, countY, i, j-1)+getValueGlobal(h_input, countX, countY, i+1, j-1)
					-getValueGlobal(h_input, countX, countY, i-1, j+1)-2*getValueGlobal(h_input, countX, countY, i, j+1)-getValueGlobal(h_input, countX, countY, i+1, j+1);
			
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
void applyDoubleThreshold(std::vector<float>& h_outputCpu, const std::vector<float>& h_input,  std::size_t countX, std::size_t countY) {
	
	
	float sum = 0.0;
	for (int i = 0; i < (int)countX; i++) {
		for (int j = 0; j < (int)countY; j++) {
			sum += h_input[getIndexGlobal(countX, i, j)];
		}
	}

	float mean = sum / (countX * countY);

	/*high_threshold = mean * 1.33;  // Adjust the multiplier as needed
	low_threshold = mean * 0.66;   // Adjust the multiplier as needed
	std::cout << "mean: " << mean << std::endl;
	std::cout << "high_threshold: " << high_threshold << std::endl;
	std::cout << "low_threshold " << low_threshold << std::endl;
	
	high_threshold = median * 1.33;//0.25*255;
	low_threshold = median * 0.66;//0.05*255;*/
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
			//int src_pos = x + (y * countX);
			if (h_input[ getIndexGlobal(countX, i, j) ] == 127) {
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
	/*for (int i = 0; i < (int)countX; i++) {
		for (int j = 0; j < (int)countY; j++) {
			std::cout << h_outputCpu[getIndexGlobal(countX, i, j)] << ", ";
		}
	}*/
}

/******************************************************************************************************************************
* Function name: main
*
* Parameters:
*  h_outputCpu -
*  h_input -
*  countX -
*  countY -
* Return:
*  int
********************************************************************************************************************************
*/

int main(int argc, char** argv) {
	// Create a context	
	//cl::Context context(CL_DEVICE_TYPE_GPU);
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.size() == 0) {
		std::cerr << "No platforms found" << std::endl;
		return 1;
	}
	int platformId = 0;
	for (size_t i = 0; i < platforms.size(); i++) {
		if (platforms[i].getInfo<CL_PLATFORM_NAME>() == "AMD Accelerated Parallel Processing") {
			platformId = i;
			break;
		}
	}
	cl_context_properties prop[4] = { CL_CONTEXT_PLATFORM, (cl_context_properties) platforms[platformId] (), 0, 0 };
	std::cout << "Using platform '" << platforms[platformId].getInfo<CL_PLATFORM_NAME>() << "' from '" << platforms[platformId].getInfo<CL_PLATFORM_VENDOR>() << "'" << std::endl;
	cl::Context context(CL_DEVICE_TYPE_GPU, prop);


	// Get a device of the context
	int deviceNr = argc < 2 ? 1 : atoi(argv[1]);
	std::cout << "Using device " << deviceNr << " / " << context.getInfo<CL_CONTEXT_DEVICES>().size() << std::endl;
	ASSERT (deviceNr > 0);
	ASSERT ((size_t) deviceNr <= context.getInfo<CL_CONTEXT_DEVICES>().size());
	cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[deviceNr - 1];
	std::vector<cl::Device> devices;
	devices.push_back(device);
	OpenCL::printDeviceInfo(std::cout, device);

	// Create a command queue
	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

	// Load the source code
	cl::Program program = OpenCL::loadProgramSource(context, "../../../src/CannyEdgeDetection.cl");

	// Compile the source code. This is similar to program.build(devices) but will print more detailed error messages
	OpenCL::buildProgram(program, devices);

	// Use an image (Valve.pgm) as input data
	std::vector<float> inputData;
	std::size_t inputWidth, inputHeight;
	Core::readImagePGM("../../../src/InputImages/Valve.pgm", inputData, inputWidth, inputHeight);

	// Declare some values
	std::size_t wgSizeX = 16; // Number of work items per work group in X direction
	std::size_t wgSizeY = 16;
	std::size_t countX = wgSizeX * (inputWidth/16); // Overall number of work items in X direction = Number of elements in X direction
	std::size_t countY = wgSizeY * (inputHeight/16);
	//countX *= 3; countY *= 3;
	std::size_t count = countX * countY; // Overall number of elements
	std::size_t size = count * sizeof (float); // Size of data in bytes


	// Allocate space for output data from CPU and GPU on the host
	std::vector<float> h_input (count);
	
	std::vector<float> h_outputCpu_Gaussian (count);
	std::vector<float> h_outputCpu_Sobel(count);
	std::vector<int>   h_out_segment(count);
	std::vector<float> h_outputCpu_NonMaxSupression(count);

	std::vector<float> h_outputCpu_HistogramEqualization(count);
	std::vector<float> h_outputGpu_HistogramEqualization(count);

	std::vector<int> h_outputGpu_HistogramCalculation(NUMBER_OF_BINS);
	std::vector<int> cdf(NUMBER_OF_BINS);

	std::vector<float> h_outputCpu_DoubleThreshold(count);
	std::vector<float> h_outputCpu_Hysteresis(count);
	std::vector<float> h_outputGpu(count);
	std::vector<int> h_out_segmentGpu(count);
	std::vector<float> h_outputGpuDoublethreshold(count);
	std::vector<float> h_outputGpu_Hysteresis(count);

	// Allocate space for input and output data on the device
	//TODO
	cl::Buffer d_input(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_output(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_output_HistogramCalculation(context, CL_MEM_READ_WRITE, NUMBER_OF_BINS* sizeof(int));
	cl::Buffer d_input_HistogramEqualization(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_output_HistogramEqualization(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_outputCpu_Gaussian(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_outputCpu_Sobel(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_inputDt(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_outputDt(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_inputHst(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_outputHst(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_out_segment(context, CL_MEM_READ_WRITE, size);
	
	
	// Initialize memory to 0xff (useful for debugging because otherwise GPU memory will contain information from last execution)
	memset(h_input.data(), 255, size);
	
	memset(h_outputCpu_Gaussian.data(), 255, size);
	memset(h_outputCpu_Sobel.data(), 255, size);
	memset(h_out_segment.data(), 255, size);
	memset(h_outputCpu_NonMaxSupression.data(), 255, size);
	memset(h_outputCpu_DoubleThreshold.data(), 255, size);
	memset(h_outputCpu_Hysteresis.data(), 255, size);
	memset(h_outputGpu.data(), 255, size);
	memset(h_outputGpu_HistogramCalculation.data(), 255, NUMBER_OF_BINS * sizeof(int));
	memset(cdf.data(), 255, NUMBER_OF_BINS * sizeof(int));
	memset(h_outputGpu_HistogramEqualization.data(), 255, size);
	memset(h_outputGpuDoublethreshold.data(), 255, size);
	memset(h_outputGpu_Hysteresis.data(), 255, size);
	memset(h_out_segmentGpu.data(), 255, size);

	
	//TODO: GPU

	//////// Load input data ////////////////////////////////
	
	
	for (size_t j = 0; j < countY; j++) {
		for (size_t i = 0; i < countX; i++) {

			h_input[i + countX * j] = inputData[(i % inputWidth) + inputWidth * (j % inputHeight)]; //not normalized

			if (h_input[i + countX * j] < 0)
			{
				h_input[i + countX * j] = 0;
			}
			else if (h_input[i + countX * j] > 255)
			{
				h_input[i + countX * j] = 255;

			}


		}
	}
		
	// Do calculation on the host side

	Core::TimeSpan cpubegin = Core::getCurrentTime();
	histogramEqualization(h_outputCpu_HistogramEqualization, h_input, countX, countY);
	gaussianFilter(h_outputCpu_Gaussian, h_outputCpu_HistogramEqualization, countX, countY);
	//gaussianFilter(h_outputCpu_Gaussian, h_input, countX, countY);
	sobelEdgeDetector(h_outputCpu_Sobel, h_outputCpu_Gaussian, h_out_segment, countX, countY);
	nonMaxSuppression( h_outputCpu_NonMaxSupression, h_outputCpu_Sobel,  h_out_segment, countX, countY);
	applyDoubleThreshold(h_outputCpu_DoubleThreshold, h_outputCpu_NonMaxSupression,  countX, countY);
	applyEdgeHysteresis(h_outputCpu_Hysteresis, h_outputCpu_DoubleThreshold, countX, countY);
	Core::TimeSpan cpuend = Core::getCurrentTime();
	
	//////// Store CPU output image ///////////////////////////////////
	//Core::writeImagePGM("input.pgm", h_input, countX, countY);
	Core::writeImagePGM("output_gaussianfilter_cpu.pgm", h_outputCpu_Gaussian, countX, countY);
	Core::writeImagePGM("output_sobel_cpu.pgm", h_outputCpu_Sobel, countX, countY);
	Core::writeImagePGM("output_non_max_suppression.pgm", h_outputCpu_NonMaxSupression, countX, countY);
	Core::writeImagePGM("output_DoubleThreshold.pgm", h_outputCpu_DoubleThreshold, countX, countY);
	Core::writeImagePGM("output_Hysterisis.pgm", h_outputCpu_Hysteresis, countX, countY);
	std::cout << std::endl;

	// Reinitialize output memory to 0xff
	memset(h_outputGpu.data(), 255, size);
	memset(cdf.data(), 255, NUMBER_OF_BINS * sizeof(int));
	memset(h_outputGpu_HistogramEqualization.data(), 255, size);
	memset(h_out_segmentGpu.data(), 255, size);
	memset(h_outputGpuDoublethreshold.data(), 255, size);
	memset(h_outputGpu_Hysteresis.data(), 255, size);
	//TODO: GPU

	// Copy input data to device
	// TODO	

	queue.enqueueWriteBuffer(d_input, true, 0, size, h_input.data(), NULL, NULL);
	cl::Kernel calculateHistogramKernel(program, "calculateHistogramKernel");
	calculateHistogramKernel.setArg<cl::Buffer>(0, d_input);
	calculateHistogramKernel.setArg<cl::Buffer>(1, d_output_HistogramCalculation);

	queue.enqueueNDRangeKernel(calculateHistogramKernel,
		cl::NullRange,
		cl::NDRange(countX, countY),
		cl::NDRange(wgSizeX, wgSizeY),
		NULL,
		NULL);

	queue.enqueueReadBuffer(d_output_HistogramCalculation, true, 0, NUMBER_OF_BINS * sizeof(int), h_outputGpu_HistogramCalculation.data(), NULL, NULL);

	cdf[0] = h_outputGpu_HistogramCalculation[0];
	for (int i = 1; i < NUMBER_OF_BINS; i++)
	{
		cdf[i] = cdf[i - 1] + h_outputGpu_HistogramCalculation[i];
		std::cout << h_outputGpu_HistogramCalculation[i] << ",";
	}

	/* Histogram Equalization */
	queue.enqueueWriteBuffer(d_input, true, 0, size, h_input.data(), NULL, NULL);
	queue.enqueueWriteBuffer(d_output_HistogramCalculation, true, 0, NUMBER_OF_BINS * sizeof(int), cdf.data(), NULL, NULL);
	cl::Kernel histogramEqualizationKernel(program, "histogramEqualizationKernel");
	histogramEqualizationKernel.setArg<cl::Buffer>(0, d_input);
	histogramEqualizationKernel.setArg<cl::Buffer>(1, d_output_HistogramCalculation);
	histogramEqualizationKernel.setArg<cl::Buffer>(2, d_output_HistogramEqualization);
	
	
	queue.enqueueNDRangeKernel(histogramEqualizationKernel,
		cl::NullRange,
		cl::NDRange(countX, countY),
		cl::NDRange(wgSizeX, wgSizeY),
		NULL,
		NULL);

	queue.enqueueReadBuffer(d_output_HistogramEqualization, true, 0, size, h_outputGpu_HistogramEqualization.data(), NULL, NULL);

	
	

	
	cl::Event event1;
	queue.enqueueWriteBuffer(d_input, true, 0, size, h_outputCpu_Gaussian.data(), NULL, &event1);


	// Create a kernel object
	cl::Kernel sobelKernel(program, "sobelKernel");


	// Launch kernel on the device
	//TODO
	sobelKernel.setArg<cl::Buffer>(0, d_input);
	sobelKernel.setArg<cl::Buffer>(1, d_output);
	sobelKernel.setArg<cl::Buffer>(2, d_out_segment);


	cl::Event event2;
	queue.enqueueNDRangeKernel(sobelKernel,
		cl::NullRange,
		cl::NDRange(countX, countY),
		cl::NDRange(wgSizeX, wgSizeY),
		NULL,
		&event2);

	// Copy output data back to host
	// TODO

	cl::Event event3;
	queue.enqueueReadBuffer(d_output, true, 0, size, h_outputGpu.data(), NULL, &event3);
	queue.enqueueReadBuffer(d_out_segment, true, 0, size, h_out_segmentGpu.data(), NULL, &event3);
	
	cl::Event eventDt1;
	queue.enqueueWriteBuffer(d_inputDt, true, 0, size,/*h_outputGpu_NonMaxSupression.data()*/h_outputCpu_NonMaxSupression.data(), NULL, &eventDt1);
	// Create a kernel object
	cl::Kernel DoubleThresholdKernel(program, "DoubleThresholdKernel");
	// Launch kernel on the device
	//TODO
	DoubleThresholdKernel.setArg<cl::Buffer>(0, d_inputDt);
	DoubleThresholdKernel.setArg<cl::Buffer>(1, d_outputDt);
	DoubleThresholdKernel.setArg<cl_float>(2, low_threshold);
	DoubleThresholdKernel.setArg<cl_float>(3, high_threshold);


	cl::Event eventDt2;
	queue.enqueueNDRangeKernel(DoubleThresholdKernel, cl::NullRange, cl::NDRange(countX, countY), cl::NDRange(wgSizeX, wgSizeY), NULL, &eventDt2);
	// Copy output data back to host
	//TODO
	cl::Event eventDt3;
	queue.enqueueReadBuffer(d_outputDt, true, 0, size, h_outputGpuDoublethreshold.data(), NULL, &eventDt3);

	cl::Event eventHst1;
	queue.enqueueWriteBuffer(d_inputHst, true, 0, size, h_outputGpuDoublethreshold.data(), NULL, &eventHst1);
	// Create a kernel object
	cl::Kernel HysteresisKernel(program, "HysteresisKernel");
	// Launch kernel on the device
	//TODO
	HysteresisKernel.setArg<cl::Buffer>(0, d_inputHst);
	HysteresisKernel.setArg<cl::Buffer>(1, d_outputHst);

	cl::Event eventHst2;
	queue.enqueueNDRangeKernel(HysteresisKernel,cl::NullRange, cl::NDRange(countX, countY), cl::NDRange(wgSizeX, wgSizeY), NULL, &eventHst2);
	// Copy output data back to host
	//TODO
	cl::Event eventHst3;
	queue.enqueueReadBuffer(d_outputHst, true, 0, size, h_outputGpu_Hysteresis.data(), NULL, &eventHst3);
	// Print performance data
	//TODO
	/*std::cout << "performance data for implementation :" << std::endl;
	Core::TimeSpan cputime = cpuend - cpubegin;
	std::cout << "cpu time: " << cputime.toString() << std::endl;
	Core::TimeSpan gputime1 = OpenCL::getElapsedTime(event2);

	Core::TimeSpan gputime2 = OpenCL::getElapsedTime(event3);

	Core::TimeSpan totalgputime = gputime1 + gputime2;
	std::cout << "GPU time before copying output data: " << gputime1.toString() << " speedup before copy = " << (cputime.getSeconds() / gputime1.getSeconds()) << std::endl;
	std::cout << "GPU time after copying output data: " << totalgputime.toString() << " speedup after copy = " << (cputime.getSeconds() / totalgputime.getSeconds()) << std::endl;
	*/
	//////// Store GPU output image ///////////////////////////////////
	Core::writeImagePGM("output_sobel_gpu.pgm", h_outputGpu, countX, countY);
	Core::writeImagePGM("output_DoubleThreshold_gpu.pgm", h_outputGpuDoublethreshold, countX, countY);
	Core::writeImagePGM("output_HysteresisGPU.pgm", h_outputGpu_Hysteresis, countX, countY);
	// Check whether results are correct
	std::size_t errorCount = 0;
	for (size_t i = 0; i < countX; i = i + 1) { //loop in the x-direction
		for (size_t j = 0; j < countY; j = j + 1) { //loop in the y-direction
			size_t index = i + j * countX;
			// Allow small differences between CPU and GPU results (due to different rounding behavior)
			if (!(std::abs(h_outputGpu_HistogramEqualization[index] - h_outputCpu_HistogramEqualization[index]) <= 1e-5)) {
				if (errorCount < 15)
					std::cout << "Result for " << i << "," << j << " is incorrect: GPU value is " << h_outputGpu_HistogramEqualization[index] << ", CPU value is " << h_outputCpu_HistogramEqualization[index] << std::endl;
				else if (errorCount == 15)
					std::cout << "..." << std::endl;
				errorCount++;
			}
		}
	}
	if (errorCount != 0) {
		std::cout << "Found " << errorCount << " incorrect results" << std::endl;
		return 1;
	} 

	std::cout << std::endl;
	

	std::cout << "Success" << std::endl;
	
	return 0;
}

