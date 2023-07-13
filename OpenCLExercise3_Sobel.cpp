//////////////////////////////////////////////////////////////////////////////
// OpenCL exercise 3: Sobel filter
//////////////////////////////////////////////////////////////////////////////

// includes
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
#include <cmath>

#include <boost/lexical_cast.hpp>
#define M_PI acos(-1.0)

float low_threshold = 0.05;
float high_threshold = 0.25;

//////////////////////////////////////////////////////////////////////////////
// CPU implementation
//////////////////////////////////////////////////////////////////////////////
int getIndexGlobal(std::size_t countX, int i, int j) {
	return j * countX + i;
}
// Read value from global array a, return 0 if outside image
float getValueGlobal(const std::vector<float>& a, std::size_t countX, std::size_t countY, int i, int j) {
	if (i < 0 || (size_t) i >= countX || j < 0 || (size_t) j >= countY)
		return 0;
	else
		return a[getIndexGlobal(countX, i, j)];
}

/*1. Gaussian Blur to remove noise */
void gaussianfilter(std::vector<float>& h_outputCpu, const std::vector<float>& h_input, std::size_t countX, std::size_t countY)
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

/* 2. Sobel edge detection */
void sobelHost( std::vector<float>& h_outputCpu, const std::vector<float>& h_input, std::vector<int>& h_out_segment, std::size_t countX, std::size_t countY)
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
			//if (Gx != 0.0 || Gy != 0.0)
			//{
				if ((theta <= 22.5 && theta >= -22.5) || (theta <= -157.5) || (theta >= 157.5))
					segment = 1;  // "-"
				else if ((theta > 22.5 && theta <= 67.5) || (theta > -157.5 && theta <= -112.5))
					segment = 2;  // "/" 
				else if ((theta > 67.5 && theta <= 112.5) || (theta >= -112.5 && theta < -67.5))
					segment = 3;  // "|"
				else if ((theta >= -67.5 && theta < -22.5) || (theta > 112.5 && theta < 157.5))
					segment = 4;  // "\"  
				else
					std::cout << "error " << theta << std::endl;
				h_out_segment[getIndexGlobal(countX, i, j)] = segment;
			//}
		}
	}
}
/*3. Non Max Supression */
void non_max_suppression(std::vector<float>& h_outputCpu, const std::vector<float>& h_input, std::vector<int>& h_in_segment, std::size_t countX, std::size_t countY)
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

void apply_double_threshold(std::vector<float>& h_outputCpu, const std::vector<float>& h_input, float strong_threshold, float weak_threshold, std::size_t countX, std::size_t countY) {
	
	
	float sum = 0.0;
	for (int i = 0; i < (int)countX; i++) {
		for (int j = 0; j < (int)countY; j++) {
			sum += h_input[getIndexGlobal(countX, i, j)];
		}
	}

	float mean = sum / (countX * countY);

	float highThreshold = mean * 2;  // Adjust the multiplier as needed
	float lowThreshold = mean * 1;   // Adjust the multiplier as needed

	int strong_edge = 255;
	int weak_edge = 128;
	
	for (int i = 0; i < (int)countX; i++) {
		for (int j = 0; j < (int)countY; j++) {
			
			if (h_input[getIndexGlobal(countX, i, j)] > highThreshold)
				h_outputCpu[getIndexGlobal(countX, i, j)] = 1.0;      //absolutely edge
			else if (h_input[getIndexGlobal(countX, i, j)] > lowThreshold)
			{
				h_outputCpu[getIndexGlobal(countX, i, j)] = 0.5;      //potential edge
				
			}
			else
				h_outputCpu[getIndexGlobal(countX, i, j)] = 0;       //absolutely not edge
		}
	}
}

void apply_edge_hysteresis(std::vector<float>& h_outputCpu, const std::vector<float>& h_input, std::size_t countX, std::size_t countY) {
	memcpy(h_outputCpu.data(), h_input.data(), countX * countY * sizeof(float));
	for (int i = 1; i < countX - 1; i++) {
		for (int j = 1; j < countY - 1; j++) {
			//int src_pos = x + (y * countX);
			if (h_input[ getIndexGlobal(countX, i, j) ] == 0.5) {
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
/*To Do CPU: Apply step 4 on h_outputCpu2 and then step 5 on its output*/
//////////////////////////////////////////////////////////////////////////////
// Main function
//////////////////////////////////////////////////////////////////////////////
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
	cl::Program program = OpenCL::loadProgramSource(context, "C:/Users/SWATHI/Documents/subject materials/semester 3/GPU lab/Exercise 1/For Windows/Opencl-Basics-Windows/Opencl-ex1/src/OpenCLExercise3_Sobel.cl");

	// Compile the source code. This is similar to program.build(devices) but will print more detailed error messages
	OpenCL::buildProgram(program, devices);

	// Declare some values
	std::size_t wgSizeX = 16; // Number of work items per work group in X direction
	std::size_t wgSizeY = 16;
	std::size_t countX = wgSizeX * 40; // Overall number of work items in X direction = Number of elements in X direction
	std::size_t countY = wgSizeY * 30;
	//countX *= 3; countY *= 3;
	std::size_t count = countX * countY; // Overall number of elements
	std::size_t size = count * sizeof (float); // Size of data in bytes

	// Allocate space for output data from CPU and GPU on the host
	std::vector<float> h_input (count);
	std::vector<float> h_outputCpu_Gaussian (count);
	std::vector<float> h_outputCpu_Sobel(count);
	std::vector<int>   h_out_segment(count);
	std::vector<float> h_outputCpu_NonMaxSupression(count);
	std::vector<float> h_outputGpu (count);

	std::vector<float> h_outputCpu_DoubleThreshold(count);
	std::vector<float> h_outputCpu_Hysteresis(count);
	

	// Allocate space for input and output data on the device
	//TODO
	cl::Buffer d_input(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_output(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_outputCpu_Gaussian(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_outputCpu_Sobel(context, CL_MEM_READ_WRITE, size);

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
	//TODO: GPU

	//////// Load input data ////////////////////////////////
	
	// Use an image (Valve.pgm) as input data
	{
		std::vector<float> inputData;
		std::size_t inputWidth, inputHeight;
		Core::readImagePGM("C:/Users/SWATHI/Documents/subject materials/semester 3/GPU lab/Exercise 1/For Windows/Opencl-Basics-Windows/Opencl-ex1/src/Valve.pgm", inputData, inputWidth, inputHeight);
		for (size_t j = 0; j < countY; j++) {
			for (size_t i = 0; i < countX; i++) {
				h_input[i + countX * j] = inputData[(i % inputWidth) + inputWidth * (j % inputHeight)];
			}
		}
	}

	// Do calculation on the host side

	Core::TimeSpan cpubegin = Core::getCurrentTime();
	gaussianfilter(h_outputCpu_Gaussian, h_input, countX, countY);
	sobelHost(h_outputCpu_Sobel, h_outputCpu_Gaussian, h_out_segment, countX, countY);
	non_max_suppression( h_outputCpu_NonMaxSupression, h_outputCpu_Sobel,  h_out_segment, countX, countY);
	apply_double_threshold(h_outputCpu_DoubleThreshold, h_outputCpu_NonMaxSupression, high_threshold, low_threshold, countX, countY);
	apply_edge_hysteresis(h_outputCpu_Hysteresis, h_outputCpu_DoubleThreshold, countX, countY);
	Core::TimeSpan cpuend = Core::getCurrentTime();
	//////// Store CPU output image ///////////////////////////////////
	Core::writeImagePGM("C:/Users/SWATHI/Documents/subject materials/semester 3/GPU lab/output_gaussianfilter_cpu.pgm", h_outputCpu_Gaussian, countX, countY);
	Core::writeImagePGM("C:/Users/SWATHI/Documents/subject materials/semester 3/GPU lab/output_sobel_cpu.pgm", h_outputCpu_Sobel, countX, countY);
	Core::writeImagePGM("C:/Users/SWATHI/Documents/subject materials/semester 3/GPU lab/output_non_max_suppression.pgm", h_outputCpu_NonMaxSupression, countX, countY);
	Core::writeImagePGM("C:/Users/SWATHI/Documents/subject materials/semester 3/GPU lab/output_DoubleThreshold.pgm", h_outputCpu_DoubleThreshold, countX, countY);
	Core::writeImagePGM("C:/Users/SWATHI/Documents/subject materials/semester 3/GPU lab/output_Hysterisis.pgm", h_outputCpu_Hysteresis, countX, countY);
	std::cout << std::endl;

	
	// Iterate over all implementations (task 1 - 3)
	for (int impl = 1; impl <= 1; impl++) {
		std::cout << "Implementation #" << impl << ":" << std::endl;

		// Reinitialize output memory to 0xff
		memset(h_outputGpu.data(), 255, size);
		//TODO: GPU

		// Copy input data to device
		//TODO


		cl::Event event1;
		
		queue.enqueueWriteBuffer(d_input, true, 0, size, h_outputCpu_Gaussian.data(), NULL, &event1);


		// Create a kernel object
		
		cl::Kernel sobelKernel(program, "sobelKernel");

		// Launch kernel on the device
		//TODO
		
		sobelKernel.setArg<cl::Buffer>(0, d_input);
		sobelKernel.setArg<cl::Buffer>(1, d_output);
		//sobelKernel.setArg<cl::Buffer>(2, d_out_segment);
		
		
		cl::Event event2;

		queue.enqueueNDRangeKernel(sobelKernel,
			cl::NullRange,
			cl::NDRange(countX, countY),
			cl::NDRange(wgSizeX, wgSizeY),
			NULL,
			&event2);

		// Copy output data back to host
		//TODO

		cl::Event event3;
		queue.enqueueReadBuffer(d_output, true, 0, size, h_outputGpu.data(), NULL, &event3);
		//queue.enqueueReadBuffer(d_out_segment, true, 0, size, h_outputGpu.data(), NULL, &event3);
		// Print performance data
		//TODO
		std::cout << "performance data for implementation "<<impl<<":"<< std::endl;
		Core::TimeSpan cputime = cpuend - cpubegin;
		std::cout << "cpu time: " << cputime.toString() << std::endl;
		Core::TimeSpan gputime1 = OpenCL::getElapsedTime(event2);

		Core::TimeSpan gputime2 = OpenCL::getElapsedTime(event3); 

		Core::TimeSpan totalgputime = gputime1+gputime2;
		std::cout << "GPU time before copying output data: " << gputime1.toString() << " speedup before copy = " << (cputime.getSeconds() / gputime1.getSeconds())<< std::endl;
		std::cout << "GPU time after copying output data: " << totalgputime.toString() << " speedup after copy = " << (cputime.getSeconds() / totalgputime.getSeconds())<< std::endl;

		//////// Store GPU output image ///////////////////////////////////
		Core::writeImagePGM("output_sobel_gpu_" + boost::lexical_cast<std::string> (impl) + ".pgm", h_outputGpu, countX, countY);

		// Check whether results are correct
		std::size_t errorCount = 0;
		for (size_t i = 0; i < countX; i = i + 1) { //loop in the x-direction
			for (size_t j = 0; j < countY; j = j + 1) { //loop in the y-direction
				size_t index = i + j * countX;
				// Allow small differences between CPU and GPU results (due to different rounding behavior)
				if (!(std::abs (h_outputCpu_Gaussian[index] - h_outputGpu[index]) <= 1e-5)) {
					if (errorCount < 15)
						std::cout << "Result for " << i << "," << j << " is incorrect: GPU value is " << h_outputGpu[index] << ", CPU value is " << h_outputCpu_Gaussian[index] << std::endl;
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
	}

	std::cout << "Success" << std::endl;

	return 0;
}
