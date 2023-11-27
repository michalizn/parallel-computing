#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <immintrin.h> // Include Intel Intrinsics header
#include "src/util.h"
#include "src/opencl_util.h"
#include <CL/cl.h>

// Is used to find out frame times
int previousFinishTime = 0;
unsigned int frameNumber = 0;
unsigned int seed = 0;

typedef struct {
    uint16_t x;
    uint16_t y;
} coord_t;

const coord_t neighbour_offsets[8] = {
    {-1, -1}, {0, -1},  {+1, -1}, {-1, 0},
    {+1, 0},  {-1, +1}, {0, +1},  {+1, +1},
};

/* OpenCL */
// Use this to check the output of each API call
cl_int status;
// Retrieve the number of platforms
cl_uint numPlatforms = 0;
// ## You may add your own variables here ##
// Allocate enough space for each platform
cl_platform_id *platforms = NULL;
// Retrieve the number of devices
cl_uint numDevices = 0;
// Allocate enough space for each device
cl_device_id *devices;
// Create a context and associate it with the devices
cl_context context;
// Create a command queue and associate it with the device 
cl_command_queue cmdQueue;

// Create the kernels
cl_kernel kernelSobel3x3;
cl_kernel kernelPhaseAndMagnitude;
cl_kernel kernelNonMaxSuppression;
// cl_kernel kernelEdgeTracing;
// Create the program
cl_program program;
// Create the buffers
cl_mem bufInputImage;
cl_mem bufSobel_x;
cl_mem bufSobel_y;
cl_mem bufPhase;
cl_mem bufMagnitude;
cl_mem bufOutputImage;
// Create the events
cl_event kernelSobel3x3Event;
cl_event kernelPhaseAndMagnitudeEvent;
cl_event kernelNonMaxSuppressionEvent;
// cl_event kernelEdgeTracingEvent;

cl_event bufferWriteInputImageEvent;
cl_event bufferWriteSobelxEvent;
cl_event bufferWriteSobelyEvent;
cl_event bufferWritePhaseEvent;
cl_event bufferWriteMagnitudeEvent;
cl_event bufferReadOutputImageEvent;
cl_event bufferReadSobelxEvent;
cl_event bufferReadSobelyEvent;
cl_event bufferReadPhaseEvent;
cl_event bufferReadMagnitudeEvent;

// Global constant for program source
const char* programSource;

void chk(cl_int status, const char* cmd) {

   if(status != CL_SUCCESS) {
      printf("%s failed (%d)\n", cmd, status);
      exit(-1);
   }
}

// Utility function to convert 2d index with offset to linear index
// Uses clamp-to-edge out-of-bounds handling
size_t
idx(size_t x, size_t y, size_t width, size_t height, int xoff, int yoff) {
    size_t resx = x;
    if ((xoff > 0 && x < width - xoff) || (xoff < 0 && x >= (-xoff)))
        resx += xoff;
    size_t resy = y;
    if ((yoff > 0 && y < height - yoff) || (yoff < 0 && y >= (-yoff)))
        resy += yoff;
    return resy * width + resx;
}

void
sobel3x3(
    const uint8_t *restrict in, size_t width, size_t height,
    int16_t *restrict output_x, int16_t *restrict output_y) {
    // LOOP 1.1
#pragma omp parallel for
    for (size_t y = 0; y < height; y++) {
        // LOOP 1.2
        //#pragma omp parallel for              // when used it is slower
        for (size_t x = 0; x < width; x++) {
            size_t gid = y * width + x;
            /* 3x3 sobel filter, first in x direction */
            output_x[gid] = (-1) * in[idx(x, y, width, height, -1, -1)] +
                            1 * in[idx(x, y, width, height, 1, -1)] +
                            (-2) * in[idx(x, y, width, height, -1, 0)] +
                            2 * in[idx(x, y, width, height, 1, 0)] +
                            (-1) * in[idx(x, y, width, height, -1, 1)] +
                            1 * in[idx(x, y, width, height, 1, 1)];
            /* 3x3 sobel filter, in y direction */
            output_y[gid] = (-1) * in[idx(x, y, width, height, -1, -1)] +
                            1 * in[idx(x, y, width, height, -1, 1)] +
                            (-2) * in[idx(x, y, width, height, 0, -1)] +
                            2 * in[idx(x, y, width, height, 0, 1)] +
                            (-1) * in[idx(x, y, width, height, 1, -1)] +
                            1 * in[idx(x, y, width, height, 1, 1)];
        }
    }
}

void
phaseAndMagnitude(
    const int16_t *restrict in_x, const int16_t *restrict in_y, size_t width,
    size_t height, uint8_t *restrict phase_out,
    uint16_t *restrict magnitude_out) {
    // LOOP 2.1
#pragma omp parallel for
    for (size_t y = 0; y < height; y++) {
        // LOOP 2.2
        //#pragma omp parallel for        //  when used it is slower
        for (size_t x = 0; x < width; x++) {
            size_t gid = y * width + x;

            // Output in range -PI:PI
            float angle = atan2f(in_y[gid], in_x[gid]);

            // Shift range -1:1
            angle /= PI;

            // Shift range -127.5:127.5
            angle *= 127.5;

            // Shift range 0.5:255.5
            angle += (127.5 + 0.5);

            // Downcasting truncates angle to range 0:255
            phase_out[gid] = (uint8_t)angle;
            
            magnitude_out[gid] = abs(in_x[gid]) + abs(in_y[gid]);
        }
    }
}

void
nonMaxSuppression(
    const uint16_t *restrict magnitude, const uint8_t *restrict phase,
    size_t width, size_t height, int16_t threshold_lower,
    uint16_t threshold_upper, uint8_t *restrict out) {
    // LOOP 3.1
#pragma omp parallel for
    for (size_t y = 0; y < height; y++) {
        // LOOP 3.2
        //#pragma omp parallel for        //  when used it is slower
        for (size_t x = 0; x < width; x++) {
            size_t gid = y * width + x;

            uint8_t sobel_angle = phase[gid];

            if (sobel_angle > 127) {
                sobel_angle -= 128;
            }

            int sobel_orientation = 0;

            if (sobel_angle < 16 || sobel_angle >= (7 * 16)) {
                sobel_orientation = 2;
            } else if (sobel_angle >= 16 && sobel_angle < 16 * 3) {
                sobel_orientation = 1;
            } else if (sobel_angle >= 16 * 3 && sobel_angle < 16 * 5) {
                sobel_orientation = 0;
            } else if (sobel_angle > 16 * 5 && sobel_angle <= 16 * 7) {
                sobel_orientation = 3;
            }

            uint16_t sobel_magnitude = magnitude[gid];
            /* Non-maximum suppression
             * Pick out the two neighbours that are perpendicular to the
             * current edge pixel */
            uint16_t neighbour_max = 0;
            uint16_t neighbour_max2 = 0;
            switch (sobel_orientation) {
                case 0:
                    neighbour_max =
                        magnitude[idx(x, y, width, height, 0, -1)];
                    neighbour_max2 =
                        magnitude[idx(x, y, width, height, 0, 1)];
                    break;
                case 1:
                    neighbour_max =
                        magnitude[idx(x, y, width, height, -1, -1)];
                    neighbour_max2 =
                        magnitude[idx(x, y, width, height, 1, 1)];
                    break;
                case 2:
                    neighbour_max =
                        magnitude[idx(x, y, width, height, -1, 0)];
                    neighbour_max2 =
                        magnitude[idx(x, y, width, height, 1, 0)];
                    break;
                case 3:
                default:
                    neighbour_max =
                        magnitude[idx(x, y, width, height, 1, -1)];
                    neighbour_max2 =
                        magnitude[idx(x, y, width, height, -1, 1)];
                    break;
            }
            // Suppress the pixel here
            if ((sobel_magnitude < neighbour_max) ||
                (sobel_magnitude < neighbour_max2)) {
                sobel_magnitude = 0;
            }

            /* Double thresholding */
            // Marks YES pixels with 255, NO pixels with 0 and MAYBE pixels
            // with 127
            uint8_t t = 127;
            if (sobel_magnitude > threshold_upper) t = 255;
            if (sobel_magnitude <= threshold_lower) t = 0;
            out[gid] = t;
        }
    }
}

void
edgeTracing(uint8_t *restrict image, size_t width, size_t height) {
    // Uses a stack-based approach to incrementally spread the YES
    // pixels to every (8) neighbouring MAYBE pixel.
    //
    // Modifies the pixels in-place.
    //
    // Since the same pixel is never added to the stack twice,
    // the maximum stack size is quaranteed to never be above
    // the image size and stack overflow should be impossible
    // as long as stack size is 2*2*image_size (2 16-bit coordinates per
    // pixel).
    coord_t *tracing_stack = malloc(width * height * sizeof(coord_t));
    coord_t *tracing_stack_pointer = tracing_stack;

    // LOOP 4.1
//#pragma omp parallel for              // cannot be used, gives an arror mesasage
    for (uint16_t y = 0; y < height; y++) {
        // LOOP 4.2
        //#pragma omp parallel for 
        for (uint16_t x = 0; x < width; x++) {
            // Collect all YES pixels into the stack
            if (image[y * width + x] == 255) {
                coord_t yes_pixel = {x, y};
                *tracing_stack_pointer = yes_pixel;
//#pragma omp atomic
                tracing_stack_pointer++;  // increments by sizeof(coord_t)
            }
        }
    }

    // Empty the tracing stack one-by-one
    // LOOP 4.3
    while (tracing_stack_pointer != tracing_stack) {
        tracing_stack_pointer--;
        coord_t known_edge = *tracing_stack_pointer;
        // LOOP 4.4
        for (int k = 0; k < 8; k++) {
            coord_t dir_offs = neighbour_offsets[k];
            coord_t neighbour = {
                known_edge.x + dir_offs.x, known_edge.y + dir_offs.y};

            // Clamp to edge to prevent the algorithm from leaving the image.
            // Not using the idx()-function, since we want to preserve the x
            // and y on their own, since the pixel might be added to the stack
            // in the end.
            if (neighbour.x < 0) neighbour.x = 0;
            if (neighbour.x >= width) neighbour.x = width - 1;
            if (neighbour.y < 0) neighbour.y = 0;
            if (neighbour.y >= height) neighbour.y = height - 1;

            // Only MAYBE neighbours are potential edges
            if (image[neighbour.y * width + neighbour.x] == 127) {
                // Convert MAYBE to YES
                image[neighbour.y * width + neighbour.x] = 255;

                // Add the newly added pixel to stack, so changes will
                // propagate
                *tracing_stack_pointer = neighbour;
                tracing_stack_pointer++;
            }
        }
    }
    // Clear all remaining MAYBE pixels to NO, these were not reachable from
    // any YES pixels
    // LOOP 4.5
#pragma omp parallel for
    for (int y = 0; y < height; y++) {
        // LOOP 4.6
        //#pragma omp parallel for
    #pragma omp simd
        for (int x = 0; x < width; x++) {
            if (image[y * width + x] == 127) {
                image[y * width + x] = 0;
            }
        }
    }

}

void
cannyEdgeDetection(
    uint8_t *restrict input, size_t width, size_t height,
    uint16_t threshold_lower, uint16_t threshold_upper,
    uint8_t *restrict output, double *restrict runtimes) {
    
    // Init the image size
    size_t image_size = width * height;

    // Use this to check the output of each API call
    cl_int status;

    // Allocate arrays for intermediate results
    int16_t *sobel_x = malloc(image_size * sizeof(int16_t));
    assert(sobel_x);

    int16_t *sobel_y = malloc(image_size * sizeof(int16_t));
    assert(sobel_y);

    uint8_t *phase = malloc(image_size * sizeof(uint8_t));
    assert(phase);

    uint16_t *magnitude = malloc(image_size * sizeof(uint16_t));
    assert(magnitude);

    cl_double times[14];
/* Write to the buffers */
    status = clEnqueueWriteBuffer(cmdQueue, bufInputImage, CL_TRUE,
        0, image_size * sizeof(uint8_t), input, 0, NULL, &bufferWriteInputImageEvent);
    // If something worng, report error
    if (status != CL_SUCCESS) {
        printf("OpenCL error: %s\n", clErrorString(status));
    }
    chk(status, "clEnqueueWriteBuffer");

    status = clEnqueueWriteBuffer(cmdQueue, bufSobel_x, CL_TRUE,
        0, image_size * sizeof(int16_t), sobel_x, 0, NULL, &bufferWriteSobelxEvent);
    // If something worng, report error
    if (status != CL_SUCCESS) {
        printf("OpenCL error: %s\n", clErrorString(status));
    }
    chk(status, "clEnqueueWriteBuffer");

    status = clEnqueueWriteBuffer(cmdQueue, bufSobel_y, CL_TRUE,
        0, image_size * sizeof(int16_t), sobel_y, 0, NULL, &bufferWriteSobelyEvent);
    // If something worng, report error
    if (status != CL_SUCCESS) {
        printf("OpenCL error: %s\n", clErrorString(status));
    }
    chk(status, "clEnqueueWriteBuffer");

    status = clEnqueueWriteBuffer(cmdQueue, bufPhase, CL_TRUE,
        0, image_size * sizeof(uint8_t), phase, 0, NULL, &bufferWritePhaseEvent);
    // If something worng, report error
    if (status != CL_SUCCESS) {
        printf("OpenCL error: %s\n", clErrorString(status));
    }
    chk(status, "clEnqueueWriteBuffer");

    status = clEnqueueWriteBuffer(cmdQueue, bufMagnitude, CL_TRUE,
        0, image_size * sizeof(uint16_t), magnitude, 0, NULL, &bufferWriteMagnitudeEvent);
    // If something worng, report error
    if (status != CL_SUCCESS) {
        printf("OpenCL error: %s\n", clErrorString(status));
    }
    chk(status, "clEnqueueWriteBuffer");
// /* Creation of the program */
//     // Create a program with source code
//     program = clCreateProgramWithSource(context, 1, &programSource, NULL, &status);
//     chk(status, "clCreateProgramWithSource");

//     // Build (compile) the program for the device
//     status = clBuildProgram(program, numDevices, devices, 
//         NULL, NULL, NULL);
//     size_t log_size;
//     // If something worng, report error
//     if (status != CL_SUCCESS) {
//         printf("OpenCL error: %s\n", clErrorString(status));
//     }

//     status = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
//     // If something worng, report error
//     if (status != CL_SUCCESS) {
//         printf("OpenCL error: %s\n", clErrorString(status));
//     }

//     char *log = (char *) malloc(log_size);

//     status = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
//     // If something worng, report error
//     if (status != CL_SUCCESS) {
//         printf("OpenCL error: %s\n", clErrorString(status));
//     }
//     printf("%s\n", "Build program log: \n");
//     printf("%s\n", log);
//     chk(status, "clBuildProgram");
/* Creation of the kernels */
    kernelSobel3x3 = clCreateKernel(program, "sobel3x3", &status);
    chk(status, "clCreateKernel");

    kernelPhaseAndMagnitude = clCreateKernel(program, "phaseAndMagnitude", &status);
    chk(status, "clCreateKernel");

    kernelNonMaxSuppression = clCreateKernel(program, "nonMaxSuppression", &status);
    chk(status, "clCreateKernel");

    // kernelEdgeTracing = clCreateKernel(program, "edgeTracing", &status);
    // chk(status, "clCreateKernel");
/* Setting the arguments of the kernelSobel3x3 */
    // Associate the input and output buffers with the kernel 
    status = clSetKernelArg(kernelSobel3x3, 0, sizeof(cl_mem), &bufInputImage);
    // If something worng, report error
    if (status != CL_SUCCESS) {
        printf("OpenCL error: %s\n", clErrorString(status));
    }
    chk(status, "clSetKernelArg");
    status = clSetKernelArg(kernelSobel3x3, 1, sizeof(cl_mem), &bufSobel_x);
    // If something worng, report error
    if (status != CL_SUCCESS) {
        printf("OpenCL error: %s\n", clErrorString(status));
    }
    chk(status, "clSetKernelArg");
    status = clSetKernelArg(kernelSobel3x3, 2, sizeof(cl_mem), &bufSobel_y);
    // If something worng, report error
    if (status != CL_SUCCESS) {
        printf("OpenCL error: %s\n", clErrorString(status));
    }
    chk(status, "clSetKernelArg");
    status = clSetKernelArg(kernelSobel3x3, 3, sizeof(cl_uint), &width);
    // If something worng, report error
    if (status != CL_SUCCESS) {
        printf("OpenCL error: %s\n", clErrorString(status));
    }
    chk(status, "clSetKernelArg");
    status = clSetKernelArg(kernelSobel3x3, 4, sizeof(cl_uint), &height);
    // If something worng, report error
    if (status != CL_SUCCESS) {
        printf("OpenCL error: %s\n", clErrorString(status));
    }
    chk(status, "clSetKernelArg");
/* Setting the arguments of the kernelPhaseAndMagnitude */
    // Associate the input and output buffers with the kernel 
    status = clSetKernelArg(kernelPhaseAndMagnitude, 0, sizeof(cl_mem), &bufSobel_x);
    // If something worng, report error
    if (status != CL_SUCCESS) {
        printf("OpenCL error: %s\n", clErrorString(status));
    }
    chk(status, "clSetKernelArg");
    status = clSetKernelArg(kernelPhaseAndMagnitude, 1, sizeof(cl_mem), &bufSobel_y);
    // If something worng, report error
    if (status != CL_SUCCESS) {
        printf("OpenCL error: %s\n", clErrorString(status));
    }
    chk(status, "clSetKernelArg");
    status = clSetKernelArg(kernelPhaseAndMagnitude, 2, sizeof(cl_uint), &width);
    // If something worng, report error
    if (status != CL_SUCCESS) {
        printf("OpenCL error: %s\n", clErrorString(status));
    }
    chk(status, "clSetKernelArg");
    status = clSetKernelArg(kernelPhaseAndMagnitude, 3, sizeof(cl_uint), &height);
    // If something worng, report error
    if (status != CL_SUCCESS) {
        printf("OpenCL error: %s\n", clErrorString(status));
    }
    chk(status, "clSetKernelArg");
    status = clSetKernelArg(kernelPhaseAndMagnitude, 4, sizeof(cl_mem), &bufPhase);
    // If something worng, report error
    if (status != CL_SUCCESS) {
        printf("OpenCL error: %s\n", clErrorString(status));
    }
    chk(status, "clSetKernelArg");
    status = clSetKernelArg(kernelPhaseAndMagnitude, 5, sizeof(cl_mem), &bufMagnitude);
    // If something worng, report error
    if (status != CL_SUCCESS) {
        printf("hh OpenCL error: %s\n", clErrorString(status));
    }
    chk(status, "clSetKernelArg");
/* Setting the arguments of the kernelNonMaxSuppression */
    // Associate the input and output buffers with the kernel 
    status = clSetKernelArg(kernelNonMaxSuppression, 0, sizeof(cl_mem), &bufMagnitude);
    // If something worng, report error
    if (status != CL_SUCCESS) {
        printf("hh OpenCL error: %s\n", clErrorString(status));
    }
    chk(status, "clSetKernelArg");
    status = clSetKernelArg(kernelNonMaxSuppression, 1, sizeof(cl_mem), &bufPhase);
    // If something worng, report error
    if (status != CL_SUCCESS) {
        printf("OpenCL error: %s\n", clErrorString(status));
    }
    chk(status, "clSetKernelArg");
    status = clSetKernelArg(kernelNonMaxSuppression, 2, sizeof(cl_uint), &width);
    // If something worng, report error
    if (status != CL_SUCCESS) {
        printf("OpenCL error: %s\n", clErrorString(status));
    }
    chk(status, "clSetKernelArg");
    status = clSetKernelArg(kernelNonMaxSuppression, 3, sizeof(cl_uint), &height);
    // If something worng, report error
    if (status != CL_SUCCESS) {
        printf("OpenCL error: %s\n", clErrorString(status));
    }
    chk(status, "clSetKernelArg");
    status = clSetKernelArg(kernelNonMaxSuppression, 4, sizeof(int16_t), &threshold_lower);
    // If something worng, report error
    if (status != CL_SUCCESS) {
        printf("OpenCL error: %s\n", clErrorString(status));
    }
    chk(status, "clSetKernelArg");
    status = clSetKernelArg(kernelNonMaxSuppression, 5, sizeof(uint16_t), &threshold_upper);
    // If something worng, report error
    if (status != CL_SUCCESS) {
        printf("OpenCL error: %s\n", clErrorString(status));
    }
    chk(status, "clSetKernelArg");
    status = clSetKernelArg(kernelNonMaxSuppression, 6, sizeof(cl_mem), &bufOutputImage);
    // If something worng, report error
    if (status != CL_SUCCESS) {
        printf("OpenCL error: %s\n", clErrorString(status));
    }
    chk(status, "clSetKernelArg");
/* Setting the arguments of the kernelEdgeTracing*/
    // status = clSetKernelArg(kernelEdgeTracing, 0, sizeof(cl_mem), &bufOutputImage);
    // // If something worng, report error
    // if (status != CL_SUCCESS) {
    //     printf("OpenCL error: %s\n", clErrorString(status));
    // }
    // chk(status, "clSetKernelArg");
    // status = clSetKernelArg(kernelEdgeTracing, 1, sizeof(size_t), &width);
    // // If something worng, report error
    // if (status != CL_SUCCESS) {
    //     printf("OpenCL error: %s\n", clErrorString(status));
    // }
    // chk(status, "clSetKernelArg");
    // status = clSetKernelArg(kernelEdgeTracing, 2, sizeof(size_t), &height);
    // // If something worng, report error
    // if (status != CL_SUCCESS) {
    //     printf("OpenCL error: %s\n", clErrorString(status));
    // }
    // chk(status, "clSetKernelArg");
/* Global work size */
    // Define an index space (global work size) of work 
    // items for execution. A workgroup size (local work size) 
    // is not required, but can be used.
    size_t globalWorkSize[2];   
 
    globalWorkSize[0] = width;
    globalWorkSize[1] = height;

    // size_t localWorkSize[1] = {width * height * sizeof(coord_t)};
/* Execution of the kernels */
    // Execute the kernel for execution
    status = clEnqueueNDRangeKernel(cmdQueue, kernelSobel3x3, 2, NULL, 
        globalWorkSize, NULL, 0, NULL, &kernelSobel3x3Event);
    // If something worng, report error
    if (status != CL_SUCCESS) {
        printf("OpenCL error: %s\n", clErrorString(status));
    }
    chk(status, "clEnqueueNDRange");
    // Execute the kernel for execution
    status = clEnqueueNDRangeKernel(cmdQueue, kernelPhaseAndMagnitude, 2, NULL, 
        globalWorkSize, NULL, 0, NULL, &kernelPhaseAndMagnitudeEvent);
    // If something worng, report error
    if (status != CL_SUCCESS) {
        printf("OpenCL error: %s\n", clErrorString(status));
    }
    chk(status, "clEnqueueNDRange");
    // Execute the kernel for execution
    status = clEnqueueNDRangeKernel(cmdQueue, kernelNonMaxSuppression, 2, NULL, 
        globalWorkSize, NULL, 0, NULL, &kernelNonMaxSuppressionEvent);
    // If something worng, report error
    if (status != CL_SUCCESS) {
        printf("OpenCL error: %s\n", clErrorString(status));
    }
    chk(status, "clEnqueueNDRange");
    // Execute the kernel for execution
    // status = clEnqueueNDRangeKernel(cmdQueue, kernelEdgeTracing, 2, NULL, 
    //     globalWorkSize, localWorkSize, 0, NULL, &kernelEdgeTracingEvent);
    // // If something worng, report error
    // if (status != CL_SUCCESS) {
    //     printf("OpenCL error: %s\n", clErrorString(status));
    // }
    // chk(status, "clEnqueueNDRange");
/* Read from buffers */
    // Read the device output buffer to the host output array
    clEnqueueReadBuffer(cmdQueue, bufSobel_x, CL_TRUE, 0, 
        image_size * sizeof(int16_t), sobel_x, 0, NULL, &bufferReadSobelxEvent);
    chk(status, "clEnqueueReadBuffer");

    clEnqueueReadBuffer(cmdQueue, bufSobel_y, CL_TRUE, 0, 
        image_size * sizeof(int16_t), sobel_y, 0, NULL, &bufferReadSobelyEvent);
    chk(status, "clEnqueueReadBuffer");

    // Read the device output buffer to the host output array
    clEnqueueReadBuffer(cmdQueue, bufMagnitude, CL_TRUE, 0, 
        image_size * sizeof(uint16_t), magnitude, 0, NULL, &bufferReadMagnitudeEvent);
    chk(status, "clEnqueueReadBuffer");

    // Read the device output buffer to the host output array
    clEnqueueReadBuffer(cmdQueue, bufPhase, CL_TRUE, 0, 
        image_size * sizeof(uint8_t), phase, 0, NULL, &bufferReadPhaseEvent);
    chk(status, "clEnqueueReadBuffer");

    // Read the device output buffer to the host output array
    clEnqueueReadBuffer(cmdQueue, bufOutputImage, CL_TRUE, 0, 
        image_size * sizeof(uint8_t), output, 0, NULL, &bufferReadOutputImageEvent);
    chk(status, "clEnqueueReadBuffer");

    cl_double start_edgeTracing = gettimemono_ns();
    edgeTracing(output, width, height);  // modifies output in-place
    cl_double end_edgeTracing = gettimemono_ns();

    // Get the execution times of buffer Write action 
    times[0] = (cl_double)getStartEndTime(bufferWriteInputImageEvent)*(cl_double)(1e-06);
    times[1] = (cl_double)getStartEndTime(bufferWriteSobelxEvent)*(cl_double)(1e-06);
    times[2] = (cl_double)getStartEndTime(bufferWriteSobelyEvent)*(cl_double)(1e-06);
    times[3] = (cl_double)getStartEndTime(bufferWritePhaseEvent)*(cl_double)(1e-06);
    times[4] = (cl_double)getStartEndTime(bufferWriteMagnitudeEvent)*(cl_double)(1e-06);
    times[5] = (cl_double)getStartEndTime(bufferReadOutputImageEvent)*(cl_double)(1e-06);
    times[6] = (cl_double)getStartEndTime(bufferReadSobelxEvent)*(cl_double)(1e-06);
    times[7] = (cl_double)getStartEndTime(bufferReadSobelyEvent)*(cl_double)(1e-06);
    times[8] = (cl_double)getStartEndTime(bufferReadPhaseEvent)*(cl_double)(1e-06);
    times[9] = (cl_double)getStartEndTime(bufferReadMagnitudeEvent)*(cl_double)(1e-06);
    // Get the execution times of kernels
    times[10] = (cl_double)getStartEndTime(kernelSobel3x3Event)*(cl_double)(1e-06);
    times[11] = (cl_double)getStartEndTime(kernelPhaseAndMagnitudeEvent)*(cl_double)(1e-06);
    times[12] = (cl_double)getStartEndTime(kernelNonMaxSuppressionEvent)*(cl_double)(1e-06);
    times[13] = (end_edgeTracing - end_edgeTracing)*(cl_double)(1e-06);
    //cl_double time_EdgeTracingEvent = (cl_double)getStartEndTime(kernelEdgeTracingEvent)*(cl_double)(1e-06);

    // Canny edge detection algorithm consists of the following functions:
    // times[0] = gettimemono_ns();
    //sobel3x3(input, width, height, sobel_x, sobel_y);

    // times[1] = gettimemono_ns();
    // phaseAndMagnitude(sobel_x, sobel_y, width, height, phase, magnitude);

    // times[2] = gettimemono_ns();
    //nonMaxSuppression(magnitude, phase, width, height, threshold_lower, threshold_upper, output);

    // times[4] = gettimemono_ns();
    // Release intermediate arrays

    free(sobel_x);
    free(sobel_y);
    free(phase);
    free(magnitude);

    for (int i = 0; i < 14; i++) {
        runtimes[i] = times[i];
    }
}

// Needed only in Part 2 for OpenCL initialization
void
init(
    size_t width, size_t height, uint16_t threshold_lower,
    uint16_t threshold_upper) {

    size_t image_size = width * height;  
     
    status = clGetPlatformIDs(0, NULL, &numPlatforms);
    // If something worng, report error
    if (status != CL_SUCCESS) {
        printf("OpenCL error: %s\n", clErrorString(status));
    }

    platforms = (cl_platform_id*)malloc(numPlatforms*sizeof(cl_platform_id));
 
    // Fill in the platforms
    status = clGetPlatformIDs(numPlatforms, platforms, NULL);
    // If something worng, report error
    if (status != CL_SUCCESS) {
        printf("OpenCL error: %s\n", clErrorString(status));
    }

    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, 
        NULL, &numDevices);
    // If something worng, report error
    if (status != CL_SUCCESS) {
        printf("OpenCL error: %s\n", clErrorString(status));
    }

    devices = (cl_device_id*)malloc(numDevices*sizeof(cl_device_id));

    // Fill in the devices 
    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL,        
        numDevices, devices, NULL);
    // If something worng, report error
    if (status != CL_SUCCESS) {
        printf("OpenCL error: %s\n", clErrorString(status));
    }

    context = clCreateContext(NULL, numDevices, devices, NULL, 
        NULL, &status);

    cmdQueue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &status);

    bufInputImage = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint8_t)*image_size,                       
       NULL, &status);

    bufSobel_x = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int16_t)*image_size,                        
        NULL, &status);

    bufSobel_y = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int16_t)*image_size,                        
        NULL, &status);

    bufPhase = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint8_t)*image_size,                        
        NULL, &status);

    bufMagnitude = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int16_t)*image_size,                        
        NULL, &status);

    bufOutputImage = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint8_t)*image_size,
        NULL, &status);

        /* Read the OpenCL source */
    programSource = read_source("canny.cl");
    
    /* Creation of the program */
    // Create a program with source code
    program = clCreateProgramWithSource(context, 1, &programSource, NULL, &status);
    chk(status, "clCreateProgramWithSource");

    // Build (compile) the program for the device
    status = clBuildProgram(program, numDevices, devices, 
        NULL, NULL, NULL);
    size_t log_size;
    // If something worng, report error
    if (status != CL_SUCCESS) {
        printf("OpenCL error: %s\n", clErrorString(status));
    }

    status = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    // If something worng, report error
    if (status != CL_SUCCESS) {
        printf("OpenCL error: %s\n", clErrorString(status));
    }

    char *log = (char *) malloc(log_size);

    status = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
    // If something worng, report error
    if (status != CL_SUCCESS) {
        printf("OpenCL error: %s\n", clErrorString(status));
    }
    printf("%s\n", "Build program log: \n");
    printf("%s\n", log);
    chk(status, "clBuildProgram");
}

void
destroy() {
/* Free OpenCL resources */
    // Release the kernels
    clReleaseKernel(kernelSobel3x3);
    clReleaseKernel(kernelPhaseAndMagnitude);
    clReleaseKernel(kernelNonMaxSuppression);
    // Release the Events
    clReleaseEvent(kernelSobel3x3Event);
    clReleaseEvent(kernelPhaseAndMagnitudeEvent);
    clReleaseEvent(kernelNonMaxSuppressionEvent);
    clReleaseEvent(bufferWriteInputImageEvent);
    clReleaseEvent(bufferWriteSobelxEvent);
    clReleaseEvent(bufferWriteSobelyEvent);
    clReleaseEvent(bufferWritePhaseEvent);
    clReleaseEvent(bufferWriteMagnitudeEvent);
    clReleaseEvent(bufferReadOutputImageEvent);
    clReleaseEvent(bufferReadSobelxEvent);
    clReleaseEvent(bufferReadSobelyEvent);
    clReleaseEvent(bufferReadPhaseEvent);
    clReleaseEvent(bufferReadMagnitudeEvent);
    // Release context, program and cmdQueue
    clReleaseContext(context);
    clReleaseProgram(program);
    clReleaseCommandQueue(cmdQueue);
    // Release buffers
    clReleaseMemObject(bufInputImage);
    clReleaseMemObject(bufSobel_x);
    clReleaseMemObject(bufSobel_y);
    clReleaseMemObject(bufPhase);
    clReleaseMemObject(bufMagnitude);
    clReleaseMemObject(bufOutputImage);
}

////////////////////////////////////////////////
// ¤¤ DO NOT EDIT ANYTHING AFTER THIS LINE ¤¤ //
////////////////////////////////////////////////

enum PROCESSING_MODE { DEFAULT, BIG_MODE, SMALL_MODE, VIDEO_MODE };
// ¤¤ DO NOT EDIT THIS FUNCTION ¤¤
int
main(int argc, char **argv) {
    enum PROCESSING_MODE mode = DEFAULT;
    if (argc > 1) {
        char *mode_c = argv[1];
        if (strlen(mode_c) == 2) {
            if (strncmp(mode_c, "-B", 2) == 0) {
                mode = BIG_MODE;
            } else if (strncmp(mode_c, "-b", 2) == 0) {
                mode = SMALL_MODE;
            } else if (strncmp(mode_c, "-v", 2) == 0) {
                mode = VIDEO_MODE;
            } else {
                printf(
                    "Invalid usage! Please set either -b, -B, -v or "
                    "nothing\n");
                return -1;
            }
        } else {
            printf("Invalid usage! Please set either -b, -B, -v nothing\n");
            return -1;
        }
    }
    int benchmarking_iterations = 1;
    if (argc > 2) {
        benchmarking_iterations = atoi(argv[2]);
    }

    char *input_image_path = "";
    char *output_image_path = "";
    uint16_t threshold_lower = 0;
    uint16_t threshold_upper = 0;
    switch (mode) {
        case BIG_MODE:
            input_image_path = "hameensilta.pgm";
            output_image_path = "hameensilta_output.pgm";
            // Arbitrarily selected to produce a nice-looking image
            // DO NOT CHANGE THESE WHEN BENCHMARKING
            threshold_lower = 120;
            threshold_upper = 300;
            printf(
                "Enabling %d benchmarking iterations with the large %s "
                "image\n",
                benchmarking_iterations, input_image_path);
            break;
        case SMALL_MODE:
            input_image_path = "x.pgm";
            output_image_path = "x_output.pgm";
            threshold_lower = 750;
            threshold_upper = 800;
            printf(
                "Enabling %d benchmarking iterations with the small %s "
                "image\n",
                benchmarking_iterations, input_image_path);
            break;
        case VIDEO_MODE:
            if (system("which ffmpeg > /dev/null 2>&1") ||
                system("which ffplay > /dev/null 2>&1")) {
                printf(
                    "Video mode is disabled because ffmpeg is not found\n");
                return -1;
            }
            benchmarking_iterations = 0;
            input_image_path = "people.mp4";
            threshold_lower = 120;
            threshold_upper = 300;
            printf(
                "Playing video %s with FFMPEG. Error check disabled.\n",
                input_image_path);
            break;
        case DEFAULT:
        default:
            input_image_path = "x.pgm";
            output_image_path = "x_output.pgm";
            // Carefully selected to produce a discontinuous edge without edge
            // tracing
            threshold_lower = 750;
            threshold_upper = 800;
            printf("Running with %s image\n", input_image_path);
            break;
    }

    uint8_t *input_image = NULL;
    size_t width = 0;
    size_t height = 0;
    if (mode == VIDEO_MODE) {
        width = 3840;
        height = 2160;
        init(width, height, threshold_lower, threshold_upper);

        uint8_t *output_image = malloc(width * height);
        assert(output_image);

        int count;
        uint16_t *frame = malloc(width * height * 3);
        assert(frame);
        char pipein_cmd[1024];
        snprintf(
            pipein_cmd, 1024,
            "ffmpeg -i %s -f image2pipe -vcodec rawvideo -an -s %zux%zu "
            "-pix_fmt gray - 2> /dev/null",
            input_image_path, width, height);
        FILE *pipein = popen(pipein_cmd, "r");
        char pipeout_cmd[1024];
        snprintf(
            pipeout_cmd, 1024,
            "ffplay -f rawvideo -pixel_format gray -video_size %zux%zu "
            "-an - 2> /dev/null",
            width, height);
        FILE *pipeout = popen(pipeout_cmd, "w");
        double runtimes[4];
        while (1) {
            count = fread(frame, 1, height * width, pipein);
            if (count != height * width) break;

            cannyEdgeDetection(
                frame, width, height, threshold_lower, threshold_upper,
                output_image, runtimes);

            double total_time =
                runtimes[0] + runtimes[1] + runtimes[2] + runtimes[3];
            printf("FPS: %0.1f\n", 1000 / total_time);
            fwrite(output_image, 1, height * width, pipeout);
        }
        fflush(pipein);
        pclose(pipein);
        fflush(pipeout);
        pclose(pipeout);
    } else {
        if ((input_image = read_pgm(input_image_path, &width, &height))) {
            printf(
                "Input image read succesfully. Size %zux%zu\n", width,
                height);
        } else {
            printf("Read failed\n");
            return -1;
        }
        init(width, height, threshold_lower, threshold_upper);

        uint8_t *output_image = malloc(width * height);
        assert(output_image);

        int all_the_runs_were_succesful = 1;
        cl_double avg_runtimes[14] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        double avg_total = 0.0;

        for (int iter = 0; iter < benchmarking_iterations; iter++) {
            double iter_runtimes[14];

            cannyEdgeDetection(input_image, width, height, threshold_lower, threshold_upper,
            output_image, iter_runtimes);

            for (int n = 0; n < 14; n++) {
                avg_runtimes[n] += iter_runtimes[n] / benchmarking_iterations;
                avg_total += iter_runtimes[n] / benchmarking_iterations;
            }

            uint8_t *output_image_ref = malloc(width * height);
            assert(output_image_ref);
            cannyEdgeDetection_ref(
                input_image, width, height, threshold_lower, threshold_upper,
                output_image_ref);

            uint8_t *fused_comparison = malloc(width * height);
            assert(fused_comparison);
            int failed = validate_result(
                output_image, output_image_ref, width, height,
                fused_comparison);
            if (failed) {
                all_the_runs_were_succesful = 0;
                printf(
                    "Error checking failed for benchmark iteration %d!\n"
                    "Writing your output to %s. The image that should've "
                    "been generated is written to ref.pgm\n"
                    "Generating fused.pgm for debugging purpose. Light-grey "
                    "pixels should've been white and "
                    "dark-grey pixels black. Corrupted pixels are colored "
                    "middle-grey\n",
                    iter, output_image_path);

                write_pgm("ref.pgm", output_image_ref, width, height);
                write_pgm("fused.pgm", fused_comparison, width, height);
            }
        }

        printf("Execution time of buffer Write of InputImage        : %0.6f ms\n\r", avg_runtimes[0]);
        printf("Execution time of buffer Write of Sobelx            : %0.6f ms\n\r", avg_runtimes[1]);
        printf("Execution time of buffer Write of Sobely            : %0.6f ms\n\r", avg_runtimes[2]);
        printf("Execution time of buffer Write of Phase             : %0.6f ms\n\r", avg_runtimes[3]);
        printf("Execution time of buffer Write of Magnitude         : %0.6f ms\n\r", avg_runtimes[4]);

        printf("Execution time of buffer Read of OutputImageEvent   : %0.6f ms\n\r", avg_runtimes[5]);
        printf("Execution time of buffer Read of SobelxEvent        : %0.6f ms\n\r", avg_runtimes[6]);
        printf("Execution time of buffer Read of SobelyEvent        : %0.6f ms\n\r", avg_runtimes[7]);
        printf("Execution time of buffer Read of PhaseEvent         : %0.6f ms\n\r", avg_runtimes[8]);
        printf("Execution time of buffer Read of MagnitudeEvent     : %0.6f ms\n\r", avg_runtimes[9]);

        printf("Execution time of Sobel3x3 kernel                   : %0.6f ms\n\r", avg_runtimes[10]);
        printf("Execution time of PhaseAndMagnitude kernel          : %0.6f ms\n\r", avg_runtimes[11]);
        printf("Execution time of NonMaxSuppression kernel          : %0.6f ms\n\r", avg_runtimes[12]);
        // printf("Execution time of edgeTracing kernel                : %0.6f ms\n\r", avg_runtimes[14]);
        // printf("Execution time of edgeTracing function              : %0.16f ms\n\r", avg_runtimes[13]);
        printf("Total time                                          : %0.3f ms\n", avg_total);

        // printf("Execution time of NonMaxSuppression: %f ms\n\r", time_EdgeTracingEvent);

        // printf("Sobel3x3 time          : %0.3f ms\n", avg_runtimes[0]);
        // printf("phaseAndMagnitude time : %0.3f ms\n", avg_runtimes[1]);
        // printf("nonMaxSuppression time : %0.3f ms\n", avg_runtimes[2]);
        // printf("edgeTracing time       : %0.3f ms\n", avg_runtimes[3]);
        // printf("Total time             : %0.3f ms\n", avg_total);
        write_pgm(output_image_path, output_image, width, height);
        printf("Wrote output to %s\n", output_image_path);
        if (all_the_runs_were_succesful) {
            printf("Error checks passed!\n");
        } else {
            printf("There were failing runs\n");
        }
    }
    destroy();
    return 0;
}