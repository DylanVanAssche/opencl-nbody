#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "time_utils.h"
#include "ocl_utils.h"
#include "renderer.h"
#include "math.h"

#define DIMENSION_1D 1
#define DIMENSION_2D 2

#ifdef __APPLE__
# include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

void usage(char* prog_name) //gewoon een printje
{
    printf("Usage: %s <number of bodies>\n", prog_name);
}

//Functie om een buffer op de GPU te maken, geeft buffer terug
cl_mem makeBufferOnGPU(int length)
{
	// Error code
	cl_int error; //errorcode maken

	// Maak buffer die leeg is
    cl_mem buffer = clCreateBuffer(
			g_context,
            CL_MEM_READ_WRITE,
            sizeof(cl_float3) * length,
			NULL,
			&error
		);

	// Set error location code
    ocl_err(error);

	// Process and check for errros
    ocl_err(clFinish(g_command_queue));
    return buffer;
}

//functie om dingen te doen
void simulate_gravity(cl_float3* host_pos, cl_float3* host_speed, cl_float3* host_dist, cl_mem gpu_pos, cl_mem gpu_speed, cl_mem gpu_dist, cl_kernel kernel_pos,cl_kernel kernel_speed, int length,cl_int error_pos, cl_int error_speed)
{

//constanten mogen er eigenlijk uit, zijn verhuisd naar de kernel
    const float delta_time = 1.f;
    // const float grav_constant = 6.67428e-11;
    const float grav_constant = 1;
    const float mass_of_sun = 2;
    const float mass_grav = grav_constant * mass_of_sun * mass_of_sun;
    const float distance_to_nearest_star = 50;

//vorige code, is vervangen door kernel
/* FOR LUS 1
   for (int i = 0; i < length; ++i)
    {
        for (int j = 0; j < length; ++j)
        {

            if (i == j)
                continue;

            cl_float3 pos_a = host_pos[i];
            cl_float3 pos_b = host_pos[j];

            float dist_x = (pos_a.s[0] - pos_b.s[0]) * distance_to_nearest_star;
            float dist_y = (pos_a.s[1] - pos_b.s[1]) * distance_to_nearest_star;
            float dist_z = (pos_a.s[2] - pos_b.s[2]) * distance_to_nearest_star;


            float distance = sqrt(
                    dist_x * dist_x +
                    dist_y * dist_y +
                    dist_z * dist_z);

            float force_x = -mass_grav * dist_x / (distance * distance * distance);
            float force_y = -mass_grav * dist_y / (distance * distance * distance);
            float force_z = -mass_grav * dist_z / (distance * distance * distance);

            float acc_x = force_x / mass_of_sun;
            float acc_y = force_y / mass_of_sun;
            float acc_z = force_z / mass_of_sun;

            host_speed[i].s[0] += acc_x * delta_time;
            host_speed[i].s[1] += acc_y * delta_time;
            host_speed[i].s[2] += acc_z * delta_time;
        }

    }
*/

	//kernel_speed, die in main gemaakt is, oproepen
	ocl_err(error_speed);

	// Set kernel arguments: 2 pointers naar buffers, 1 lengte
	int arg_num = 0;
	ocl_err(clSetKernelArg(kernel_speed, arg_num++, sizeof(cl_mem), &gpu_pos));
	ocl_err(clSetKernelArg(kernel_speed, arg_num++, sizeof(cl_mem), &gpu_speed));
	ocl_err(clSetKernelArg(kernel_speed, arg_num++, sizeof(cl_mem), &gpu_dist));

    // Kopieer buffers
	ocl_err(clEnqueueWriteBuffer(g_command_queue, gpu_pos, CL_TRUE, 0, sizeof(cl_float3) * length,host_pos, 0, NULL, NULL));
	ocl_err(clEnqueueWriteBuffer(g_command_queue, gpu_speed, CL_TRUE, 0, sizeof(cl_float3) * length,host_speed, 0, NULL, NULL));
	ocl_err(clEnqueueWriteBuffer(g_command_queue, gpu_dist, CL_TRUE, 0, sizeof(cl_float3) * length,host_dist, 0, NULL, NULL));

	// Call kernel 2D
	size_t global_work_sizes_speed[] = {length,length};
	time_measure_start("computation");
	ocl_err(clEnqueueNDRangeKernel(g_command_queue, kernel_speed, DIMENSION_2D, NULL, global_work_sizes_speed, NULL, 0, NULL, NULL));
	ocl_err(clFinish(g_command_queue));
	time_measure_stop_and_print("computation");

	// Read result
	time_measure_start("data_transfer");
	ocl_err(clEnqueueReadBuffer(g_command_queue, gpu_speed, CL_TRUE, 0, sizeof(cl_float3) * length, host_speed, 0, NULL, NULL));
	ocl_err(clEnqueueReadBuffer(g_command_queue, gpu_dist, CL_TRUE, 0, sizeof(cl_float3) * length, host_dist, 0, NULL, NULL));
	time_measure_stop_and_print("data_transfer");


/*  FOR LUS 2
    for (int i = 0; i < length; ++i)
    {
        host_pos[i].s[0] += (host_speed[i].s[0] * delta_time) / distance_to_nearest_star;
        host_pos[i].s[1] += (host_speed[i].s[1] * delta_time) / distance_to_nearest_star;
        host_pos[i].s[2] += (host_speed[i].s[2] * delta_time) / distance_to_nearest_star;
    }
	//code om bovenstaande for-lus te parallelliseren
*/
	//kernel_pos, die in main gemaakt is, oproepen
    // Create kernel
    ocl_err(error_pos);

    // Set kernel arguments: 2 pointers naar buffers, 1 lengte
    arg_num = 0;
    ocl_err(clSetKernelArg(kernel_pos, arg_num++, sizeof(cl_mem), &gpu_pos));
	ocl_err(clSetKernelArg(kernel_pos, arg_num++, sizeof(cl_mem), &gpu_speed));

    // Kopieer buffers
    ocl_err(clEnqueueWriteBuffer(g_command_queue, gpu_pos, CL_TRUE, 0, sizeof(cl_float3) * length,host_pos, 0, NULL, NULL));
    ocl_err(clEnqueueWriteBuffer(g_command_queue, gpu_speed, CL_TRUE, 0, sizeof(cl_float3) * length,host_speed, 0, NULL, NULL));

    // Call kernel 1D
    size_t global_work_sizes_pos[] = {length};
    time_measure_start("computation");
    ocl_err(clEnqueueNDRangeKernel(g_command_queue, kernel_pos, DIMENSION_1D, NULL, global_work_sizes_pos, NULL, 0, NULL, NULL));
    ocl_err(clFinish(g_command_queue));
    time_measure_stop_and_print("computation");

    // Read result
    time_measure_start("data_transfer");
    ocl_err(clEnqueueReadBuffer(g_command_queue, gpu_pos, CL_TRUE, 0, sizeof(cl_float3) * length, host_pos, 0, NULL, NULL));
    time_measure_stop_and_print("data_transfer");
}

int main(int argc, char** argv) {
    if (argc < 2)
    {
        usage(argv[0]);
        return 1;
    }
    int length = atoi(argv[1]);

	cl_platform_id platform = ocl_select_platform(); // Selecteer OpenCL platform + device ID
    cl_device_id device = ocl_select_device(platform);
    init_ocl(device); // Maak device ready
    create_program("kernel.cl", "");

    // Maak kernel en vang eventuele errors op
	cl_int error_pos,error_speed;
    cl_kernel kernel_pos = clCreateKernel(g_program, "calc_pos_a", &error_pos);
    cl_kernel kernel_speed  = clCreateKernel(g_program, "calc_speed_a", &error_speed);
    init_gl();

    // Alloceer buffers op host
    cl_float3 *host_pos = malloc(sizeof(cl_float3) * length);
    cl_float3 *host_speed = malloc(sizeof(cl_float3) * length);
    cl_float3 *host_dist = malloc(sizeof(cl_float3) * length);

    // Alloceer buffers op GPU
	cl_mem gpu_pos = makeBufferOnGPU(length);
	cl_mem gpu_speed = makeBufferOnGPU(length);
	cl_mem gpu_dist = makeBufferOnGPU(length);

    // Init buffers op host
    for (int i = 0; i < length; ++i)
    {
        float offset;

        if (rand() < RAND_MAX / 2)
            offset = -5.f;
        else
            offset = 5.f;


        host_pos[i].s[0] = ((float)rand() / (float)RAND_MAX) * 2.f - 1.f + offset;
        host_pos[i].s[1] = ((float)rand() / (float)RAND_MAX) * 2.f - 1.f;
        host_pos[i].s[2] = ((float)rand() / (float)RAND_MAX) * 2.f - 1.f;

        host_speed[i].s[0] = 0.f;
        host_speed[i].s[1] = 0.f;
        host_speed[i].s[2] = 0.f;

		host_dist[i].s[0] = 0.f;
        host_dist[i].s[1] = 0.f;
        host_dist[i].s[2] = 99999999.f; // init op een groot getal anders is 0 altijd de min afstand
    }

    // Run until exit
    int is_done = 0;
    while (!is_done)
    {
        is_done = render_point_cloud(host_pos, length); // GUI
        time_measure_start("simulation step");
        simulate_gravity(host_pos, host_speed, host_dist, gpu_pos, gpu_speed, gpu_dist, kernel_pos, kernel_speed, length, error_pos, error_speed);
        // DEBUG
        for(int k=0; k < length; k++) {
			printf("gem= %f, max= %f, min= %f\n", host_dist[k].s[0]/length, host_dist[k].s[1], host_dist[k].s[2]);
		}
        time_measure_stop_and_print("simulation step");
    }
}
