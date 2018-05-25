/*
Simon Vandevelde, Dylan Van Assche
2de for lus paralleliseren zonder atomische operaties
*/
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "time_utils.h"
#include "ocl_utils.h"
#include "renderer.h"
#include "math.h"

#define DIMENSION 1

#ifdef __APPLE__
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif

void usage(char* prog_name)
{
    printf("Usage: %s <number of bodies>\n", prog_name);
}

// Buffer maken op de GPU om data in op te slaan
cl_mem makeBufferOnGPU(int length)
{
	// Error code
	cl_int error;

	// Make 2D buffer die leeg is
    cl_mem buffer = clCreateBuffer(
			g_context,
            CL_MEM_READ_WRITE,	//we moeten zowel kunnen lezen als schrijven in de buffer
            sizeof(cl_float3) * length,
			NULL,
			&error
		);

//
//cl_mem clCreateBuffer ( 	cl_context context,
//  	cl_mem_flags flags,
// 	size_t size,
//  	void *host_ptr, 	//eventuele pointer naar reeds bestaande buffer (moet even groot of groter zijn dan size)
//  	cl_int *errcode_ret)
//
//
	// Set error location code
    ocl_err(error);

	// Process and check for errros
    ocl_err(clFinish(g_command_queue));
    return buffer;
}

//met tweede for-lus geparallelliseerd
void simulate_gravity(cl_float3* host_pos, cl_float3* host_speed, cl_mem gpu_pos, cl_mem gpu_speed,cl_kernel kernel, int length,cl_int error)
{
    // FOR LUS 1
    const float delta_time = 1.f;
    // const float grav_constant = 6.67428e-11;
    const float grav_constant = 1;
    const float mass_of_sun = 2;
    const float mass_grav = grav_constant * mass_of_sun * mass_of_sun;
    const float distance_to_nearest_star = 50;

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
	    // Create kernel
	    ocl_err(error);


    /*
    FOR LUS 2

    for (int i = 0; i < length; ++i)
    {
        host_pos[i].s[0] += (host_speed[i].s[0] * delta_time) / distance_to_nearest_star;
        host_pos[i].s[1] += (host_speed[i].s[1] * delta_time) / distance_to_nearest_star;
        host_pos[i].s[2] += (host_speed[i].s[2] * delta_time) / distance_to_nearest_star;
    }*/
	//code om bovenstaande for-lus te parallelliseren

    // Create kernel
    ocl_err(error);

    // Set kernel arguments: 2 pointers naar buffers, 1 lengte
    int arg_num = 0;
    ocl_err(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &gpu_pos));
	ocl_err(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &gpu_speed));


    // Kopieer buffers
    ocl_err(clEnqueueWriteBuffer(g_command_queue, gpu_pos, CL_TRUE, 0, sizeof(cl_float3) * length,host_pos, 0, NULL, NULL));
    ocl_err(clEnqueueWriteBuffer(g_command_queue, gpu_speed, CL_TRUE, 0, sizeof(cl_float3) * length,host_speed, 0, NULL, NULL)); // enkel pos is bijgewerkt

//
// cl_int clEnqueueWriteBuffer ( 	cl_command_queue command_queue,
//  	cl_mem buffer,
//  	cl_bool blocking_write,		//wachten op volledige transfer
//  	size_t offset,			//bufferoffset
//  	size_t cb,			//buffersize
//  	const void *ptr,		//pointer naar de te overschrijven data die zich op host bevindt
//  	cl_uint num_events_in_wait_list,//wachtrij aanmaken
// 	const cl_event *event_wait_list,//bv wachten tot andere operatie klaar is
//  	cl_event *event)		//om de status van een event op te vragen
//

    // Call kernel 1D
    size_t global_work_sizes[] = {length};
    time_measure_start("computation");
    ocl_err(clEnqueueNDRangeKernel(g_command_queue, kernel, DIMENSION, NULL, global_work_sizes, NULL, 0, NULL, NULL)); // Stuur opdracht naar GPU

// cl_int clEnqueueNDRangeKernel ( 	cl_command_queue command_queue,
//  	cl_kernel kernel,		//kernel die we willen runnen
//  	cl_uint work_dim,		//dimensie van work-items
//  	const size_t *global_work_offset, //moet altijd null zijn volgens documentatie
//  	const size_t *global_work_size,	//pointer naar array die het aantal global_work-items beschrijven
//  	const size_t *local_work_size, //pointer naar array die het aantal lobal_work-items beschrijven
//  	cl_uint num_events_in_wait_list, //wachtrij 
//  	const cl_event *event_wait_list,
// 	cl_event *event)		//event om te pollen

    ocl_err(clFinish(g_command_queue)); // wacht tot GPU klaar is
	printf("c:");
    time_measure_stop_and_print("computation");

    // Read result
    time_measure_start("data_transfer");
    ocl_err(clEnqueueReadBuffer(g_command_queue, gpu_pos, CL_TRUE, 0, sizeof(cl_float3) * length, host_pos, 0, NULL, NULL)); // Lees GPU data naar host
//werkt zoals de writebuffer, maar dan in de andere richting
    printf("d:");
    time_measure_stop_and_print("data_transfer");
}

int main(int argc, char** argv) {
    if (argc < 2)
    {
        usage(argv[0]);
        return 1;
    }
    int length = atoi(argv[1]);

	cl_platform_id platform = ocl_select_platform(); // Selecteer OpenCL platform en device ID
    cl_device_id device = ocl_select_device(platform);
    init_ocl(device); // maak device ready voor OpenCL
    create_program("kernel.cl", "");

	cl_int error; // Maak een kernel en catch eventuele errors
    cl_kernel kernel = clCreateKernel(g_program, "calc_pos", &error);
    init_gl();

    // Alloceer buffers op host
    cl_float3 *host_pos = malloc(sizeof(cl_float3) * length);
    cl_float3 *host_speed = malloc(sizeof(cl_float3) * length);

    // Alloceer buffers op GPU
	cl_mem gpu_pos = makeBufferOnGPU(length);
	cl_mem gpu_speed = makeBufferOnGPU(length);

    // Init host buffers op default values
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
    }

    // Blijf uitvoeren tot exit
    int is_done = 0;
    while (!is_done)
    {
        is_done = render_point_cloud(host_pos, length); // GUI
        time_measure_start("simulation step");
        simulate_gravity(host_pos, host_speed, gpu_pos, gpu_speed, kernel, length, error); // berekenen snelheid, positie, ... van elk lichaam
	    printf("s:");
        time_measure_stop_and_print("simulation step");
    }
}
