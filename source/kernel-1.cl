#define DELTA_TIME 1.f
#define DISTANCE_TO_NEAREST_STAR 50


__kernel void calc_pos(__global float3 *gpu_pos, __global float3 *gpu_speed)
{
	const int i = get_global_id(0);
	gpu_pos[i].s0 += (gpu_speed[i].s0 * DELTA_TIME) / DISTANCE_TO_NEAREST_STAR;
    gpu_pos[i].s1 += (gpu_speed[i].s1 * DELTA_TIME) / DISTANCE_TO_NEAREST_STAR;
    gpu_pos[i].s2 += (gpu_speed[i].s2 * DELTA_TIME) / DISTANCE_TO_NEAREST_STAR;

}
