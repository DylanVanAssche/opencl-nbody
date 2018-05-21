#define DELTA_TIME 1.f
#define DISTANCE_TO_NEAREST_STAR 50
#define GRAV_CONSTANT 1
#define MASS_OF_SUN 2

//definieren van union om atomische operaties mogelijk te maken
typedef union
{
    float3 vec;	//float3 zoals we hem kennen en liefhebben
    float arr[3]; //'opgesplitse' float3 om hem atomisch te kunnen gebruiken
} float3_; //is geen verschil in geheugen tov float3! 
	//daarom kunnen we perfect nog float3 meegeven vanuit host

//functie om atomisch dingen op te tellen, zonder race-condities //is magisch niet kennen
inline void AtomicAdd(volatile __global float *source, const float operand) {
	//definieren twee unions, om te gebruiken als variabele en te vergelijken
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal,newVal.intVal) != prevVal.intVal);
}


//***************kernelcode voor niet-atomische posberekening******************************
__kernel void calc_pos(__global float3 *gpu_pos, __global float3 *gpu_speed)
{
	const int i = get_global_id(0); //neemt de global id van de werkgroep, vervangt teller 'i' van for-lus
	gpu_pos[i].s0 += (gpu_speed[i].s0 * DELTA_TIME) / DISTANCE_TO_NEAREST_STAR;
        gpu_pos[i].s1 += (gpu_speed[i].s1 * DELTA_TIME) / DISTANCE_TO_NEAREST_STAR;
        gpu_pos[i].s2 += (gpu_speed[i].s2 * DELTA_TIME) / DISTANCE_TO_NEAREST_STAR;


}


//***************kernelcode voor atomische posberekening******************************
__kernel void calc_pos_a(__global float3_ *gpu_pos, __global float3 *gpu_speed)
{
	const int i = get_global_id(0); //zie boven

	AtomicAdd(&gpu_pos[i].arr[0], ((float)(gpu_speed[i].s0 * DELTA_TIME)/DISTANCE_TO_NEAREST_STAR)); //gebruiken van atomicadd om atomisch op te tellen
	AtomicAdd(&gpu_pos[i].arr[1], ((float)(gpu_speed[i].s1 * DELTA_TIME)/DISTANCE_TO_NEAREST_STAR));
	AtomicAdd(&gpu_pos[i].arr[2], ((float)(gpu_speed[i].s2 * DELTA_TIME)/DISTANCE_TO_NEAREST_STAR));

}
//***************kernelcode voor niet-atomische speedberekening******************************
__kernel void calc_speed(__global float3 *gpu_pos, __global float3 *gpu_speed)
{
	const int i = get_global_id(0); //in twee dimensies, dus twee teller's nodig
	const int j = get_global_id(1); //global_id's vervangen 'i' en 'j'
	const int mass_grav = GRAV_CONSTANT*MASS_OF_SUN*MASS_OF_SUN; //allemaal code uit voorbeeld, niet belangrijk

	if (i == j)
	{
            return; //continue vervangen door return
	}
        float3 pos_a = gpu_pos[i];
        float3 pos_b = gpu_pos[j];

        float dist_x = (pos_a.s0 - pos_b.s0) * DISTANCE_TO_NEAREST_STAR;
        float dist_y = (pos_a.s1 - pos_b.s1) * DISTANCE_TO_NEAREST_STAR;
        float dist_z = (pos_a.s2 - pos_b.s2) * DISTANCE_TO_NEAREST_STAR;


        float distance = sqrt(
             dist_x * dist_x +
             dist_y * dist_y +
             dist_z * dist_z);

        float force_x = -mass_grav * dist_x / (distance * distance * distance);
        float force_y = -mass_grav * dist_y / (distance * distance * distance);
        float force_z = -mass_grav * dist_z / (distance * distance * distance);

        float acc_x = force_x / MASS_OF_SUN;
        float acc_y = force_y / MASS_OF_SUN;
        float acc_z = force_z / MASS_OF_SUN;

        gpu_speed[i].s0 += acc_x * DELTA_TIME;
        gpu_speed[i].s1 += acc_y * DELTA_TIME;
        gpu_speed[i].s2 += acc_z * DELTA_TIME;


}
//***************kernelcode voor atomische speedberekening******************************
__kernel void calc_speed_a(__global float3 *gpu_pos, __global float3_ *gpu_speed) //meegeven float3_ ipv float3! Geen conversie nodig hiervoor, zie union boven
{
	const int i = get_global_id(0); //zie boven
	const int j = get_global_id(1);
	const int mass_grav = GRAV_CONSTANT*MASS_OF_SUN*MASS_OF_SUN;

	if (i == j)
	{
            return; //zie boven
	}
        float3 pos_a = gpu_pos[i];
        float3 pos_b = gpu_pos[j];

        float dist_x = (pos_a.s0 - pos_b.s0) * DISTANCE_TO_NEAREST_STAR;
        float dist_y = (pos_a.s1 - pos_b.s1) * DISTANCE_TO_NEAREST_STAR;
        float dist_z = (pos_a.s2 - pos_b.s2) * DISTANCE_TO_NEAREST_STAR;


        float distance = sqrt(
             dist_x * dist_x +
             dist_y * dist_y +
             dist_z * dist_z);

        float force_x = -mass_grav * dist_x / (distance * distance * distance);
        float force_y = -mass_grav * dist_y / (distance * distance * distance);
        float force_z = -mass_grav * dist_z / (distance * distance * distance);

        float acc_x = force_x / MASS_OF_SUN;
        float acc_y = force_y / MASS_OF_SUN;
        float acc_z = force_z / MASS_OF_SUN;

	AtomicAdd(&gpu_speed[i].arr[0], (float)acc_x * DELTA_TIME); //atomische toevoeging
    	AtomicAdd(&gpu_speed[i].arr[1], (float)acc_y * DELTA_TIME);
    	AtomicAdd(&gpu_speed[i].arr[2], (float)acc_z * DELTA_TIME);



}
