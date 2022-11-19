#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <math.h> 
#include <thrust/shuffle.h>
#include <thrust/random.h>

#define PI 3.14159265
#define beeMax 10000000
#define threadsPerBlock 256

using namespace std;
using namespace thrust;

struct plot
{
    unsigned long long total_flowers;
    unsigned long long pollinated_flowers = 0;
    unsigned long long pollinated_last_year = 0;
    unsigned long long pollinated_today = 0;
    unsigned long long flowers_available = 0;
};

struct hive
{
    int total_nutrition;
    int should_bee_die = 0;
    int should_brood_die = 0;
    int total_bees = 0;
    int total_nurse_bees = 0;
    int total_worker_bees = 0;
    int total_broods = 0;
    int unattained_broods = 0;
    int foraged_increment = 0;
    int nutrition_deficit = 0;
    int eggLaid = 0;
};

struct bee
{
    hive* hive_ptr;
    int age;
    float nutrition; //decay by 0.01 per day, dies after negative -1
    int type; //0 = worker, 1 = drone

    __device__ __host__ bee() {}

    __device__ __host__ bee(hive *h, int a, float n, int t)
    {
        hive_ptr = h;
        age = a;
        nutrition = n;
        type = t;
    }
};

__host__ __device__ bool operator<(const bee &a, const bee &b) { return (a.age > b.age); };

struct brood
{
    int age;
    hive* hive_ptr;

    __device__ __host__ brood() {}

    __device__ __host__ brood(hive *h, int a) 
    {
        hive_ptr = h;
        age = a;
    }
};

int bee_cnt = 0;
device_vector<plot> d_plots(18*18);//18 km x 18 km
device_vector<bee> d_bees(beeMax);
device_vector<brood> d_broods(beeMax);

int hive_cnt = 1;
device_vector<hive> d_hives(1000);

//statics
__device__ int broodConsumption;
__device__ int beeConsumption;
double brood_mature_death_rate = 0.02;
__device__ double egg_laying_rate_multiplier = 1;
__device__ int egg_laying_rate_base = 100;
__device__ double starvation_death_winter_multiplier = 1.5;
__device__ double starvation_death_winter_divisor = 10;
__device__ double starvation_deficit_power = 0.5;
__device__ double starvation_deficit_multiplier = 0;
//
int bc = 2;
int pc = 6;

__device__ float total = 0;

__device__ double germination_rate = 0.3;

__global__ void plotPollinationUpdate(plot *p, int plot_cnt, hive *h, double efficiency, int day)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < plot_cnt)
    {
        if(day >= 50 && day <= 100)
        {
            unsigned long long blossomedFlowers = (p[index].pollinated_last_year*35) * germination_rate;
            p[index].total_flowers += blossomedFlowers/(101-day);
        }

        if(p[index].pollinated_today > p[index].flowers_available)
        {
            //this will be replaced with something that addresses the number of pollination per plot by any bee at a given time soon
            //most likely another set of arrays per plot where it stores the amount of pollination done by any bee given i from a hive.
            h[index].total_nutrition -= (int)efficiency * (p[index].pollinated_flowers - p[index].total_flowers)/111;
            //I see
            p[index].pollinated_today = p[index].flowers_available;
        }

        p[index].pollinated_flowers += p[index].pollinated_today;

        if(day >= 320 && day <= 364)
        {
            p[index].total_flowers -= p[index].total_flowers/(365 - day); // p[index].backup_total * (1.0-((day-240.0)/30.0));
            p[index].pollinated_last_year = p[index].pollinated_flowers;
        }
        if(day == 364)
        {
            p[index].pollinated_flowers = 0;
            p[index].total_flowers = 0;
        }
        
        p[index].flowers_available = p[index].total_flowers - p[index].pollinated_flowers;
        if(p[index].total_flowers < p[index].pollinated_flowers) p[index].flowers_available = 0;
    }
}

__global__ void broodUpdate(brood *b, int brood_cnt)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < brood_cnt)
    {
        atomicAdd(&(*(b[index].hive_ptr)).total_broods, 1);//hmm consumes 1 only, wtf man
        b[index].age += 1;
        if(b[index].age >= 3 && b[index].age <= 8)
        {
            //atomicAdd(&(*(b[index].hive_ptr)).total_nutrition, -6);
            atomicAdd(&(*(b[index].hive_ptr)).total_nutrition, -broodConsumption);
        }
    }
}

__device__ double eggLayingRate(double day)
{
    double rate = 0.0000000001 * pow(day, 5) - 0.0000000823  * pow(day, 4) + 0.0000143494  * pow(day, 3) + 0.0001276996  * pow(day, 2) - 0.0670571925 * day + 0.5824552258;
    if(rate < 0 || day < 50) rate = 0;
    //return rate * 30 + 100;
    return (rate * 200 + 100) * egg_laying_rate_multiplier;
    //return (rate * 200 + 100)*egg_laying_rate_multiplier; //with a multiplier of 200, you will have 2000-70000 population
    //return (cos((day/365 - 0.5) * PI * 2 ) + 1) * 1000 + 500;
}

__device__ double eggLayingRate2(double day)
{
    double rate = - 0.0000000174 * pow(day, 5) + 0.0000160050 * pow(day, 4) - 0.0049974512 * pow(day, 3) + 0.5490013102 * pow(day, 2) - 5.2106195939 * day + 54.5018666934;
    if(rate <= 0)
    {
        rate = 100;
    }
    return rate;
}

__global__ void hiveUpdate(hive *h, int hive_cnt, int day, plot *p)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < hive_cnt)
    {
        int remainingDays = 0;

        if(day < 150)
        {
            remainingDays = 150 - day;
        }
        else if(day > 280)
        {
            remainingDays = 150 + (365 - day);
        }

        int totalNutritionRequirements = (int)((h[index].total_bees*beeConsumption + h[index].total_broods*broodConsumption));//brood 6 death
        h[index].nutrition_deficit = h[index].foraged_increment - totalNutritionRequirements;;

        int insufficientNutrition = 0;

        if(remainingDays > 0) 
        {
            //h[index].should_die += (0.01 * h[index].total_bees);
            insufficientNutrition += max((int)(totalNutritionRequirements * starvation_death_winter_multiplier) - h[index].total_nutrition/remainingDays, 0)/(remainingDays/starvation_death_winter_divisor);

            //h[index].should_die += max(totalNutritionRequirements - h[index].total_nutrition/remainingDays, 0)/remainingDays;
        }
        else
        {
            insufficientNutrition += (int)pow((-min(h[index].nutrition_deficit, 0)), starvation_deficit_power) * starvation_deficit_multiplier;
        }

        //h[index].eggLaid = eggLayingRate2(day);
        h[index].eggLaid = ((p[0].flowers_available / 2000000000000)+1) * eggLayingRate(day);
        //h[index].eggLaid = eggLayingRate(day);

        h[index].should_bee_die = insufficientNutrition/(2*beeConsumption);
        h[index].should_brood_die = insufficientNutrition/(2*broodConsumption);

        h[index].nutrition_deficit += insufficientNutrition;

        if(h[index].total_nutrition < 0)
        {
            h[index].should_bee_die = h[index].total_bees;
            h[index].should_brood_die = h[index].total_broods;
        }


        h[index].unattained_broods = 0;
        //h[index].unattained_broods = max(h[index].total_broods - h[index].total_nurse_bees*4, 0);
        h[index].total_broods = 0;
        h[index].total_bees = 0;
        h[index].total_nurse_bees = 0;
        h[index].total_worker_bees = 0;
        h[index].foraged_increment = 0;
    }
}

__global__ void beeGlobalUpdate(bee *b, int bee_cnt, int lifespan)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < bee_cnt)
    {
        b[index].age += 1;
        //(*(b[blockIdx.x].hive)).total_nutrition -= 0.01;
        //if((*(b[index].hive_ptr)).total_nutrition < 10)
        atomicAdd(&(*(b[index].hive_ptr)).total_nutrition, -beeConsumption);
        atomicAdd(&(*(b[index].hive_ptr)).total_bees, 1);

        if(b[index].age >= 0 && b[index].age <= lifespan*0.7)
        {
            atomicAdd(&(*(b[index].hive_ptr)).total_nurse_bees, 1);
        }
    }
};

__global__ void beeForageUpdate(bee *b, int bee_cnt, plot *p, double efficiency, double lifespan)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < bee_cnt)
    {
        if(b[index].type == 0 && b[index].age >= 0.5*lifespan)
        {
            //p[0].pollinated_flowers += 1;
            atomicAdd(&(*p).pollinated_today, 111*efficiency/8.57087);
            atomicAdd(&(*(b[index].hive_ptr)).total_nutrition, (int)efficiency);
            atomicAdd(&(*(b[index].hive_ptr)).total_worker_bees, 1);
            atomicAdd(&(*(b[index].hive_ptr)).foraged_increment, (int)efficiency);
        }
    }
}

struct bee_death_functor : thrust::unary_function<bee, bool>
{
    const float lifespan;

    bee_death_functor(float _a) : lifespan(_a) {}

    __host__ __device__
    bool operator()(const bee &b)
    {
        return b.age <= lifespan;
    }
};

struct brood_mature_functor : thrust::unary_function<brood, bool>
{
    __host__ __device__
    bool operator()(const brood &x)
    {
        return x.age < 21;
    }
};

int getPlotNum(int& x, int& y)
{
    return y*18+x;
}

void addBee(bee& add_type, const int& count)
{
    thrust::fill(d_bees.begin()+bee_cnt, d_bees.begin()+bee_cnt+count, add_type);
    bee_cnt += count;
}

void addRatioBee(const bee& b, const int& count)
{
    if(count <= 0) return;
    int drone = count/101;
    thrust::fill(d_bees.begin()+bee_cnt, d_bees.begin()+bee_cnt+drone, bee(b.hive_ptr, b.age, b.nutrition, 1));
    thrust::fill(d_bees.begin()+bee_cnt+drone, d_bees.begin()+bee_cnt+count, bee(b.hive_ptr, b.age, b.nutrition, 0));
    bee_cnt += count;
}

int simulationDays = 50;
bool verbose = true;

void outValue(string name, int val)
{
    if(verbose) cout << name << ": " << val << " | ";
    else cout << val << "|";
}

void outValue(string name, float val)
{
    if(verbose) cout << name << ": " << val << " | ";
    else cout << val << "|";
}

void outValue(string name, unsigned long long val)
{
    if(verbose) cout << name << ": " << val << " | ";
    else cout << val << "|";
}

void outValue(string name, double val)
{
    if(verbose) cout << name << ": " << val << " | ";
    else cout << val << "|";
}


double forageEfficiency(double day)
{
    if(day < 60 || day > 340) return 0;
    double efficiency = 0.0000000247 * pow(day, 4) - 0.0000193509 * pow(day, 3) + 0.0044706630 * pow(day, 2) - 0.2590425605 * day + 1.6599060149;
    if(efficiency < 0) efficiency = 0;
    return efficiency * 0.64;
}

int lifeSpan(double day)
{
    return (int)(0.0067211910 * pow(day, 2) - 2.5117894242 * day + 252.9914557348);
}

int main(int argc, char** argv)
{

    hive h;
    h.total_nutrition = 12000000;

    for(int i = 1; i < argc; i++)
    {
        if(!strcmp(argv[i], "-s"))
        {
            verbose = false;
        }

        if(!strcmp(argv[i], "-d"))
        {
            if(i+1 < argc)
            {
                simulationDays = atoi(argv[i+1]);
            }
        }

        if(!strcmp(argv[i], "-bc"))
        {
            if(i+1 < argc)
            {
                bc = atoi(argv[i+1]);
            }
        }
        cudaMemcpyToSymbol(beeConsumption, &bc, sizeof(int), 0, cudaMemcpyHostToDevice);

        if(!strcmp(argv[i], "-pc"))
        {
            if(i+1 < argc)
            {
                pc = atoi(argv[i+1]);
            }
        }
        cudaMemcpyToSymbol(broodConsumption, &pc, sizeof(int), 0, cudaMemcpyHostToDevice);

        if(!strcmp(argv[i], "-e"))
        {
            if(i+1 < argc)
            {
                double eg = atof(argv[i+1]);
                cudaMemcpyToSymbol(egg_laying_rate_multiplier, &eg, sizeof(double), 0, cudaMemcpyHostToDevice);
            }
        }

        if(!strcmp(argv[i], "-eb"))
        {
            if(i+1 < argc)
            {
                int eb = atoi(argv[i+1]);
                cudaMemcpyToSymbol(egg_laying_rate_base, &eb, sizeof(int), 0, cudaMemcpyHostToDevice);
            }
        }

        if(!strcmp(argv[i], "-sw"))
        {
            if(i+1 < argc)
            {
                double stvw = atof(argv[i+1]);
                cudaMemcpyToSymbol(starvation_death_winter_multiplier, &stvw, sizeof(double), 0, cudaMemcpyHostToDevice);
            }
        }

        if(!strcmp(argv[i], "-swd"))
        {
            if(i+1 < argc)
            {
                double stvw = atof(argv[i+1]);
                cudaMemcpyToSymbol(starvation_death_winter_divisor, &stvw, sizeof(double), 0, cudaMemcpyHostToDevice);
            }
        }

        if(!strcmp(argv[i], "-sdm")) { if(i+1 < argc) {
            double sdm = atof(argv[i+1]);
            cudaMemcpyToSymbol(starvation_deficit_multiplier, &sdm, sizeof(double), 0, cudaMemcpyHostToDevice);
        } }

        if(!strcmp(argv[i], "-sdp")) { if(i+1 < argc) {
            double sdp = atof(argv[i+1]);
            cudaMemcpyToSymbol(starvation_deficit_power, &sdp, sizeof(double), 0, cudaMemcpyHostToDevice);
        } }

        if(!strcmp(argv[i], "-in"))
        {
            if(i+1 < argc)
            {
                h.total_nutrition = atoi(argv[i+1]);
            }
        }
    }

    plot h_plot;
    //h_plot.total_flowers = 20000000000;
    h_plot.total_flowers = 200000000000;
    h_plot.pollinated_flowers = 0;

    thrust::fill(d_plots.begin(), d_plots.end(), h_plot);
    plot* d_plot_array = thrust::raw_pointer_cast(d_plots.data());
    
    thrust::fill(d_hives.begin(), d_hives.begin()+1, h);
    hive* d_hive_array = thrust::raw_pointer_cast(d_hives.data());

    host_vector<bee> h_bees(beeMax);

    //initialize bees
    //thrust::fill(d_bees.begin(), d_bees.begin()+bee_cnt, new_bee);

    for(int age = 0; age <= 30; age++)
    {
        addRatioBee(bee(&d_hive_array[0], age, 1, 0), h.total_nutrition/24000);
    }

    //50000 + 9900*14 - 11000*10
    //5000 + 9900*14 - 15000*10 = -6400
    //6400/10 = 640

    brood new_brood = brood(&d_hive_array[0], 0);

    //initialize broods
    int brood_cnt = 1500;
    int matured_brood = 0;
    thrust::fill(d_broods.begin(), d_broods.begin() + brood_cnt, new_brood);
    brood* d_brood_array = thrust::raw_pointer_cast(d_broods.data());

    bee* d_bee_array = thrust::raw_pointer_cast(d_bees.data());

    cudaDeviceSynchronize();

    for(int day = 0; day < simulationDays; day++)
    {
        outValue("Day", day);
        outValue("Brood", brood_cnt);
        outValue("Bee", bee_cnt);

        //thrust::sort(d_bees.begin(), d_bees.begin() + bee_cnt);
        thrust::default_random_engine g;
        thrust::shuffle(d_bees.begin(), d_bees.begin() + bee_cnt, g);

        /*h_bees = d_bees;
        outValue("FIRST", h_bees[0].age);
        outValue("LAST", h_bees[bee_cnt-1].age);*/
        double efficiency = forageEfficiency(day%365);
        outValue("EF", efficiency);
        double lspan = lifeSpan(day%365);
        beeForageUpdate<<<(bee_cnt + threadsPerBlock-1)/threadsPerBlock, threadsPerBlock>>>(d_bee_array, bee_cnt, d_plot_array, efficiency, lspan);
        cudaDeviceSynchronize();

        plotPollinationUpdate<<<(1 + threadsPerBlock-1)/threadsPerBlock, threadsPerBlock>>>(d_plot_array, 1, d_hive_array, efficiency, day%365);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_plot, &d_plot_array[0], sizeof(plot), cudaMemcpyDeviceToHost);
        outValue("UF", h_plot.flowers_available);
        outValue("PF", h_plot.pollinated_flowers);
        outValue("TF", h_plot.total_flowers);

        beeGlobalUpdate<<<(bee_cnt + threadsPerBlock-1)/threadsPerBlock, threadsPerBlock>>>(d_bee_array, bee_cnt, lspan);
        cudaDeviceSynchronize();

        outValue("LSP", lspan);

        detail::normal_iterator<device_ptr<bee>> alive_bee_end = thrust::copy_if(d_bees.begin(), d_bees.begin() + bee_cnt, d_bees.begin(), bee_death_functor(lifeSpan(day%365)));

        //thrust::transform(d_broods.begin(), d_broods.begin() + brood_cnt, d_broods.begin(), brood_functor());
        broodUpdate<<<(brood_cnt + threadsPerBlock-1)/threadsPerBlock, threadsPerBlock>>>(d_brood_array, brood_cnt);
        cudaDeviceSynchronize();

        hive o_hive;

        cudaMemcpy(&o_hive, &d_hive_array[0], sizeof(hive), cudaMemcpyDeviceToHost);

        outValue("Nurse", o_hive.total_nurse_bees);
        outValue("Worker", o_hive.total_worker_bees);

        hiveUpdate<<<(hive_cnt + threadsPerBlock-1)/threadsPerBlock, threadsPerBlock>>>(d_hive_array, hive_cnt, day%365, d_plot_array);
        cudaDeviceSynchronize();

        cudaMemcpy(&o_hive, &d_hive_array[0], sizeof(hive), cudaMemcpyDeviceToHost);

        outValue("Net", o_hive.nutrition_deficit);

        //int brood_die = (int)o_hive.should_die/2;

        //starvation death
        //alive_bee_end -= min((int)o_hive.should_die, (int)(alive_bee_end - d_bees.begin()));
        alive_bee_end -= min((int)o_hive.should_bee_die, (int)(alive_bee_end - d_bees.begin()));

        outValue("D", (int)(d_bees.begin() - alive_bee_end + bee_cnt));
        bee_cnt = alive_bee_end - d_bees.begin();


        outValue("S", o_hive.should_bee_die + o_hive.should_brood_die);

        brood_cnt -= min(o_hive.should_brood_die + o_hive.unattained_broods, brood_cnt);
        //brood_cnt -= min(o_hive.unattained_broods, brood_cnt);
    
        outValue("U", o_hive.unattained_broods);

        detail::normal_iterator<device_ptr<brood>> unmatured_bee_end = thrust::copy_if(d_broods.begin(), d_broods.begin()+brood_cnt, d_broods.begin(), brood_mature_functor());
        matured_brood =(d_broods.begin() + brood_cnt) - unmatured_bee_end;
        matured_brood = int(matured_brood * (1-brood_mature_death_rate));
        outValue("M", matured_brood);
        
       // if(day > 673) return;
        addRatioBee(bee(&d_hive_array[0], 0, 1, 0), matured_brood);

        brood_cnt = unmatured_bee_end - d_broods.begin();

        outValue("N", o_hive.total_nutrition);

        //lay eggs
        int eggLaid = o_hive.eggLaid;
        outValue("Egg", eggLaid);

        thrust::fill(unmatured_bee_end, unmatured_bee_end+eggLaid, new_brood);
        brood_cnt+=eggLaid;
        cout << endl;
    }

    cudaDeviceSynchronize();

    /*h_hives = d_hives;
    cout << o_hive.total_nutrition << endl;*/
}