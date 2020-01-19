#if defined(__STDC__)
#  if (__STDC_VERSION__ >= 199901L)
#     define _XOPEN_SOURCE 700
#  endif
#endif
#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sched.h>
#include <omp.h>
#include <math.h>

#define G 10e-06
#define epsilon 0.5
#define gamma 0.05

//This represents a particle position, velocity, force (in a 3d vector) and energy
typedef struct { 
    double p[3];
    double v[3];
    double E;
} particle;

//OpenMP utility functions
int read_proc__self_stat ( int, int * );
int get_cpu_id           ( void       );

particle * readParticles(FILE *ifp, int n);
double nBodyEvolution(particle *par, int n, double m);
void updateParticles(particle *par, int n, double m, double t, double *Fx, double *Fy, double *Fz);
int sgn(double a, double b);
void writeParticle(particle * pars, FILE *fp, int *floatPointPrecision, int *n, int *numOfFiles, double *time);

int main(int argc, char **argv){

    //General usage variables
    unsigned short int n_it = 0;

    //SomeOpenMP variables
    int nThreads;

    //We need to read these variables in order from our file
    int floatPointPrecision;
    int n; //We will use this to sto
    int numOfFiles; //This is the number of files in which we split the I
    double time; //This is the initial condition generation, so we are at time zero


    // #if defined(_OPENMP)  
    //     #pragma omp parallel
    //     {
    //     #pragma omp master
    //     {
    //         nThreads = omp_get_num_threads();
    //         printf("omp summation with %d threads\n", nThreads );
    //     }

    //     int me = omp_get_thread_num();
    //     #pragma omp critical
    //         printf("thread %2d is running on core %2d\n", me, get_cpu_id() );    
    //     }
    // #endif

    //Files declaration
    FILE *ifp;
    char inputFileName[] = "initialConditions.ic";
    ifp = fopen(inputFileName, "r");


    fread(&floatPointPrecision,sizeof(int), 1, ifp);
    fread(&n, sizeof(int), 1, ifp);
    fread(&numOfFiles, sizeof(int),1, ifp);
    fread(&time, sizeof(double), 1, ifp);


    #if defined(DEBUG)
        printf("The floatPointPrecision of our program is: %d\n", floatPointPrecision);
        printf("The number of particles of our program is: %d\n", n);
        printf("The number of files of our program is: %d\n", numOfFiles);
        printf("The starting time step is: %f\n", time);
    #endif


    //Particles declaration and read, we do it serially
    particle *pars;
    pars = readParticles(ifp, n);

    
    #if defined(DEBUG)
        for(int q = 0; q < n; q++){
            printf("Position %d:\t%f; %f; %f\n", q, pars[q].p[0], pars[q].p[1], pars[q].p[2]);
            printf("Velocity %d:\t%f; %f; %f\n", q, pars[q].v[0], pars[q].v[1], pars[q].v[2]);
            printf("Energy %d:\t%f\n", q, pars[q].E);
            printf("\n\n");
        }
    #endif

    //The mass of every particle is constant
    const double m = 100./(double) n;
    time = 0;

    //Particle update section
    do{
        printf("Computing %d-st iteration:\n\n", n_it);
        time = nBodyEvolution(pars, n, m);
        n_it++;

        FILE *ofp;
        char ofpFilename[30];
        snprintf(ofpFilename, sizeof(ofpFilename), "particles_%d_openMp.ic", n_it);
        ofp = fopen(ofpFilename, "w");
        writeParticle(pars, ofp, &floatPointPrecision, &n, &numOfFiles, &time);
        fclose(ofp);
    }while( n_it < 2);

    // #if defined(DEBUG)
    //     for(int q = 0; q < n; q++){
    //         printf("Position %d:\t%f; %f; %f\n", q, pars[q].p[0], pars[q].p[1], pars[q].p[2]);
    //         printf("Velocity %d:\t%f; %f; %f\n", q, pars[q].v[0], pars[q].v[1], pars[q].v[2]);
    //         printf("Energy %d:\t%f\n", q, pars[q].E);
    //         printf("\n\n");
    //     }
    // #endif

}

particle * readParticles(FILE *ifp, int n){

    //ParticleRead
    particle *pars;

    if((pars = malloc(n*sizeof(particle))) == NULL){
        printf("Not enough space for all our particles");
        exit(-1);
    }

    fread(pars, sizeof(particle), n, ifp);

    return pars;

}

void writeParticle(particle * pars, FILE *ofp, int *floatPointPrecision, int *n, int *numOfFiles, double *time){

    //Write header
    fwrite(floatPointPrecision, sizeof(int), 1, ofp);
    fwrite(n, sizeof(int), 1, ofp);
    fwrite(numOfFiles, sizeof(int), 1, ofp);
    fwrite(time, sizeof(double), 1, ofp);

    //Write particles
    fwrite(pars, sizeof(particle), *n, ofp);
}


double nBodyEvolution(particle *par, int n, double m){

    //Unreasonably high first value so that we correctly calculate the min
    double tMin = 10;
    double t;

    //We need to save the forces for each particles. Having them in the struct would have meant
    //Useless movement of data (which is a costly operation) between processes, which is something
    //We want to avoid. We don't need the force outside of this function and the function that 
    //is called at the end.
    double *Fx;
    double *Fy;
    double *Fz;
    double *times; //This should be specific for our openMp implementation to avoid data race

    if((Fx = malloc(n*sizeof(double))) == NULL){
        printf("Not enough space to allocate Fx\n");
        exit(-1);
    }

    if((Fy = malloc(n*sizeof(double))) == NULL){
        printf("Not enough space to allocate Fx\n");
        exit(-1);
    }

    if((Fz = malloc(n*sizeof(double))) == NULL){
        printf("Not enough space to allocate Fx\n");
        exit(-1);
    }

    if((times = malloc(n*sizeof(double))) == NULL){
        printf("Not enough space to allocate Fx\n");
        exit(-1);
    }

    #if defined(_OPENMP)
        #pragma omp parallel for 
    #endif
        for(int register i = 0; i < n; i++){     
            for( int register k = 0; k < n; k++){

                //TODO: change this to k != i ? :
                if( k != i){
                    //We will use the soften potential (gamma factor)
                    Fx[i] += G*m*m*sgn(par[i].p[0], par[k].p[0]) / ( (par[i].p[0] - par[k].p[0]) * (par[i].p[0] - par[k].p[0]) + gamma/(double) n);
                    Fy[i] += G*m*m*sgn(par[i].p[1], par[k].p[1]) / ( (par[i].p[1] - par[k].p[1]) * (par[i].p[1] - par[k].p[1]) + gamma/(double) n);
                    Fz[i] += G*m*m*sgn(par[i].p[2], par[k].p[2]) / ( (par[i].p[2] - par[k].p[2]) * (par[i].p[2] - par[k].p[2]) + gamma/(double) n);
                }

            }

            //We have to take the minimum between the maximum of times
            //Each time is calculated as || V || * eps / || a |||
            //TODO:maybe there is space for optimization even here just considering the squares
            times[i] = epsilon * m *
                sqrt(par[i].v[0]*par[i].v[0] + par[i].v[1]*par[i].v[1] + par[i].v[2]*par[i].v[2]) /
                sqrt(Fx[i]*Fx[i] + Fy[i]*Fy[i] + Fz[i]*Fz[i]);

        }

    //It's better to handle our shared variable in a serial way to avoid data race
    for(int register i = 0; i < n; i++){
        if(tMin > times[i])
            tMin = times[i];
    }


    #if defined(DEBUG)
        printf("\n\ntMin: %f\n\n", tMin);
    #endif

    updateParticles(par, n, m, tMin, Fx, Fy, Fz);

    free(Fx);
    free(Fy);
    free(Fz);

    return tMin;
}



void updateParticles(particle *par, int n, double m, double t, double *Fx, double *Fy, double *Fz){

    //Second doubt: for the way we have defined our deltaV, we are now summing two times our initial v
    //I would suggest a very stupid update like the one that follows
    #if defined(_OPENMP)
        #pragma omp parallel
        {
    #endif

    //If we define this as parallel, these two variables will be different for each thread,
    //thus avoiding data race
    double positionDistance = 0;
    double squaredVelocity = 0;

    //v = a*t + v0
    #if defined(_OPENMP)
        #pragma omp for
    #endif
        for( int register i = 0; i < n; i++){
            par[i].v[0] += Fx[i]/m*t;
            par[i].v[1] += Fy[i]/m*t;
            par[i].v[2] += Fz[i]/m*t; 
        }

    //s  = v*t + s0
    #if defined(_OPENMP)
        #pragma omp for
    #endif
        for( int register i = 0; i < n; i++){
            par[i].p[0] += par[i].v[0]*t;
            par[i].p[1] += par[i].v[1]*t;
            par[i].p[2] += par[i].v[2]*t; 
        }
    
    //E = Ekin + U
    #if defined(_OPENMP)
        #pragma omp for
    #endif
        for( int register i = 0; i < n; i++){
            for( int k = 0; k < n; k++)
                //TODO: change this to k != i ? :
                if( k != i)
                    positionDistance += sqrt((par[i].p[0]-par[k].p[0])*(par[i].p[0] - par[k].p[0]) + 
                        (par[i].p[1]-par[k].p[1])*(par[i].p[1] - par[k].p[1]) + (par[i].p[2]-par[k].p[2])*(par[i].p[2] - par[k].p[2]) + gamma); 
            
            squaredVelocity = par[i].v[0]*par[i].v[0] + par[i].v[1]*par[i].v[1] + par[i].v[2]*par[i].v[2];
            par[i].E += 0.5*m*squaredVelocity +  G*(m*n-1) / positionDistance; //The total mass is considered       
        }

    #if defined(_OPENMP)
        }
    #endif
}

//Auxiliary function, dunno if we need this
int sgn(double a, double b){
    if(a > b)
        return 1;
    else if( a  < b)
        return -1;    
}

int get_cpu_id( void )
{
#if defined(_GNU_SOURCE)                              // GNU SOURCE ------------
  
  return  sched_getcpu( );

#else

#ifdef SYS_getcpu                                     //     direct sys call ---
  
  int cpuid;
  if ( syscall( SYS_getcpu, &cpuid, NULL, NULL ) == -1 )
    return -1;
  else
    return cpuid;
  
#else      

  unsigned val;
  if ( read_proc__self_stat( CPU_ID_ENTRY_IN_PROCSTAT, &val ) == -1 )
    return -1;

  return (int)val;

#endif                                                // -----------------------
#endif

}



int read_proc__self_stat( int field, int *ret_val )
/*
  Other interesting fields:

  pid      : 0
  father   : 1
  utime    : 13
  cutime   : 14
  nthreads : 18
  rss      : 22
  cpuid    : 39

  read man /proc page for fully detailed infos
 */
{
  // not used, just mnemonic
  // char *table[ 52 ] = { [0]="pid", [1]="father", [13]="utime", [14]="cutime", [18]="nthreads", [22]="rss", [38]="cpuid"};

  *ret_val = 0;

  FILE *file = fopen( "/proc/self/stat", "r" );
  if (file == NULL )
    return -1;

  char   *line = NULL;
  int     ret;
  size_t  len;
  ret = getline( &line, &len, file );
  fclose(file);

  if( ret == -1 )
    return -1;

  char *savetoken = line;
  char *token = strtok_r( line, " ", &savetoken);
  --field;
  do { token = strtok_r( NULL, " ", &savetoken); field--; } while( field );

  *ret_val = atoi(token);

  free(line);

  return 0;
}