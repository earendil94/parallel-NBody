#include<stdio.h>
#include<stdlib.h>
#include<stddef.h>
#include<math.h>
#include<mpi.h>
#include<unistd.h>

#define defaultSeed 100
#define G 10e-06
#define default 10
#define epsilon 0.05

//This represents a particle position, velocity, and energy
typedef struct { 
    double p[3];
    double v[3];
    double E;
} particle;

void particleInitialize(int n, int size, double m, int rank); 
void domainDecomposition(int rank, int size, particle *par, int n, int procParticles);

int main(int argc, char **argv){

    FILE * ofp;
    char outputFileName[50];
    
    int N, m;

    if(argc < 2){
        printf("Using default number of particles\n\n");
        N = default;
    } else{
        N = atoi(argv[1]);
    }

    m = 100/N;

    int rank, size, remainder;
    
    //MPI common initial routines
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
       
    //1. Each process writes n/p particles to the file   
    particleInitialize(N, size, m, rank);

    MPI_Finalize();

}


void particleInitialize(int n, int size, double m, int rank){

    //Random number generator between initialization
    int seed = rank*10 + 1; 
    const int rand_max = RAND_MAX;

    //This is the number of particles that every process has to work on
    const int procParticles = n/size + (rank < n%size);

    #if defined(DEBUG)
        printf("Rank %d here. This is my number of particles: %d\n", rank, procParticles);
    #endif

	//Allocation of the particles array
    particle *par;
    if (( par = malloc(procParticles*sizeof(particle))) == NULL ){
        printf("I am sorry, we ran out of memory\n");
        exit(-1);
    };

    for(int register i = 0; i < procParticles; i++){
      
        //Random position values [0;1]
        par[i].p[0] = (double)rand_r(&seed)/rand_max;
        par[i].p[1] = (double)rand_r(&seed)/rand_max;
        par[i].p[2] = (double)rand_r(&seed)/rand_max;

        //Random velocity values [-0.05; 0.05]
        par[i].v[0] = ((double)rand_r(&seed)*2/rand_max - 1)/20;
        par[i].v[1] = ((double)rand_r(&seed)*2/rand_max - 1)/20;
        par[i].v[2] = ((double)rand_r(&seed)*2/rand_max - 1)/20;

        //Multiplication operator is more efficient than pow of math library
        par[i].E = 1./2. * m * par[i].v[0] * par[i].v[0] + par[i].v[1] * par[i].v[1] + par[i].v[2] * par[i].v[2];

    }

    domainDecomposition(rank, size, par, n, procParticles);

}

//We need this function in order for our quicksort to work
int compareParticles(const void *p1, const void *p2){
        particle *par1 = (particle *)p1;
        particle *par2 = (particle *)p2;
        if(par1->p[0] - par2->p[0] > 0)
            return 1;
        else if(par1->p[0] - par2->p[0] < 0)
            return -1;
        else
            return 0;
}

void domainDecomposition(int rank, int size, particle *par, int n, int procParticles){

    //MPI custom type declaration
    int nitems = 3;
    MPI_Datatype types[nitems];
    MPI_Datatype MPI_PARTICLE;
    MPI_Aint offsets[nitems];
    int blocklenghts[nitems];

    types[0] = MPI_DOUBLE;
    types[1] = MPI_DOUBLE;
    types[2] = MPI_DOUBLE;

    offsets[0] = offsetof(particle,p);
    offsets[1] = offsetof(particle,v);
    offsets[2] = offsetof(particle,E);

    blocklenghts[0] = 3;
    blocklenghts[1] = 3;
    blocklenghts[2] = 1;

    MPI_Type_create_struct(nitems,blocklenghts,offsets,types,&MPI_PARTICLE);
    MPI_Type_commit(&MPI_PARTICLE);

    //Let's order the variables
    qsort(par, procParticles,sizeof(particle),compareParticles);

    //We will need this array in order to know how many particles to send to the i-th process
    int *procSendOffsets;
    if((procSendOffsets = calloc(size, sizeof(int))) == NULL){
        printf("Not enough memory to allocate the array, sorry!\n");
        exit(-1);
    }

    //Calculate the right amount of particles to send to every process
    int i = 0;
    double slice = 1./size;

    for(int k = 0; k < size; ++k){
        while(par[i].p[0] < (k+1)*slice && i < procParticles){
            procSendOffsets[k]++;
            ++i;
        }
    }
    
    #if defined(DEBUG)
        for(int k = 0; k < size; ++k)
            printf("In rank %d, procOffset %d is: %d\n", rank, k, procSendOffsets[k]);
    #endif

    //Let's scatter the procSendOffsets array
    //In this way, each MPI process knows how many elements to expect from the first, second, ..., p
    //It's hard to use collectives since the offsetsArray is of dimension size instead of size-1
    int *procReceiveOffsets;
    if((procReceiveOffsets = malloc(size*sizeof(int))) == NULL){
        printf("There is no space for the receive offsets array\n");
        exit(-1);
    }

    //Each process sends to all the other processes
    for(int k = 0; k < size; ++k)
        if(rank != k)
            MPI_Send(&procSendOffsets[k], 1, MPI_INT, k, rank + 50, MPI_COMM_WORLD);

    //Each process receives from all the other processes
    for(int k = 0; k < size; ++k)
        if(rank != k)
            MPI_Recv(&procReceiveOffsets[k], 1, MPI_INT, k, k+50 ,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        else
            procReceiveOffsets[k] = procSendOffsets[k]; 
    

    #if defined(DEBUG)
        for(int k = 0; k < size; ++k)
            printf("In rank %d, procReceiveOffsets %d is: %d\n", rank, k, procReceiveOffsets[k]);
    #endif

    //Now that we know the offsets, let's calculate the displacements
    int *sendDisplacement;
    int *receiveDisplacement;

    if((sendDisplacement = malloc(size*sizeof(int))) == NULL){
        printf("There is no space for the receive offsets array\n");
        exit(-1);
    }

    if((receiveDisplacement = malloc(size*sizeof(int))) == NULL){
        printf("There is no space for the receive offsets array\n");
        exit(-1);
    }

    //First displacement is the beginning of the array
    sendDisplacement[0] = 0;
    for(int register k = 1; k < size; ++k)
        sendDisplacement[k] = sendDisplacement[k-1] + procSendOffsets[k-1];

    //Same thing for the receive displacement
    receiveDisplacement[0] = 0;
    for(int register k = 1; k < size; ++k)
        receiveDisplacement[k] = receiveDisplacement[k-1] + procReceiveOffsets[k-1];

    //We store all the other particles in a final array
    particle *domainParticles;
    int domainParticlesDim = receiveDisplacement[size-1] + procReceiveOffsets[size-1];

    #if defined(IC_DEBUG)
        printf("Rank %d domainParticlesDim = %d\n",  rank, domainParticlesDim);
    #endif
    

    //It needs to contain all the particles that we receive
    if((domainParticles = malloc(domainParticlesDim*sizeof(particle))) == NULL){
        printf("There is no space for the receive offsets array\n");
        exit(-1);
    }

    //Let's use MPI_AlltoAllV
    MPI_Alltoallv(par, procSendOffsets, sendDisplacement, MPI_PARTICLE, domainParticles, 
    procReceiveOffsets, receiveDisplacement, MPI_PARTICLE, MPI_COMM_WORLD);

    //Re-sort
    qsort(domainParticles, domainParticlesDim ,sizeof(particle), compareParticles);

    #if defined(IC_DEBUG)
        sleep(1*rank);
        printf("Rank %d i.c.\n", rank);
        for(int i = 0; i < domainParticlesDim; i++){
            printf("Position %d:\t%f; %f; %f\n", i, domainParticles[i].p[0], domainParticles[i].p[1], domainParticles[i].p[2]);
            printf("Velocity %d:\t%f; %f; %f\n", i, domainParticles[i].v[0], domainParticles[i].v[1], domainParticles[i].v[2]);
            printf("Energy %d:\t%f\n", i, domainParticles[i].E);
            printf("\n\n");
        }
    #endif
    
    
    //Let's free useless stuff here
    free(par);
    free(procSendOffsets);
    free(procReceiveOffsets);
    
    //PARTICLES WRITING SECTION
    //Here we will use MPI I/O to write on the same file the slices
    //Each process will occupy the position of the file given by its rank.
    //In order to arrange the offset, anybody needs to know how many particles are preceeding in the file.
    //Therefore we have no choice but to introduce yet another communication

    int *allSlicesParticles;
    if((allSlicesParticles = malloc(size*sizeof(int))) == NULL){
        printf("No memory left for our communication array, big sorry!\n");
        exit(-1);
    }

    //This collective function comes very handy (TODO: modify procSendOffsets with the same logic)
    MPI_Allgather(&domainParticlesDim,1,MPI_INT,allSlicesParticles,1,MPI_INT,MPI_COMM_WORLD);

    #if defined(IC_DEBUG)
        printf("rank %d allSlicesParticles:\n", rank);
        for(int k = 0; k < size; ++k)
            printf("\t%d\n",allSlicesParticles[k]);
    #endif

    //Displacement for every process
    int fileOffset = 0;

    for(int k = 0; k < rank; k++)
        fileOffset += allSlicesParticles[k];

    #if defined(IC_DEBUG)
        printf("rank %d fileOffset: %d\n", rank, fileOffset);
    #endif

    MPI_File fhw;
	MPI_File_open(MPI_COMM_WORLD, "initialConditions.ic", MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fhw);
    
    //TODO:check position of the header writing in our program
    //Here we define and write some headers
    if(rank == 0){
        const int floatPointPrecision = 8;
        const int numOfFiles = 1; //This is the number of files in which we split the I
        const double timeStep = 0; //This is the initial condition generation, so we are at time zero

        MPI_File_write(fhw, &floatPointPrecision, 1, MPI_INT, MPI_STATUS_IGNORE);
        MPI_File_write(fhw, &n, 1, MPI_INT, MPI_STATUS_IGNORE);
        MPI_File_write(fhw, &numOfFiles, 1, MPI_INT, MPI_STATUS_IGNORE);
        MPI_File_write(fhw, &timeStep, 1, MPI_INT, MPI_STATUS_IGNORE);
    }

    #if defined(IC_DEBUG)
        printf("I am rank %d and this is my fileOffset: %d\n", rank, fileOffset);
        printf("I am rank %d and this is my sliceParticleDim: %d\n", rank, domainParticlesDim);
    #endif
    //We need 4*sizeof(int) to take into consideration the offset of the header
    MPI_File_set_view(fhw, fileOffset*sizeof(particle) + 3*sizeof(int) + sizeof(double), MPI_PARTICLE, MPI_PARTICLE, "native", MPI_INFO_NULL);
    MPI_File_write(fhw, domainParticles, domainParticlesDim, MPI_PARTICLE, MPI_STATUS_IGNORE);
    MPI_File_close(&fhw);

}