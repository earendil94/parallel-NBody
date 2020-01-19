#include<stdio.h>
#include<stdlib.h>

typedef struct { 
    double p[3];
    double v[3];
    double E;
} particle;

int main(int argc, char **argv){

    if(argc < 2){
        printf("I need to know which file you want to transform in csv\n");
        exit(-1);
    }

    FILE *ifp, *ofp;
    char *filenameInput = argv[1];
    char filenameOutput[40];
    sprintf(filenameOutput, "%s.csv", filenameInput);

    ifp = fopen(filenameInput, "r");

    //Reading the header
    int floatPointPrecision;
    int n;
    int numOfFiles;
    double time;
    fread(&floatPointPrecision, sizeof(int), 1, ifp);
    fread(&n, sizeof(int), 1, ifp);
    fread(&numOfFiles, sizeof(int), 1, ifp);
    fread(&time, sizeof(double), 1, ifp);

    printf("Check:\n");
    printf("\t floatPointPrec: %d\n", floatPointPrecision);
    printf("\t n: %d\n", n);
    printf("\t numOfFiles: %d\n", numOfFiles);
    printf("\t time: %f\n", time);

    //Particles read
    particle *pars;

    if((pars = malloc(sizeof(particle) * n)) == NULL){
        printf("Not enough memory to allocate our particles array\n");
        exit(-1);
    }

    fread(pars, sizeof(particle), n, ifp);

    fclose(ifp);

    ofp = fopen(filenameOutput, "w");

    //printf("First particle position: %f\n", pars[0].p[0]);

    char header[] = "x;y;z;vx;vy;vz\n";

    fprintf(ofp, "%s", header);

    for(int register i = 0; i < n; ++i){
        fprintf(ofp, "%f;%f;%f;%f;%f;%f\n", 
        pars[i].p[0], pars[i].p[1],pars[i].p[2], pars[i].v[0], pars[i].v[1], pars[i].v[2]);
    }

    fclose(ofp);

    return 0;
}