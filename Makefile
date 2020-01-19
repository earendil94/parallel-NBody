CC=gcc
MPICC=mpicc

all: openBody.x MPIBody.x initialConditions.x particlesToCSV.x

openBody.x: src/openBody.c
	$(CC) -fopenmp -o $@ $< -std=c99 -lm

MPIBody.x: src/MPIBody.c
	$(MPICC) -o $@ $< -std=c99 -lm

initialConditions.x: src/initialConditions.c
	$(MPICC) -o $@ $< -std=gnu99

particlesToCSV.x: src/particlesToCSV.c
	$(CC) -o $@ $< -std=c99

OB: openBody.c
	gcc -fopenmp -o openBody openBody.c -std=c99 -lrt -lm

IC: initialConditions.c
	mpicc -o initialConditions initialConditions.c -std=gnu99

MB: MPIBody.c
	mpicc -o MPIBody MPIBody.c -std=gnu99 -lm

#Debug makefile
DebugOB: openBody.c
	gcc -DDEBUG -fopenmp -o openBody openBody.c -std=c99 -lrt -lm

DebugIC: src/initialConditions.c
	mpicc -g -DIC_DEBUG -o initialConditions.x src/initialConditions.c -std=gnu99

DebugMB: src/MPIBody.c
	mpicc -g -DDEBUG -o MPIBody.x src/MPIBody.c -std=gnu99 -lm

DebugHB: src/hybridBody.c
	$(MPICC) -DDEBUG -fopenmp -o hybridBody.x src/hybridBody.c -std=c99 -lm

clean:
	rm *.x