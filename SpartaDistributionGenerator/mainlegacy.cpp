#include <iostream>
#include <mpi.h>

#include "SpartaAeroInterface.hpp"

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    // Do communication set up
#define MAX_CODES 6
#define SPARTA_ID 5

    int rank, globalsize, localsize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &globalsize);
    MPI_Comm sparta_comm;
    MPI_Comm_split(MPI_COMM_WORLD, MAX_CODES + 1, rank, &sparta_comm); //wrong should use color
    MPI_Comm_size(sparta_comm, &localsize);
    int* leaders = new int[MAX_CODES];
    int* newleaders = new int[MAX_CODES];
    for (int i=0; i<MAX_CODES; ++i)
        leaders[i] = -1;

    int localRank;
    MPI_Comm_rank(sparta_comm, &localRank);
    if (localRank == 0)
        leaders[SPARTA_ID] = rank;

    if (localRank == 0) {
        for(int i = 0; i < MAX_CODES; i++) {
            printf("leaders[%d] = %d\n",i, leaders[i]);
            fflush(stdout);
        }
    }
    MPI_Allreduce(leaders, newleaders, MAX_CODES, MPI_INTEGER, MPI_MAX, MPI_COMM_WORLD);

    if (localRank == 0) {
        for(int i = 0; i < MAX_CODES; i++) {
            printf("newleaders[%d] = %d\n",i, newleaders[i]);
            fflush(stdout);
        }
    }

    MPI_Comm comm2;
    for (int i=0; i<MAX_CODES; ++i) {
     if (i != SPARTA_ID && newleaders[i] >= 0) {
       int tag;
       if (SPARTA_ID < i)
        tag = MAX_CODES*(SPARTA_ID+1)+i+1;
       else
        tag = MAX_CODES*(i+1)+SPARTA_ID+1;
       
       MPI_Intercomm_create(sparta_comm, 0, MPI_COMM_WORLD, newleaders[i], tag, &comm2);
     }
   }

    delete[] leaders;
    delete[] newleaders;


// #define SPARTATAG 887766
//     const int numDOFS = 72;
//     double* dataToSend = new double[numDOFS];
//     for (int i = 0; i < numDOFS; i++) {
//         if (i > 22 || i <= 11) {
//         // if (i < 35 || i > 40) {
//             dataToSend[i] = 0;
//         }
//         else {
//             dataToSend[i] = 100000000.0;
//         }
//     }

//     if (localRank == 0) {
//         for (int t = 0; t <= 100; t++) {
//             // printf("sending %d\n", t);
//             MPI_Send(dataToSend, numDOFS, MPI_DOUBLE, 0, SPARTATAG, comm2);
//         }
//     }
    
    int sparta_rank;
    MPI_Comm_rank(sparta_comm, &sparta_rank);
    printf("SPARTA rank %d (global rank %d) starting.\n", sparta_rank, rank);

    void *sparta_ptr;
    char *input[] = { (char *)"main" };

    sparta_open(1, input, sparta_comm, &sparta_ptr);
    sparta_command(sparta_ptr, (char *)"shell echo 'from sparta from aero'");
    sparta_close(sparta_ptr);
    // MPI_Comm_free(&sparta_comm);

// int rank;
// 	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
// 	MPI_Comm sparta_comm;
//     	int color = (rank >= 1 && rank <= 3) ? 1 : MPI_UNDEFINED;
//     	MPI_Comm_split(MPI_COMM_WORLD, color, rank, &sparta_comm);

//     	if (color == 1) {
//         // Inside SPARTA communicator (ranks 1–3)
//         	 int sparta_rank;
//         	 MPI_Comm_rank(sparta_comm, &sparta_rank);
//         	 printf("SPARTA rank %d (global rank %d) starting.\n", sparta_rank, rank);
        
//         	 void *sparta_ptr;
//         	 char *input[] = { (char *)"main" };
        
//         	 sparta_open(1, input, sparta_comm, &sparta_ptr);
//                  sparta_command(sparta_ptr, (char *)"shell echo 'from sparta from aero'");
//                  sparta_close(sparta_ptr);
// 		 MPI_Comm_free(&sparta_comm);
// 		 return;
//         }
    MPI_Finalize();
}