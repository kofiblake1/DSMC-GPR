#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <mpi.h>
#include <string>
#include <typeinfo>
#include <unistd.h>
#include <unordered_map>

#include "library.h" 
#include "SpartaAeroInterface.hpp"
#include "sparta_utils.hpp"

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    std::string config_file = "config.txt";
    
    if (argc > 1) {
        config_file = argv[1];
    }

    std::unordered_map<std::string, std::string> params;
    std::ifstream infile(config_file);
    if (!infile.is_open()) {
        std::cerr << "Error: could not open SpartaAeroInterface config file " << config_file << std::endl;
        return 1;
    }

    std::string key, value;
    while (infile >> key >> value) {
        params[key] = value;
    }

    //Basic Info
    int seed = std::stoi(params["seed"]);
    // int dim = std::stoi(params["dim"]);
    std::string run_name            = params["run_name"];
    std::string sparta_input_file   = params["sparta_input_file"];
    std::string sparta_struct_file  = params["sparta_struct_file"];
    std::string sim_mode            = params["sim_mode"];
    bool to_paraview                = (params["to_paraview"] == "true");
    
    // Flow Parameters
    double u_inf                    = std::stod(params["u_inf"]);     
    double t_inf                    = std::stod(params["t_inf"]);
    double rho_inf                  = std::stod(params["rho_inf"]);
    std::string data_prefix         = params["data_prefix"];
    std::string surf_file           = params["surf_file"];
    
    // Mesh Parameters
    std::string dim                 = params["dim"];
    std::string x_min               = params["x_min"];
    std::string x_max               = params["x_max"];
    std::string y_min               = params["y_min"];
    std::string y_max               = params["y_max"];
    std::string z_min               = params["z_min"];
    std::string z_max               = params["z_max"];
    std::string x_level_1_ref       = params["x_level_1_ref"];
    std::string x_level_2_ref       = params["x_level_2_ref"];
    std::string x_level_3_ref       = params["x_level_3_ref"];
    std::string x_level_4_ref       = params["x_level_4_ref"];
    std::string y_level_1_ref       = params["y_level_1_ref"];
    std::string y_level_2_ref       = params["y_level_2_ref"];
    std::string y_level_3_ref       = params["y_level_3_ref"];
    std::string y_level_4_ref       = params["y_level_4_ref"];
    std::string z_level_1_ref       = params["z_level_1_ref"];
    std::string z_level_2_ref       = params["z_level_2_ref"];
    std::string z_level_3_ref       = params["z_level_3_ref"];
    std::string z_level_4_ref       = params["z_level_4_ref"];
    std::string surf_ref            = params["surf_ref"];
    std::string surf_ref_level      = params["surf_ref_level"];

    // Timing Parameters
    double dt                     = std::stod(params["dt"]);
    int total_timesteps            = std::stoi(params["total_timesteps"]);


    // Establishes MPMD style communication for SpartaAeroInterface (wrapper for SPARTA)
    // and AERO. This is done mimicking the MPMD communication pattern in AERO, and can
    // be found in the AERO-S directory in Comm.d/CommunicatorCore.C Communicator::split()
    // and main.C
#define MAX_CODES 6
#define FLUID_ID 0
#define STRUC_ID 1
#define HEAT_ID  2
#define EMBED_ID 3
#define HYGRO_ID 4
#define SPARTA_ID 5

    // Assigns global size and local size based on number of ranks in total MPI call vs ranks
    // allocate for Sparta (through SpartaAeroInterface)

    int rank, globalsize, localsize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &globalsize);
    MPI_Comm sparta_comm;
    MPI_Comm_split(MPI_COMM_WORLD, SPARTA_ID + 1, rank, &sparta_comm); //wrong should use color
    MPI_Comm_size(sparta_comm, &localsize);

    // Assigns the processor with localrank = 0 as the leader of the SPARTA ranks for
    // intercommunication
    int* leaders = new int[MAX_CODES];
    int* newleaders = new int[MAX_CODES];
    for (int i=0; i<MAX_CODES; ++i)
        leaders[i] = -1;

    int localRank;
    MPI_Comm_rank(sparta_comm, &localRank);

    if (localRank == 0)
        leaders[SPARTA_ID] = rank;
    MPI_Allreduce(leaders, newleaders, MAX_CODES, MPI_INTEGER, MPI_MAX, MPI_COMM_WORLD);

    std::string newleaders_string = std::to_string(newleaders[0]);
    for(int i = 1; i < MAX_CODES; i++) {
        newleaders_string += " " + std::to_string(newleaders[i]);
    }

    // for (int i = 0; i < MAX_CODES; i++) {
    //     fprintf(stderr,"newleaders[%d]=%d\n",i,newleaders[i]);
    // }

    // std::cerr << "newleaders_string: " << newleaders_string << std::endl;
    std::string aero_rank = std::to_string(newleaders[2]);

    // Create folder on root node
    if (localRank == 0)
    {
        if (std::filesystem::exists(run_name) == false)
        {
            std::filesystem::create_directory(run_name);
        }
    }

    // Launch Sparta
    char* dummy_args[1];
    dummy_args[0] = argv[0];
    try
    {
        void *sparta_ptr;
        sparta_open(1, dummy_args, sparta_comm, &sparta_ptr);

        // Load Variables
        define_string(sparta_ptr, "MESH", sparta_input_file);
        define_string(sparta_ptr, "RUN", run_name);
        define_string(sparta_ptr, "OUT", data_prefix);
        define_string(sparta_ptr, "DIM", dim);
        define_string(sparta_ptr, "AERO_LEAD", aero_rank);
        define_string(sparta_ptr, "X_MIN", x_min);
        define_string(sparta_ptr, "X_MAX", x_max);
        define_string(sparta_ptr, "Y_MIN", y_min);
        define_string(sparta_ptr, "Y_MAX", y_max);
        define_string(sparta_ptr, "Z_MIN", z_min);
        define_string(sparta_ptr, "Z_MAX", z_max);
        define_index(sparta_ptr, "newleaders", newleaders_string);

        // Load File
        sparta_file(sparta_ptr, &sparta_input_file[0]);
        sparta_close(sparta_ptr);
    }
    catch (...)
    {
        std::cerr << "Error thrown by Sparta during run." << std::endl;
    }
    delete [] leaders;
    delete [] newleaders;

    // Write out grid file for post processing
    if (localRank == 0 && to_paraview)
    {
        write_grid_txt(
            std::stoi(dim),
            std::stod(x_min),
            std::stod(x_max),
            std::stod(y_min),
            std::stod(y_max),
            std::stod(z_min),
            std::stod(z_max),
            run_name); 
    }

    MPI_Finalize();
    return 0;
}
