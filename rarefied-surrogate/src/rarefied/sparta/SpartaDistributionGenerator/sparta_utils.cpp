#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <string>
#include <typeinfo>
#include <unistd.h>

#include "../sparta-20Jan2025/src/library.h"
#include "sparta_utils.hpp"

void define_string(void* sparta_ptr, const std::string& name, const std::string& value)
{
    std::string command = "variable " + name + " string " + value;
    sparta_command(sparta_ptr, &command[0]);
}

void define_index(void* sparta_ptr, const std::string& name, const std::string& value)
{
    std::string command = "variable " + name + " index " + value;
    sparta_command(sparta_ptr, &command[0]);
}

void execute_command(void* sparta_ptr, const std::string& command)
{
    sparta_command(sparta_ptr, const_cast<char*>(command.c_str()));
}

int write_grid_txt(int dim, double xlo, double xhi, double ylo, double yhi,
    double zlo, double zhi, std::string run)
{
std::string filename = run + "/" + run + ".txt";
std::string read_grid_name = run + ".grid";
std::ofstream grid_file(filename);

if (grid_file.is_open())
{
grid_file << "dimension \t" << dim << std::endl;
grid_file << "create_box \t"
   << xlo << " "
   << xhi << " "
   << ylo << " "
   << yhi << " "
   << zlo << " "
   << zhi << std::endl;
grid_file << "read_grid \t" << read_grid_name << std::endl;
grid_file.close();
}
else
std::cerr << "Unable to open grid txt file";

return 0;
}

