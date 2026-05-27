#ifndef SPARTA_UTILS_HPP
#define SPARTA_UTILS_HPP

#include <string>

void define_string(void* sparta_ptr, const std::string& name, const std::string& value);
void define_index(void* sparta_ptr, const std::string& name, const std::string& value);
void execute_command(void* sparta_ptr, const std::string& command);
int write_grid_txt(int dim, double xlo, double xhi, double ylo, double yhi,
    double zlo, double zhi, std::string run);
#endif /* SPARTA_UTILS_HPP */
