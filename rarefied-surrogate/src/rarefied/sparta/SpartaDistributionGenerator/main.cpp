#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <mpi.h>
#include <string>
#include <typeinfo>
#include <unistd.h>
#include <unordered_map>
#include <random>
#include <cmath>
#include <vector>
#include <sstream>
#include <iomanip>
#include <chrono>

#include "library.h" 
#include "SpartaAeroInterface.hpp"
#include "sparta_utils.hpp"

struct BoundingBox {
    int region_index;
    int num_points;
    double min_x, max_x, min_y, max_y;
    double normal_x, normal_y;
    double tangent_x, tangent_y;
    double circle_center_x, circle_center_y;
    double circle_radius;
    double curvature;
    double local_aoa_rad;
    double bbox_area;
    double object_intersection_area;
    double collection_area;
};

struct MicroCircle {
    int circle_index;
    double center_x;
    double center_y;
    double radius;
};

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int localRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &localRank);

    std::string config_file = "config.txt";
    
    if (argc > 1) {
        config_file = argv[1];
    }

    std::unordered_map<std::string, std::string> params;
    std::ifstream infile(config_file);
    if (!infile.is_open()) {
        std::cerr << "Error: could not open SpartaDistributionGenerator config file " << config_file << std::endl;
        return 1;
    }

    std::string key, value;
    while (infile >> key >> value) {
        params[key] = value;
    }

    std::string sparta_input_file   = params["sparta_input_file"];
    std::string run_name            = params["run_name"];
    std::string data                = params["data"];
    // bool to_paraview                = (params["to_paraview"] == "true");
    std::string dim                 = params["dim"];
    std::string x_min               = params["x_min"];
    std::string x_max               = params["x_max"];
    std::string y_min               = params["y_min"];
    std::string y_max               = params["y_max"];
    std::string z_min               = params["z_min"];
    std::string z_max               = params["z_max"];
    std::string mach_min            = params["mach_min"];
    std::string mach_max            = params["mach_max"];
    std::string t0_min              = params["t0_min"];
    std::string t0_max              = params["t0_max"];
    std::string surface_folder      = params["surface_folder"];
    std::string surface_stem        = params["surface_stem"];
    int num_samples                 = std::stoi(params["num_samples"]);

    // Pre vs post data collection
    int predata_timesteps = std::stoi(params["predata_timesteps"]);
    int total_timesteps = std::stoi(params["total_timesteps"]);

    // Data Collection booleans
    int do_grid = (params["do_grid"] == "true") ? 1 : 0;
    int Nevery_grid = std::stoi(params["Nevery_grid"]);
    int Nrepeat_grid = std::stoi(params["Nrepeat_grid"]);
    int Nfreq_grid = std::stoi(params["Nfreq_grid"]);

    int do_surf = (params["do_surf"] == "true") ? 1 : 0;
    int Nevery_surf = std::stoi(params["Nevery_surf"]);
    int Nrepeat_surf = std::stoi(params["Nrepeat_surf"]);
    int Nfreq_surf = std::stoi(params["Nfreq_surf"]);

    int do_therm = (params["do_therm"] == "true") ? 1 : 0;
    int Nevery_therm = std::stoi(params["Nevery_therm"]);
    int Nrepeat_therm = std::stoi(params["Nrepeat_therm"]);
    int Nfreq_therm = std::stoi(params["Nfreq_therm"]);

    int do_2D = (params["do_2D"] == "true") ? 1 : 0;
    int Nevery_2D = std::stoi(params["Nevery_2D"]);
    int Nrepeat_2D = std::stoi(params["Nrepeat_2D"]);
    int Nfreq_2D = std::stoi(params["Nfreq_2D"]);
    int lo_2D = std::stoi(params["lo_2D"]);
    int hi_2D = std::stoi(params["hi_2D"]);
    int nbin_2D = std::stoi(params["nbin_2D"]);

    int do_3D = (params["do_3D"] == "true") ? 1 : 0;
    int Nevery_3D = std::stoi(params["Nevery_3D"]);
    int Nrepeat_3D = std::stoi(params["Nrepeat_3D"]);
    int Nfreq_3D = std::stoi(params["Nfreq_3D"]);
    int lo_3D = std::stoi(params["lo_3D"]);
    int hi_3D = std::stoi(params["hi_3D"]);
    int nbin_3D = std::stoi(params["nbin_3D"]);

    std::vector<std::vector<BoundingBox>> all_bboxes(num_samples);
    std::vector<std::unordered_map<int, std::vector<MicroCircle>>> all_micro_circles(num_samples);

    auto split_csv_line = [](const std::string& input) {
        std::vector<std::string> fields;
        std::stringstream ss(input);
        std::string token;
        while (std::getline(ss, token, ',')) {
            fields.push_back(token);
        }
        return fields;
    };

    

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis_mach(std::stod(mach_min), std::stod(mach_max));
    std::uniform_real_distribution<> dis_temp(std::stod(t0_min), std::stod(t0_max));
    const double gamma = 1.4;
    const double R = 287.0;

    for(int i = 0; i < num_samples; i++){
        std::string surf_file = surface_folder + "/" + surface_stem + "_" + std::to_string(i) + ".txt";
        std::string metadata_file = surface_folder + "/" + surface_stem + "_" + std::to_string(i) + "_metadata.txt";

        // Read metadata
        std::ifstream meta_in(metadata_file);
        if (!meta_in.is_open()) {
            std::cerr << "Error: could not open metadata file " << metadata_file << std::endl;
            continue;
        }
        std::string line;
        bool found = false;
        while (std::getline(meta_in, line)) {
            if (line.find("REGION SUMMARY:") != std::string::npos) {
                found = true;
                break;
            }
        }
        if (!found) {
            std::cerr << "Region summary section not found in " << metadata_file << std::endl;
            meta_in.close();
            continue;
        }
        
        // Skip summary CSV header line
        std::getline(meta_in, line);

        std::unordered_map<int, std::size_t> region_to_bbox_idx;
        
        // Parse REGION SUMMARY rows until DETAILED CIRCLE DATA section
        while (std::getline(meta_in, line)) {
            if (line.find("DETAILED CIRCLE DATA:") != std::string::npos) {
                break;
            }

            if (line.empty() || line.find(',') == std::string::npos) {
                continue;
            }

            const auto fields = split_csv_line(line);
            if (fields.size() != 16 && fields.size() != 17) {
                continue;
            }

            BoundingBox bb;
            bb.region_index = std::stoi(fields[0]);
            bb.num_points = std::stoi(fields[1]);
            bb.min_x = std::stod(fields[2]);
            bb.max_x = std::stod(fields[3]);
            bb.min_y = std::stod(fields[4]);
            bb.max_y = std::stod(fields[5]);
            bb.normal_x = std::stod(fields[6]);
            bb.normal_y = std::stod(fields[7]);
            bb.tangent_x = std::stod(fields[8]);
            bb.tangent_y = std::stod(fields[9]);
            bb.circle_center_x = std::stod(fields[10]);
            bb.circle_center_y = std::stod(fields[11]);
            bb.circle_radius = std::stod(fields[12]);
            bb.local_aoa_rad = std::stod(fields[13]);

            if (fields.size() == 17) {
                bb.bbox_area = std::stod(fields[14]);
                bb.object_intersection_area = std::stod(fields[15]);
                bb.collection_area = std::stod(fields[16]);
            } else {
                bb.bbox_area = std::stod(fields[14]); // union area in new metadata format
                bb.object_intersection_area = 0.0;
                bb.collection_area = std::stod(fields[15]);
            }
            
            // Calculate curvature as 1 / circle_radius
            bb.curvature = (bb.circle_radius != 0.0) ? 1.0 / bb.circle_radius : 0.0;
            
            region_to_bbox_idx[bb.region_index] = all_bboxes[i].size();
            all_bboxes[i].push_back(bb);
        }

        // Parse DETAILED CIRCLE DATA section
        int current_region_index = -1;
        while (std::getline(meta_in, line)) {
            if (line.empty()) {
                continue;
            }

            if (line.rfind("Region ", 0) == 0) {
                std::size_t start = std::string("Region ").size();
                std::size_t end = line.find(' ', start);
                if (end == std::string::npos) {
                    end = line.find('(', start);
                }

                if (end != std::string::npos) {
                    current_region_index = std::stoi(line.substr(start, end - start));
                }
                continue;
            }

            if (line.rfind("Circle_Index", 0) == 0 || line.find(',') == std::string::npos) {
                continue;
            }

            if (current_region_index < 0) {
                continue;
            }

            auto bbox_it = region_to_bbox_idx.find(current_region_index);
            if (bbox_it == region_to_bbox_idx.end()) {
                continue;
            }

            const auto fields = split_csv_line(line);
            if (fields.size() < 4) {
                continue;
            }

            MicroCircle mc;
            mc.circle_index = std::stoi(fields[0]);
            mc.center_x = std::stod(fields[1]);
            mc.center_y = std::stod(fields[2]);
            mc.radius = std::stod(fields[3]);
            all_micro_circles[i][current_region_index].push_back(mc);
        }
        meta_in.close();

        std::string run_name_i = run_name + "_" + std::to_string(i);
        std::filesystem::path sample_dir = std::filesystem::path(run_name) / std::to_string(i);

        double mach = dis_mach(gen);
        double air_temp = dis_temp(gen);

        double Mmin = std::stod(mach_min);
        double Mmax = std::stod(mach_max);

        double Mi = Mmin + (Mmax - Mmin) * i / std::max(1, num_samples - 1);
        double speed_of_sound = 263.41;
        double ui = Mi * speed_of_sound;

        // Create folder only on root rank
        if (localRank == 0)
        {
            std::filesystem::create_directories(sample_dir);
        }

        // Launch Sparta
        char* dummy_args[1];
        dummy_args[0] = argv[0];
        try
        {
            void *sparta_ptr;
            sparta_open(1, dummy_args, MPI_COMM_WORLD, &sparta_ptr);

            // Load Variables
            define_string(sparta_ptr, "MESH", surf_file);
            define_string(sparta_ptr, "RUN", run_name);
            define_string(sparta_ptr, "SAMPLE", std::to_string(i));
            define_string(sparta_ptr, "OUT", data);
            define_string(sparta_ptr, "DIM", dim);
            define_string(sparta_ptr, "X_MIN", x_min);
            define_string(sparta_ptr, "X_MAX", x_max);
            define_string(sparta_ptr, "Y_MIN", y_min);
            define_string(sparta_ptr, "Y_MAX", y_max);
            define_string(sparta_ptr, "Z_MIN", z_min);
            define_string(sparta_ptr, "Z_MAX", z_max);
            // define_string(sparta_ptr, "AIR_TEMP", std::to_string(air_temp));
            // define_string(sparta_ptr, "AIR_VEL", std::to_string(air_vel));

            
            define_string(sparta_ptr, "U_INF", std::to_string(ui));

            // Load File
            sparta_file(sparta_ptr, &sparta_input_file[0]);

            std::string cmd_run_predata = "run " + std::to_string(predata_timesteps);
            execute_command(sparta_ptr, cmd_run_predata);

            // Define regions as union of circle microregions
            for(const auto& bb : all_bboxes[i]) {
                std::string region_index_str = std::to_string(bb.region_index);
                auto format_double = [](double value) {
                    std::ostringstream oss;
                    oss << std::setprecision(17) << value;
                    return oss.str();
                };

                // Compute angle between tangent direction and +x unit vector in [-pi, pi]
                // Tangent vectors are read directly from metadata
                double region_theta = std::atan2(bb.tangent_y, bb.tangent_x);
                std::string region_theta_str = format_double(region_theta);
                std::string var_t_name = "v_t_region_" + region_index_str;
                std::string var_n_name = "v_n_region_" + region_index_str;

                std::vector<std::string> circle_region_names;
                MicroCircle single_valid_circle{};
                bool has_single_valid_circle = false;
                auto region_circle_it = all_micro_circles[i].find(bb.region_index);
                const std::vector<MicroCircle>* region_circles = nullptr;
                if (region_circle_it != all_micro_circles[i].end()) {
                    region_circles = &region_circle_it->second;
                }

                if (region_circles != nullptr) {
                    for (const auto& mc : *region_circles) {
                        if (mc.radius <= 0.0) {
                            continue;
                        }

                        if (mc.circle_index < 0) {
                            continue;
                        }

                        std::string circle_region_name = "region_" + region_index_str + "_c_" + std::to_string(mc.circle_index);
                        std::string cmd_circle = "region " + circle_region_name + " sphere " + format_double(mc.center_x) + " " + format_double(mc.center_y) + " 0.0 " + format_double(mc.radius);
                        execute_command(sparta_ptr, cmd_circle);
                        circle_region_names.push_back(circle_region_name);
                        if (!has_single_valid_circle) {
                            single_valid_circle = mc;
                            has_single_valid_circle = true;
                        }
                    }
                }

                if (circle_region_names.empty()) {
                    std::string fallback_cmd = "region " + region_index_str + " block " + std::to_string(bb.min_x) + " " + std::to_string(bb.max_x) + " " + std::to_string(bb.min_y) + " " + std::to_string(bb.max_y) + " INF INF";
                    execute_command(sparta_ptr, fallback_cmd);
                } else if (circle_region_names.size() == 1 && has_single_valid_circle) {
                    std::string cmd_region = "region " + region_index_str + " sphere " + format_double(single_valid_circle.center_x) + " " + format_double(single_valid_circle.center_y) + " 0.0 " + format_double(single_valid_circle.radius);
                    execute_command(sparta_ptr, cmd_region);
                } else {
                    std::ostringstream union_cmd;
                    union_cmd << "region " << region_index_str << " union " << circle_region_names.size();
                    for (const auto& circle_name : circle_region_names) {
                        union_cmd << " " << circle_name;
                    }
                    execute_command(sparta_ptr, union_cmd.str());
                }
                
                // Define group for region
                std::string cmd_group = "group group_region_" + region_index_str + " grid region " + region_index_str + " one";
                execute_command(sparta_ptr, cmd_group);

                std::string cmd_surf = "group surf_region_" + region_index_str + " surf region " + region_index_str + " one";
                execute_command(sparta_ptr, cmd_surf);

                // Output surface data for region
                if (do_surf) {
                    std::string surf_properties = "n nflux nflux_incident mflux mflux_incident fx fy press px py shx shy ke etot";
                    std::string cmd_surf_data = "compute surf_data_region_" + region_index_str + " surf surf_region_" + region_index_str + " Ar " + surf_properties;
                    execute_command(sparta_ptr, cmd_surf_data);
                    
                    std::string cmd_fix_surf_data = "fix surf_data_region_" + region_index_str + " ave/surf surf_region_" + region_index_str + " 1 100 5000 c_surf_data_region_" + region_index_str + "[*]";
                    execute_command(sparta_ptr, cmd_fix_surf_data);

                    std::string cmd_write_surf_date = "dump dump_surf_region_" + region_index_str + " surf surf_region_" + region_index_str + " 5000 ${RUN}/${SAMPLE}/${OUT}_region" + region_index_str + "surf.* id f_surf_data_region_" + region_index_str + "[*]";
                    execute_command(sparta_ptr, cmd_write_surf_date);
                }

                // Output grid data or region
                if (do_grid) {
                    std::string grid_properties = "n nrho mass massrho u v ke temp pxrho pyrho kerho";
                    std::string cmd_grid_data = "compute grid_data_region_" + region_index_str + " grid group_region_" + region_index_str + " Ar " + grid_properties;
                    execute_command(sparta_ptr, cmd_grid_data);
                
                    std::string cmd_fix_grid_data = "fix grid_data_region_" + region_index_str + " ave/grid group_region_" + region_index_str + " 1 100 5000 c_grid_data_region_" + region_index_str + "[*]";
                    execute_command(sparta_ptr, cmd_fix_grid_data);
                }

                // Thermal Properties
                if (do_therm) {
                    std::string grid_thermal_properties = "temp press";
                    std::string cmd_grid_thermal_data = "compute grid_thermal_data_region_" + region_index_str + " thermal/grid group_region_" + region_index_str + " Ar " + grid_thermal_properties;
                    execute_command(sparta_ptr, cmd_grid_thermal_data);

                    std::string cmd_fix_grid_thermal_data = "fix grid_thermal_data_region_" + region_index_str + " ave/grid group_region_" + region_index_str + " 1 100 5000 c_grid_thermal_data_region_" + region_index_str + "[*]";
                    execute_command(sparta_ptr, cmd_fix_grid_thermal_data);

                    std::string cmd_write_grid_date = "dump dump_grid_region_" + region_index_str + " grid group_region_" + region_index_str + " 5000 ${RUN}/${SAMPLE}/${OUT}_region" + region_index_str + "grid.* id f_grid_data_region_" + region_index_str + "[*] f_grid_thermal_data_region_" + region_index_str + "[*]";
                    execute_command(sparta_ptr, cmd_write_grid_date);
                }

                // Define transformed particle velocity variables in tangent/normal coordinates
                std::string cmd_var_t = "variable " + var_t_name + " particle \"vx*cos(" + region_theta_str + ") + vy*sin(" + region_theta_str + ")\"";
                execute_command(sparta_ptr, cmd_var_t);

                std::string cmd_var_n = "variable " + var_n_name + " particle \"-vx*sin(" + region_theta_str + ") + vy*cos(" + region_theta_str + ")\"";
                execute_command(sparta_ptr, cmd_var_n);
                
                // Execute histogram command for tangential velocity
                std::string cmd_t = "fix " + region_index_str + "_t ave/histo 1 100 100 -50000.0 50000.0 5000 v_" + var_t_name + " ave one beyond extra mode vector file ${RUN}/${SAMPLE}/" + region_index_str + "_t.histo region " + region_index_str;
                // execute_command(sparta_ptr, cmd_t);
                
                // Execute histogram command for normal velocity
                std::string cmd_n = "fix " + region_index_str + "_n ave/histo 1 100 100 -50000.0 50000.0 5000 v_" + var_n_name + " ave one beyond extra mode vector file ${RUN}/${SAMPLE}/" + region_index_str + "_n.histo region " + region_index_str;
                // execute_command(sparta_ptr, cmd_n);

                // 2D Histogram
                if (do_2D) {
                    std::string cmd_2D = "fix " + region_index_str + "_2D ave/histo_nd " + std::to_string(Nevery_2D) + 
                    " " + std::to_string(Nrepeat_2D) + " " + std::to_string(Nfreq_2D) + " " + std::to_string(lo_2D) + 
                    " " + std::to_string(hi_2D) + " " + std::to_string(nbin_2D) + " 2 v_" + var_t_name + " v_" + var_n_name + 
                    " ave one beyond extra mode vector file ${RUN}/${SAMPLE}/" + region_index_str + "_2D.histo region " + region_index_str;
                    execute_command(sparta_ptr, cmd_2D);
                }

                // 3D Histogram
                if (do_3D) {
                    std::string cmd_3D = "fix " + region_index_str + "_3D ave/histo_nd " + std::to_string(Nevery_3D) + 
                    " " + std::to_string(Nrepeat_3D) + " " + std::to_string(Nfreq_3D) + " " + std::to_string(lo_3D) + 
                    " " + std::to_string(hi_3D) + " " + std::to_string(nbin_3D) + " 3 v_" + var_t_name + " v_" + var_n_name + 
                    " vz ave one beyond extra mode vector file ${RUN}/${SAMPLE}/" + region_index_str + "_3D.histo region " + region_index_str;
                    execute_command(sparta_ptr, cmd_3D);
                }
            }

            int remaining_timesteps = total_timesteps - predata_timesteps;
            std::string cmd_run = "run " + std::to_string(remaining_timesteps);
            execute_command(sparta_ptr, cmd_run);

            sparta_close(sparta_ptr);
        }
        catch (...)
        {
            std::cerr << "Error thrown by Sparta during run." << std::endl;
        }

        // // Write out grid file for post processing
        // if (localRank == 0 && to_paraview)
        // {
        //     write_grid_txt(
        //         std::stoi(dim),
        //         std::stod(x_min),
        //         std::stod(x_max),
        //         std::stod(y_min),
        //         std::stod(y_max),
        //         std::stod(z_min),
        //         std::stod(z_max),
        //         run_name_i); 
        // }

        if (localRank == 0)
        {
            std::filesystem::path surf_path(surf_file);
            std::filesystem::path metadata_path(metadata_file);
            std::filesystem::path param_path = sample_dir / "sample.param";

            const double Tinf = 200.0;

            auto now = std::chrono::system_clock::now();
            std::time_t now_time = std::chrono::system_clock::to_time_t(now);
            std::tm local_tm = *std::localtime(&now_time);

            std::ofstream param_out(param_path);
            if (param_out.is_open()) {
                param_out << "sample_index = " << i << "\n";
                param_out << "mach = " << Mi << "\n";
                param_out << "Tinf = " << Tinf << "\n";
                param_out << "shape_file_name = " << surf_path.filename().string() << "\n";
                param_out << "shape_file_source_dir = " << surf_path.parent_path().string() << "\n";
                param_out << "shape_metadata_file_name = " << metadata_path.filename().string() << "\n";
                param_out << "shape_metadata_source_dir = " << metadata_path.parent_path().string() << "\n";
                param_out << "created_at = " << std::put_time(&local_tm, "%Y-%m-%d %H:%M:%S %Z") << "\n";
            } else {
                std::cerr << "Warning: could not write sample param file " << param_path << std::endl;
            }
        }

        
    }

    MPI_Finalize();
    return 0;
}