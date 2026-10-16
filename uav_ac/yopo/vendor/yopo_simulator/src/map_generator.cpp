// Repository CLI around the pinned official map algorithms, without CUDA/ROS.
#include "maps.hpp"
#include <pcl/filters/voxel_grid.h>
#include <filesystem>
#include <stdexcept>

namespace fs = std::filesystem;

int main(int argc, char **argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cout << "Usage: map_generator <collection.yaml>\n";
        return 0;
    }
    if (argc != 2) {
        std::cerr << "Usage: map_generator <collection.yaml>\n";
        return 1;
    }
    try {
        const YAML::Node config = YAML::LoadFile(argv[1]);
        const fs::path output = config["save_path"].as<std::string>();
        const int count = config["env_num"].as<int>();
        const double resolution = config["resolution"].as<double>();
        const double ply_resolution = config["ply_res"].as<double>();
        if (count <= 0 || resolution <= 0 || ply_resolution <= 0)
            throw std::runtime_error("map count and resolutions must be positive");
        if (fs::exists(output))
            throw std::runtime_error("Refusing to overwrite existing output: " + output.string());
        fs::create_directories(output);
        for (int i = 0; i < count; ++i) {
            mocka::Maps::BasicInfo info;
            info.scale = 1.0 / resolution;
            info.sizeX = int(config["x_length"].as<int>() * info.scale);
            info.sizeY = int(config["y_length"].as<int>() * info.scale);
            info.sizeZ = int(config["z_length"].as<int>() * info.scale);
            info.seed = config["seed"].as<int>() + i;
            info.cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
            mocka::Maps map;
            map.setParam(config);
            map.setInfo(info);
            map.generate(config["maze_type"].as<int>());
            if (info.cloud->empty()) throw std::runtime_error("official map is empty");
            pcl::VoxelGrid<pcl::PointXYZ> filter;
            filter.setInputCloud(info.cloud);
            filter.setLeafSize(ply_resolution, ply_resolution, ply_resolution);
            pcl::PointCloud<pcl::PointXYZ> cloud;
            filter.filter(cloud);
            const auto path = output / ("pointcloud-" + std::to_string(i) + ".ply");
            if (pcl::io::savePLYFileBinary(path.string(), cloud) != 0)
                throw std::runtime_error("cannot save " + path.string());
            std::cout << path << " (" << cloud.size() << " points)\n";
        }
    } catch (const std::exception &error) {
        std::cerr << "Map export failed: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
