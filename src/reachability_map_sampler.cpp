#include <memory>
#include <chrono>
#include <random>

#include "rclcpp/rclcpp.hpp"
#include "edi_robot_msgs/srv/reachability_map_sample_array.hpp"

#include "reachability_map_visualizer/hdf5_dataset.h"

using namespace hdf5_dataset;

void buildReachabilityMap(
  std::map<std::array<float, 3>, std::vector<std::array<double, 7>>> &reachability_map, 
  MultiMapPtr pose_col_filter, 
  MapVecDoublePtr sphere_col)
{

  float min_x = std::numeric_limits<float>::max();
  float min_y = std::numeric_limits<float>::max();
  float min_z = std::numeric_limits<float>::max();
  float max_x = std::numeric_limits<float>::lowest();
  float max_y = std::numeric_limits<float>::lowest();
  float max_z = std::numeric_limits<float>::lowest();
  
  for (const auto& sphere_pair : sphere_col)
  {
    std::array<float, 3> key = {
      (*sphere_pair.first)[0],
      (*sphere_pair.first)[1],
      (*sphere_pair.first)[2]
    };

    // Update bounds
    min_x = std::min(min_x, key[0]);
    min_y = std::min(min_y, key[1]);
    min_z = std::min(min_z, key[2]);
    max_x = std::max(max_x, key[0]);
    max_y = std::max(max_y, key[1]);
    max_z = std::max(max_z, key[2]);

    std::vector<std::array<double, 7>> poses;
    for (auto it1 = pose_col_filter.lower_bound(sphere_pair.first);
         it1 != pose_col_filter.upper_bound(sphere_pair.first); ++it1)
    {
      poses.push_back({
        (*it1->second)[0], (*it1->second)[1], (*it1->second)[2],
        (*it1->second)[3], (*it1->second)[4], (*it1->second)[5], (*it1->second)[6]
      });
    }

    reachability_map[key] = std::move(poses);
  }

  std::cout << "Reachability map bounds:\n";
  std::cout << "  X: [" << min_x << ", " << max_x << "]\n";
  std::cout << "  Y: [" << min_y << ", " << max_y << "]\n";
  std::cout << "  Z: [" << min_z << ", " << max_z << "]\n";
}

// std::vector<std::vector<double>> samplePose(
geometry_msgs::msg::PoseArray samplePose(
  std::map<std::array<float, 3>, std::vector<std::array<double, 7>>> &reachability_map,
  // const std::array<double, 3>& target,
  geometry_msgs::msg::Pose target,
  float radius,
  int n_samples
)
{
  static std::mt19937 gen(std::random_device{}());

  // apply the region constraint
  std::vector<std::array<float, 3>> filtered_spheres;
  for (const auto& [center, poses] : reachability_map)
  {
    float dx = center[0] - target.position.x;
    float dy = center[1] - target.position.y;
    float dz = center[2] - target.position.z;
    float dist = std::sqrt(dx*dx + dy*dy + dz*dz);

    if (dist <= radius && !poses.empty())
      filtered_spheres.push_back(center);
  }

  RCLCPP_INFO(rclcpp::get_logger("load_reachability_map"),
          "Filtered sphere count: %zu", filtered_spheres.size());

  // check if the map has values
  if (filtered_spheres.empty())
  {
    throw std::runtime_error("No reachable spheres found within radius of target point.");
  }

  // --- Compute bounds of filtered spheres ---
  float min_x = std::numeric_limits<float>::max();
  float min_y = std::numeric_limits<float>::max();
  float min_z = std::numeric_limits<float>::max();
  float max_x = std::numeric_limits<float>::lowest();
  float max_y = std::numeric_limits<float>::lowest();
  float max_z = std::numeric_limits<float>::lowest();

  for (const auto& c : filtered_spheres)
  {
    min_x = std::min(min_x, c[0]);
    min_y = std::min(min_y, c[1]);
    min_z = std::min(min_z, c[2]);
    max_x = std::max(max_x, c[0]);
    max_y = std::max(max_y, c[1]);
    max_z = std::max(max_z, c[2]);
  }

  min_x = 0.0;
  min_y = -0.5;
  min_z = 0.6;
  max_x = 0.6;
  max_y = 0.5;
  max_z = 1.2;

  std::cout << "Filtered reachability map bounds:\n";
  std::cout << "  X: [" << min_x << ", " << max_x << "]\n";
  std::cout << "  Y: [" << min_y << ", " << max_y << "]\n";
  std::cout << "  Z: [" << min_z << ", " << max_z << "]\n";

  // --- Spatially uniform random point inside bounds ---
  std::uniform_real_distribution<float> dist_x(min_x, max_x);
  std::uniform_real_distribution<float> dist_y(min_y, max_y);
  std::uniform_real_distribution<float> dist_z(min_z, max_z);

  geometry_msgs::msg::PoseArray sampled_poses;
  sampled_poses.header.stamp = rclcpp::Clock().now();
  sampled_poses.header.frame_id = "base_footprint";

  for (int i = 0; i < n_samples && !filtered_spheres.empty(); ++i)
  {

    std::array<float, 3> random_point = { dist_x(gen), dist_y(gen), dist_z(gen) };

    // --- Find nearest sphere to the random point ---
    auto nearest_it = std::min_element(filtered_spheres.begin(), filtered_spheres.end(),
    [&](const std::array<float,3>& a, const std::array<float,3>& b) 
    {
      float da = std::pow(a[0]-random_point[0],2) +
                std::pow(a[1]-random_point[1],2) +
                std::pow(a[2]-random_point[2],2);
      float db = std::pow(b[0]-random_point[0],2) +
                std::pow(b[1]-random_point[1],2) +
                std::pow(b[2]-random_point[2],2);
      return da < db;
    });

    geometry_msgs::msg::Pose pose;
    pose.position.x = (*nearest_it)[0];
    pose.position.y = (*nearest_it)[1];
    pose.position.z = (*nearest_it)[2];
    pose.orientation.w = 1.0;

    // Add to result
    sampled_poses.poses.push_back(pose);

    // Remove used sphere
    filtered_spheres.erase(nearest_it);
  }

  RCLCPP_INFO(rclcpp::get_logger("load_reachability_map"), "Sampled poses:");
  for (size_t i = 0; i < sampled_poses.poses.size(); ++i)
  {
      const auto& p = sampled_poses.poses[i].position;
      RCLCPP_INFO(rclcpp::get_logger("load_reachability_map"),
                  "  Pose %zu: [%.3f, %.3f, %.3f]", 
                  i+1, p.x, p.y, p.z);
  }

  return sampled_poses;
}

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);

  if (argc < 2)
  {
    RCLCPP_ERROR(rclcpp::get_logger("load_reachability_map"), 
      "Please provide the name of the reachability map.");
    return 1;
  }

  // Load the reachability map
  hdf5_dataset::Hdf5Dataset h5(argv[1]);
  h5.open();

  MultiMapPtr pose_col_filter;
  MapVecDoublePtr sphere_col;
  float res;
  h5.loadMapsFromDataset(pose_col_filter, sphere_col, res);

  RCLCPP_INFO(rclcpp::get_logger("load_reachability_map"), "Map loading complete");

  geometry_msgs::msg::Pose target;
  target.position.x = 1.2;
  target.position.x = 0.0;
  target.position.x = 0.3;
  float radius = 1.0;
  std::map<std::array<float, 3>, std::vector<std::array<double, 7>>> reachability_map;

  buildReachabilityMap(reachability_map, pose_col_filter, sphere_col);

  std::shared_ptr<rclcpp::Node> node = rclcpp::Node::make_shared("reachability_map_sampler_server");

  auto service = node->create_service<edi_robot_msgs::srv::ReachabilityMapSampleArray>(
  "reachability_map_sampler",
  [&reachability_map](const std::shared_ptr<edi_robot_msgs::srv::ReachabilityMapSampleArray::Request> request,
                      std::shared_ptr<edi_robot_msgs::srv::ReachabilityMapSampleArray::Response> response)
  {
    geometry_msgs::msg::Pose target = request->target;
    float radius = request->sampling_radius;
    int n_samples = request->n_samples;

    response->poses = samplePose(reachability_map, target, radius, n_samples);
  });

	RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Ready to send object poses.");

  rclcpp::spin(node);
  rclcpp::shutdown();
}