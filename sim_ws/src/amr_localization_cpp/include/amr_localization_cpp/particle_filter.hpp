#pragma once

#include "amr_localization_cpp/py_map_bridge.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace amr_localization_cpp
{

    class ParticleFilter
    {
    public:
        ParticleFilter(
            double dt,
            const std::string &map_path,
            int particle_count,
            double sigma_v = 0.05,
            double sigma_w = 0.1,
            double sigma_z = 0.2,
            double sensor_range_max = 8.0,
            double sensor_range_min = 0.16,
            bool global_localization = true,
            std::tuple<double, double, double> initial_pose =
                {std::numeric_limits<double>::quiet_NaN(),
                 std::numeric_limits<double>::quiet_NaN(),
                 std::numeric_limits<double>::quiet_NaN()},
            std::tuple<double, double, double> initial_pose_sigma =
                {std::numeric_limits<double>::quiet_NaN(),
                 std::numeric_limits<double>::quiet_NaN(),
                 std::numeric_limits<double>::quiet_NaN()},
            bool simulation = false);

        std::tuple<bool, std::tuple<double, double, double>> compute_pose();

        void move(double v, double w);

        void resample(const std::vector<float> &measurements);

        void show(
            const std::string &title = "",
            bool orientation = true,
            bool display = false,
            bool block = false,
            bool save_figure = false,
            const std::string &save_dir = "images");

    private:
        using Particle = std::array<double, 3>; // x, y, theta
        using Particles = std::vector<Particle>;

        Particles init_particles(
            int particle_count,
            bool global_localization,
            const std::tuple<double, double, double> &initial_pose,
            const std::tuple<double, double, double> &initial_pose_sigma);

        std::vector<double> sense(const Particle &pose) const;

        static double gaussian(double mu, double sigma, double x);

        std::vector<Segment2D> lidar_rays(
            const Particle &pose,
            const std::vector<int> &indices,
            double degree_increment = 1.5) const;

        double measurement_probability(
            const std::vector<float> &measurements,
            const Particle &particle) const;

        double dt_;
        int initial_particle_count_;
        int particle_count_;
        double sensor_range_max_;
        double sensor_range_min_;
        double sigma_v_;
        double sigma_w_;
        double sigma_z_;
        bool simulation_;
        int iteration_;
        int n_prev_clusters_;

        std::unique_ptr<PyMapBridge> map_;
        Particles particles_;

        std::mt19937 rng_;
    };

} // namespace amr_localization_cpp