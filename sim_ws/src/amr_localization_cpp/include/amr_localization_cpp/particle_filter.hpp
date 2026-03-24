#pragma once

#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "amr_localization_cpp/map.hpp"

namespace amr_localization_cpp
{

    struct Particle
    {
        double x;
        double y;
        double theta;
    };

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
            std::tuple<double, double, double> initial_pose = {0.0, 0.0, 0.0},
            std::tuple<double, double, double> initial_pose_sigma = {0.05, 0.05, 5.0 * M_PI / 180.0});

        std::tuple<bool, std::tuple<double, double, double>> compute_pose();
        void move(double v, double w);
        void resample(const std::vector<double> &measurements);

    private:
        std::vector<Particle> init_particles(
            int particle_count,
            bool global_localization,
            std::tuple<double, double, double> initial_pose,
            std::tuple<double, double, double> initial_pose_sigma);

        std::vector<double> sense(const Particle &p) const;
        std::vector<Segment2D> lidar_rays(
            const Particle &p,
            const std::vector<int> &indices,
            double degree_increment = 1.5) const;

        double measurement_probability(
            const std::vector<double> &measurements,
            const Particle &particle) const;

        static double gaussian(double mu, double sigma, double x);

    private:
        double dt_;
        int initial_particle_count_;
        int particle_count_;
        double sensor_range_max_;
        double sensor_range_min_;
        double sigma_v_;
        double sigma_w_;
        double sigma_z_;
        int iteration_;
        int n_prev_clusters_;

        std::unique_ptr<Map> map_;
        std::vector<Particle> particles_;
        mutable std::mt19937 rng_;
    };

} // namespace amr_localization_cpp