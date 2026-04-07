#include "amr_localization_cpp/particle_filter.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <unordered_map>

namespace amr_localization_cpp
{

    ParticleFilter::ParticleFilter(
        double dt,
        const std::string &map_path,
        int particle_count,
        double sigma_v,
        double sigma_w,
        double sigma_z,
        double sensor_range_max,
        double sensor_range_min,
        bool global_localization,
        std::tuple<double, double, double> initial_pose,
        std::tuple<double, double, double> initial_pose_sigma,
        bool simulation)
        : dt_(dt),
          initial_particle_count_(particle_count),
          particle_count_(particle_count),
          sensor_range_max_(sensor_range_max),
          sensor_range_min_(sensor_range_min),
          sigma_v_(sigma_v),
          sigma_w_(sigma_w),
          sigma_z_(sigma_z),
          simulation_(simulation),
          iteration_(0),
          n_prev_clusters_(particle_count)
    {
        std::random_device rd;
        rng_ = std::mt19937(rd());

        map_ = std::make_unique<PyMapBridge>(
            map_path,
            sensor_range_max_,
            0.08,
            false,
            true);

        particles_ = init_particles(
            particle_count,
            global_localization,
            initial_pose,
            initial_pose_sigma);
    }

    auto ParticleFilter::init_particles(
        int particle_count,
        bool global_localization,
        const std::tuple<double, double, double> &initial_pose,
        const std::tuple<double, double, double> &initial_pose_sigma) -> Particles
    {
        Particles particles;
        particles.reserve(static_cast<size_t>(particle_count));

        auto [x_min, y_min, x_max, y_max] = map_->bounds();

        std::uniform_real_distribution<double> dist_x(x_min, x_max);
        std::uniform_real_distribution<double> dist_y(y_min, y_max);

        std::normal_distribution<double> norm_x(
            std::get<0>(initial_pose),
            std::get<0>(initial_pose_sigma));
        std::normal_distribution<double> norm_y(
            std::get<1>(initial_pose),
            std::get<1>(initial_pose_sigma));

        const std::array<double, 4> discrete_angles{
            0.0, M_PI / 2.0, M_PI, 3.0 * M_PI / 2.0};
        std::uniform_int_distribution<int> angle_idx(0, 3);

        while (static_cast<int>(particles.size()) < particle_count)
        {
            const int needed = particle_count - static_cast<int>(particles.size());
            const int batch_size = std::max(1, static_cast<int>(needed * 2.0));

            for (int i = 0; i < batch_size && static_cast<int>(particles.size()) < particle_count; ++i)
            {
                const double x = global_localization ? dist_x(rng_) : norm_x(rng_);
                const double y = global_localization ? dist_y(rng_) : norm_y(rng_);

                if (map_->contains({x, y}))
                {
                    particles.push_back({x, y, discrete_angles[angle_idx(rng_)]});
                }
            }
        }

        return particles;
    }

    std::vector<Segment2D> ParticleFilter::lidar_rays(
        const Particle &pose,
        const std::vector<int> &indices,
        double degree_increment) const
    {
        const double x = pose[0];
        const double y = pose[1];
        const double theta = pose[2];

        const double x_start = x - 0.035 * std::cos(theta);
        const double y_start = y - 0.035 * std::sin(theta);

        std::vector<Segment2D> rays;
        rays.reserve(indices.size());

        for (int index : indices)
        {
            const double ray_angle = degree_increment * static_cast<double>(index) * M_PI / 180.0;
            const double x_end = x_start + sensor_range_max_ * std::cos(theta + ray_angle);
            const double y_end = y_start + sensor_range_max_ * std::sin(theta + ray_angle);
            rays.push_back({Point2D{x_start, y_start}, Point2D{x_end, y_end}});
        }

        return rays;
    }

    std::vector<double> ParticleFilter::sense(const Particle &pose) const
    {
        std::vector<double> z_hat;
        constexpr int n = 16;
        z_hat.reserve(n);

        std::vector<int> indices(n);
        std::iota(indices.begin(), indices.end(), 0);

        const auto rays = lidar_rays(pose, indices, 360.0 / static_cast<double>(n));

        for (const auto &ray : rays)
        {
            auto [intersection, dist] = map_->check_collision(ray, true);
            (void)intersection;
            z_hat.push_back(dist);
        }

        return z_hat;
    }

    double ParticleFilter::gaussian(double mu, double sigma, double x)
    {
        return 1.0 / (std::sqrt(2.0 * M_PI) * sigma) *
               std::exp(-((x - mu) * (x - mu)) / (2.0 * sigma * sigma));
    }

    double ParticleFilter::measurement_probability(
        const std::vector<float> &measurements,
        const Particle &particle) const
    {
        double probability = 1.0;
        constexpr int n = 16;

        if (measurements.empty())
        {
            return probability;
        }

        std::vector<double> fixed_measurements;
        fixed_measurements.reserve(measurements.size());

        for (float m : measurements)
        {
            double md = static_cast<double>(m);
            if (std::isnan(md) || (std::isinf(md) && md < 0.0))
            {
                md = sensor_range_min_;
            }
            else if (std::isinf(md) && md > 0.0)
            {
                md = sensor_range_max_;
            }
            fixed_measurements.push_back(md);
        }

        std::vector<double> z_hat = sense(particle);
        for (double &z : z_hat)
        {
            if (std::isnan(z))
            {
                z = sensor_range_max_;
            }
            else if (std::isinf(z) && z > 0.0)
            {
                z = sensor_range_max_;
            }
            else if (std::isinf(z) && z < 0.0)
            {
                z = sensor_range_min_;
            }
        }

        const int step = std::max(1, static_cast<int>(fixed_measurements.size()) / n);

        for (int i = 0; i < n; ++i)
        {
            const int idx = std::min(i * step, static_cast<int>(fixed_measurements.size()) - 1);
            const double z_real = fixed_measurements[idx];
            const double z_sensed = z_hat[i];
            probability *= gaussian(z_sensed, sigma_z_, z_real);
        }

        return probability;
    }

    void ParticleFilter::resample(const std::vector<float> &measurements)
    {
        std::vector<double> weights;
        weights.reserve(particles_.size());

        for (const auto &particle : particles_)
        {
            weights.push_back(measurement_probability(measurements, particle));
        }

        std::vector<double> bins(weights.size(), 0.0);
        std::partial_sum(weights.begin(), weights.end(), bins.begin());

        const double W = weights.empty() ? 0.0 : std::accumulate(weights.begin(), weights.end(), 0.0);

        if (particles_.empty())
        {
            return;
        }

        if (W <= 0.0 || !std::isfinite(W))
        {
            std::fill(weights.begin(), weights.end(), 1.0);
            std::partial_sum(weights.begin(), weights.end(), bins.begin());
        }

        const double total_weight = bins.back();
        std::uniform_real_distribution<double> dist_u0(
            0.0, total_weight / static_cast<double>(particle_count_));
        const double u0 = dist_u0(rng_);

        std::vector<double> u_values(static_cast<size_t>(particle_count_));
        for (int i = 0; i < particle_count_; ++i)
        {
            u_values[static_cast<size_t>(i)] =
                u0 + static_cast<double>(i) * total_weight / static_cast<double>(particle_count_);
        }

        Particles new_particles;
        new_particles.reserve(static_cast<size_t>(particle_count_));

        for (double u : u_values)
        {
            auto it = std::upper_bound(bins.begin(), bins.end(), u);
            size_t idx = static_cast<size_t>(std::distance(bins.begin(), it));
            if (idx >= particles_.size())
            {
                idx = particles_.size() - 1;
            }
            new_particles.push_back(particles_[idx]);
        }

        particles_ = std::move(new_particles);
    }

    void ParticleFilter::move(double v, double w)
    {
        iteration_++;

        std::normal_distribution<double> noise_v(0.0, sigma_v_);
        std::normal_distribution<double> noise_w(0.0, sigma_w_);

        std::vector<Point2D> old_positions;
        old_positions.reserve(particles_.size());

        for (const auto &p : particles_)
        {
            old_positions.push_back({p[0], p[1]});
        }

        for (size_t i = 0; i < particles_.size(); ++i)
        {
            auto &p = particles_[i];

            const double dv = (v + noise_v(rng_)) * dt_;
            const double dtheta = (w + noise_w(rng_)) * dt_;

            const double orientation = p[2];

            const double dx = std::cos(orientation + dtheta / 2.0) * dv;
            const double dy = std::sin(orientation + dtheta / 2.0) * dv;

            p[0] += dx;
            p[1] += dy;
            p[2] += dtheta;

            p[2] = std::fmod(p[2], 2.0 * M_PI);
            if (p[2] < 0.0)
            {
                p[2] += 2.0 * M_PI;
            }

            auto [intersection, dist] = map_->check_collision(
                {old_positions[i], {p[0], p[1]}},
                false);
            (void)dist;

            if (intersection.has_value())
            {
                p[0] = intersection->x;
                p[1] = intersection->y;
            }
        }
    }

    std::tuple<bool, std::tuple<double, double, double>> ParticleFilter::compute_pose()
    {
        bool localized = false;
        std::tuple<double, double, double> pose{
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity()};

        const int n = static_cast<int>(particles_.size());
        if (n == 0)
        {
            return {localized, pose};
        }

        // DBSCAN on x,y only, matching the Python logic
        constexpr double eps = 0.2;
        constexpr double eps2 = eps * eps;
        constexpr int min_samples = 5;

        std::vector<int> labels(n, -99); // unvisited
        int next_cluster_id = 0;

        auto region_query = [&](int i)
        {
            std::vector<int> neighbors;
            neighbors.reserve(n);
            for (int j = 0; j < n; ++j)
            {
                const double dx = particles_[i][0] - particles_[j][0];
                const double dy = particles_[i][1] - particles_[j][1];
                if (dx * dx + dy * dy <= eps2)
                {
                    neighbors.push_back(j);
                }
            }
            return neighbors;
        };

        for (int i = 0; i < n; ++i)
        {
            if (labels[i] != -99)
            {
                continue;
            }

            auto neighbors = region_query(i);
            if (static_cast<int>(neighbors.size()) < min_samples)
            {
                labels[i] = -1;
                continue;
            }

            labels[i] = next_cluster_id;
            std::queue<int> q;
            for (int nb : neighbors)
            {
                q.push(nb);
            }

            while (!q.empty())
            {
                const int j = q.front();
                q.pop();

                if (labels[j] == -1)
                {
                    labels[j] = next_cluster_id;
                }

                if (labels[j] != -99)
                {
                    continue;
                }

                labels[j] = next_cluster_id;

                auto nb_neighbors = region_query(j);
                if (static_cast<int>(nb_neighbors.size()) >= min_samples)
                {
                    for (int nb2 : nb_neighbors)
                    {
                        q.push(nb2);
                    }
                }
            }

            next_cluster_id++;
        }

        std::vector<int> valid_labels;
        for (int id = 0; id < next_cluster_id; ++id)
        {
            valid_labels.push_back(id);
        }

        auto mean_pose_of_cluster = [&](int target_label)
        {
            double sum_x = 0.0;
            double sum_y = 0.0;
            double sum_sin = 0.0;
            double sum_cos = 0.0;
            int count = 0;

            for (int i = 0; i < n; ++i)
            {
                if (labels[i] == target_label)
                {
                    sum_x += particles_[i][0];
                    sum_y += particles_[i][1];
                    sum_sin += std::sin(particles_[i][2]);
                    sum_cos += std::cos(particles_[i][2]);
                    count++;
                }
            }

            if (count == 0)
            {
                return std::tuple<double, double, double>{
                    std::numeric_limits<double>::infinity(),
                    std::numeric_limits<double>::infinity(),
                    std::numeric_limits<double>::infinity()};
            }

            return std::tuple<double, double, double>{
                sum_x / count,
                sum_y / count,
                std::atan2(sum_sin, sum_cos)};
        };

        if (valid_labels.size() == 1)
        {
            pose = mean_pose_of_cluster(valid_labels[0]);

            if (particles_.size() > 50)
            {
                particles_ = init_particles(
                    50,
                    false,
                    pose,
                    {0.05, 0.05, 5.0 * M_PI / 180.0});
            }

            localized = true;
            particle_count_ = 50;
        }
        else
        {
            if (static_cast<int>(valid_labels.size()) < n_prev_clusters_)
            {
                particle_count_ = std::max(50, static_cast<int>(particle_count_ * 0.8));
            }

            std::unordered_map<int, int> counts;
            for (int lab : labels)
            {
                if (lab != -1)
                {
                    counts[lab]++;
                }
            }

            int max_cluster = -1;
            int max_count = -1;
            for (const auto &[lab, count] : counts)
            {
                if (count > max_count)
                {
                    max_count = count;
                    max_cluster = lab;
                }
            }

            if (max_cluster != -1)
            {
                pose = mean_pose_of_cluster(max_cluster);
            }

            n_prev_clusters_ = static_cast<int>(valid_labels.size());
        }

        return {localized, pose};
    }

    void ParticleFilter::show(
        const std::string &title,
        bool orientation,
        bool display,
        bool block,
        bool save_figure,
        const std::string &save_dir)
    {
        (void)title;
        (void)orientation;
        (void)display;
        (void)block;
        (void)save_figure;
        (void)save_dir;
        // Intentionally left as a no-op in C++.
        // This preserves call sites without changing the filter logic.
    }

} // namespace amr_localization_cpp