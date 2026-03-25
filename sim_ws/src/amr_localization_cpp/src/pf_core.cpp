#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/eigen/dense.h>

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <optional>

namespace nb = nanobind;
using namespace nb::literals;

struct Point2D
{
    double x;
    double y;
};

using Segment2D = std::array<Point2D, 2>;

class ParticleFilterCore
{
public:
    ParticleFilterCore(
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
        bool simulation = false)
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

        nb::module_ maps_mod = nb::module_::import_("amr_localization_cpp.maps");
        nb::object map_cls = maps_mod.attr("Map");
        map_obj_ = map_cls(
            map_path,
            sensor_range_max_,
            0.08,
            false,
            true);

        particles_ = init_particles(
            particle_count_,
            global_localization,
            initial_pose,
            initial_pose_sigma);
    }

    void move(double v, double w)
    {
        iteration_++;

        std::normal_distribution<double> noise_v(0.0, sigma_v_);
        std::normal_distribution<double> noise_w(0.0, sigma_w_);

        Eigen::MatrixXd old_positions = particles_.leftCols(2);

        for (int i = 0; i < particles_.rows(); ++i)
        {
            const double dv = (v + noise_v(rng_)) * dt_;
            const double dtheta = (w + noise_w(rng_)) * dt_;
            const double theta = particles_(i, 2);

            particles_(i, 0) += std::cos(theta + dtheta / 2.0) * dv;
            particles_(i, 1) += std::sin(theta + dtheta / 2.0) * dv;
            particles_(i, 2) += dtheta;

            particles_(i, 2) = std::fmod(particles_(i, 2), 2.0 * M_PI);
            if (particles_(i, 2) < 0.0)
                particles_(i, 2) += 2.0 * M_PI;

            auto hit = check_collision(
                {Point2D{old_positions(i, 0), old_positions(i, 1)},
                 Point2D{particles_(i, 0), particles_(i, 1)}},
                false);

            if (std::get<0>(hit).has_value())
            {
                particles_(i, 0) = std::get<0>(hit)->x;
                particles_(i, 1) = std::get<0>(hit)->y;
            }
        }
    }

    void resample(const std::vector<float> &measurements)
    {
        std::vector<double> weights;
        weights.reserve(static_cast<size_t>(particles_.rows()));

        for (int i = 0; i < particles_.rows(); ++i)
        {
            std::array<double, 3> p{particles_(i, 0), particles_(i, 1), particles_(i, 2)};
            weights.push_back(measurement_probability(measurements, p));
        }

        std::vector<double> bins(weights.size(), 0.0);
        std::partial_sum(weights.begin(), weights.end(), bins.begin());

        if (weights.empty())
            return;

        double total_weight = bins.back();
        if (total_weight <= 0.0 || !std::isfinite(total_weight))
        {
            std::fill(weights.begin(), weights.end(), 1.0);
            std::partial_sum(weights.begin(), weights.end(), bins.begin());
            total_weight = bins.back();
        }

        std::uniform_real_distribution<double> dist_u0(0.0, total_weight / particle_count_);
        const double u0 = dist_u0(rng_);

        Eigen::MatrixXd new_particles(particle_count_, 3);
        size_t idx = 0;

        for (int j = 0; j < particle_count_; ++j)
        {
            double u = u0 + j * total_weight / particle_count_;
            while (idx < bins.size() && bins[idx] < u)
                ++idx;
            if (idx >= static_cast<size_t>(particles_.rows()))
                idx = static_cast<size_t>(particles_.rows() - 1);
            new_particles.row(j) = particles_.row(static_cast<Eigen::Index>(idx));
        }

        particles_ = std::move(new_particles);
    }

    std::tuple<bool, std::tuple<double, double, double>> compute_pose()
    {
        bool localized = false;
        std::tuple<double, double, double> pose{
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity()};

        const int n = static_cast<int>(particles_.rows());
        if (n == 0)
            return {localized, pose};

        constexpr double eps = 0.2;
        constexpr double eps2 = eps * eps;
        constexpr int min_samples = 5;

        std::vector<int> labels(n, -99);
        int next_cluster_id = 0;

        auto region_query = [&](int i)
        {
            std::vector<int> neighbors;
            neighbors.reserve(n);
            for (int j = 0; j < n; ++j)
            {
                double dx = particles_(i, 0) - particles_(j, 0);
                double dy = particles_(i, 1) - particles_(j, 1);
                if (dx * dx + dy * dy <= eps2)
                    neighbors.push_back(j);
            }
            return neighbors;
        };

        for (int i = 0; i < n; ++i)
        {
            if (labels[i] != -99)
                continue;

            auto neighbors = region_query(i);
            if (static_cast<int>(neighbors.size()) < min_samples)
            {
                labels[i] = -1;
                continue;
            }

            labels[i] = next_cluster_id;
            std::queue<int> q;
            for (int nb : neighbors)
                q.push(nb);

            while (!q.empty())
            {
                int j = q.front();
                q.pop();

                if (labels[j] == -1)
                    labels[j] = next_cluster_id;

                if (labels[j] != -99)
                    continue;

                labels[j] = next_cluster_id;
                auto nb_neighbors = region_query(j);

                if (static_cast<int>(nb_neighbors.size()) >= min_samples)
                {
                    for (int nb2 : nb_neighbors)
                        q.push(nb2);
                }
            }

            next_cluster_id++;
        }

        std::vector<int> valid_labels;
        for (int id = 0; id < next_cluster_id; ++id)
            valid_labels.push_back(id);

        auto mean_pose_of_cluster = [&](int target_label)
        {
            double sum_x = 0.0, sum_y = 0.0, sum_sin = 0.0, sum_cos = 0.0;
            int count = 0;

            for (int i = 0; i < n; ++i)
            {
                if (labels[i] == target_label)
                {
                    sum_x += particles_(i, 0);
                    sum_y += particles_(i, 1);
                    sum_sin += std::sin(particles_(i, 2));
                    sum_cos += std::cos(particles_(i, 2));
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

            if (particles_.rows() > 50)
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
                particle_count_ = std::max(50, static_cast<int>(particle_count_ * 0.8));

            std::unordered_map<int, int> counts;
            for (int lab : labels)
            {
                if (lab != -1)
                    counts[lab]++;
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
                pose = mean_pose_of_cluster(max_cluster);

            n_prev_clusters_ = static_cast<int>(valid_labels.size());
        }

        return {localized, pose};
    }

    Eigen::MatrixXd particles() const
    {
        return particles_;
    }

private:
    Eigen::MatrixXd init_particles(
        int particle_count,
        bool global_localization,
        const std::tuple<double, double, double> &initial_pose,
        const std::tuple<double, double, double> &initial_pose_sigma)
    {

        auto bounds = nb::cast<std::tuple<double, double, double, double>>(
            map_obj_.attr("bounds")());
        double x_min = std::get<0>(bounds);
        double y_min = std::get<1>(bounds);
        double x_max = std::get<2>(bounds);
        double y_max = std::get<3>(bounds);

        std::uniform_real_distribution<double> dist_x(x_min, x_max);
        std::uniform_real_distribution<double> dist_y(y_min, y_max);
        std::normal_distribution<double> norm_x(std::get<0>(initial_pose), std::get<0>(initial_pose_sigma));
        std::normal_distribution<double> norm_y(std::get<1>(initial_pose), std::get<1>(initial_pose_sigma));

        const std::array<double, 4> angles{0.0, M_PI / 2.0, M_PI, 3.0 * M_PI / 2.0};
        std::uniform_int_distribution<int> angle_idx(0, 3);

        Eigen::MatrixXd out(particle_count, 3);
        int count = 0;

        while (count < particle_count)
        {
            int needed = particle_count - count;
            int batch_size = std::max(1, static_cast<int>(needed * 2.0));

            for (int i = 0; i < batch_size && count < particle_count; ++i)
            {
                double x = global_localization ? dist_x(rng_) : norm_x(rng_);
                double y = global_localization ? dist_y(rng_) : norm_y(rng_);
                bool inside = nb::cast<bool>(
                    map_obj_.attr("contains")(std::make_tuple(x, y)));
                if (inside)
                {
                    out(count, 0) = x;
                    out(count, 1) = y;
                    out(count, 2) = angles[angle_idx(rng_)];
                    ++count;
                }
            }
        }

        return out;
    }

    std::vector<Segment2D> lidar_rays(const std::array<double, 3> &pose,
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

    std::tuple<std::optional<Point2D>, double> check_collision(const Segment2D &segment,
                                                               bool compute_distance) const
    {
        nb::list seg;
        seg.append(std::make_tuple(segment[0].x, segment[0].y));
        seg.append(std::make_tuple(segment[1].x, segment[1].y));

        auto result = map_obj_.attr("check_collision")(seg, compute_distance);
        nb::tuple tup = nb::cast<nb::tuple>(result);

        nb::object py_intersection = tup[0];
        double dist = nb::cast<double>(tup[1]);

        if (py_intersection.is_none())
            return {std::nullopt, dist};

        if (nb::isinstance<nb::list>(py_intersection) && nb::len(py_intersection) == 0)
            return {std::nullopt, dist};

        auto pt = nb::cast<std::tuple<double, double>>(py_intersection);
        return {Point2D{std::get<0>(pt), std::get<1>(pt)}, dist};
    }

    std::vector<double> sense(const std::array<double, 3> &pose) const
    {
        constexpr int n = 16;
        std::vector<int> indices(n);
        std::iota(indices.begin(), indices.end(), 0);

        auto rays = lidar_rays(pose, indices, 360.0 / n);
        std::vector<double> z_hat;
        z_hat.reserve(n);

        for (const auto &ray : rays)
        {
            auto hit = check_collision(ray, true);
            z_hat.push_back(std::get<1>(hit));
        }

        return z_hat;
    }

    static double gaussian(double mu, double sigma, double x)
    {
        return 1.0 / (std::sqrt(2.0 * M_PI) * sigma) *
               std::exp(-((x - mu) * (x - mu)) / (2.0 * sigma * sigma));
    }

    double measurement_probability(const std::vector<float> &measurements,
                                   const std::array<double, 3> &particle) const
    {
        double probability = 1.0;
        constexpr int n = 16;

        if (measurements.empty())
            return probability;

        std::vector<double> fixed_measurements;
        fixed_measurements.reserve(measurements.size());

        for (float m : measurements)
        {
            double md = static_cast<double>(m);
            if (std::isnan(md) || (std::isinf(md) && md < 0.0))
                md = sensor_range_min_;
            else if (std::isinf(md) && md > 0.0)
                md = sensor_range_max_;
            fixed_measurements.push_back(md);
        }

        auto z_hat = sense(particle);
        for (double &z : z_hat)
        {
            if (std::isnan(z))
                z = sensor_range_max_;
            else if (std::isinf(z) && z > 0.0)
                z = sensor_range_max_;
            else if (std::isinf(z) && z < 0.0)
                z = sensor_range_min_;
        }

        const int step = std::max(1, static_cast<int>(fixed_measurements.size()) / n);

        for (int i = 0; i < n; ++i)
        {
            const int idx = std::min(i * step, static_cast<int>(fixed_measurements.size()) - 1);
            probability *= gaussian(z_hat[i], sigma_z_, fixed_measurements[idx]);
        }

        return probability;
    }

private:
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

    nb::object map_obj_;
    Eigen::MatrixXd particles_;
    std::mt19937 rng_;
};

NB_MODULE(_pf_core, m)
{
    nb::class_<ParticleFilterCore>(m, "ParticleFilterCore")
        .def(nb::init<
                 double,
                 const std::string &,
                 int,
                 double,
                 double,
                 double,
                 double,
                 double,
                 bool,
                 std::tuple<double, double, double>,
                 std::tuple<double, double, double>,
                 bool>(),
             "dt"_a,
             "map_path"_a,
             "particle_count"_a,
             "sigma_v"_a = 0.05,
             "sigma_w"_a = 0.1,
             "sigma_z"_a = 0.2,
             "sensor_range_max"_a = 8.0,
             "sensor_range_min"_a = 0.16,
             "global_localization"_a = true,
             "initial_pose"_a = std::make_tuple(
                 std::numeric_limits<double>::quiet_NaN(),
                 std::numeric_limits<double>::quiet_NaN(),
                 std::numeric_limits<double>::quiet_NaN()),
             "initial_pose_sigma"_a = std::make_tuple(
                 std::numeric_limits<double>::quiet_NaN(),
                 std::numeric_limits<double>::quiet_NaN(),
                 std::numeric_limits<double>::quiet_NaN()),
             "simulation"_a = false)
        .def("move", &ParticleFilterCore::move)
        .def("resample", &ParticleFilterCore::resample)
        .def("compute_pose", &ParticleFilterCore::compute_pose)
        .def("particles", &ParticleFilterCore::particles);
}