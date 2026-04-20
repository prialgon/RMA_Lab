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
    static constexpr int kMeasurementBeams = 16;

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
                bool simulation = false,
                bool reset_on_low_likelihood = true,
                double low_likelihood_log_threshold = -6.0,
                int low_likelihood_patience = 3)
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
          n_prev_clusters_(particle_count),
          reset_on_low_likelihood_(reset_on_low_likelihood),
          low_likelihood_log_threshold_(low_likelihood_log_threshold),
          low_likelihood_patience_(std::max(1, low_likelihood_patience)),
          global_localization_(global_localization),
          initial_pose_(initial_pose),
          initial_pose_sigma_(initial_pose_sigma),
          low_likelihood_streak_(0),
          reset_happened_(false)
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

        nb::list py_segments = nb::cast<nb::list>(map_obj_.attr("_map_segments"));
        for (auto seg : py_segments)
        {
            nb::sequence pts = nb::cast<nb::sequence>(seg);
            nb::sequence p0 = nb::cast<nb::sequence>(pts[0]);
            nb::sequence p1 = nb::cast<nb::sequence>(pts[1]);
            map_segments_.push_back({
                Point2D{nb::cast<double>(p0[0]), nb::cast<double>(p0[1])},
                Point2D{nb::cast<double>(p1[0]), nb::cast<double>(p1[1])}
            });
        }

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
        reset_happened_ = false;

        std::vector<double> weights;
        std::vector<double> log_weights;
        weights.reserve(static_cast<size_t>(particles_.rows()));
        log_weights.reserve(static_cast<size_t>(particles_.rows()));

        double max_log_weight = -std::numeric_limits<double>::infinity();

        for (int i = 0; i < particles_.rows(); ++i)
        {
            std::array<double, 3> p{particles_(i, 0), particles_(i, 1), particles_(i, 2)};
            const double log_w = measurement_log_probability(measurements, p);
            log_weights.push_back(log_w);
            if (std::isfinite(log_w) && log_w > max_log_weight)
            {
                max_log_weight = log_w;
            }
        }

        if (log_weights.empty())
            return;

        if (!std::isfinite(max_log_weight))
        {
            max_log_weight = 0.0;
        }

        for (double log_w : log_weights)
        {
            if (std::isfinite(log_w))
            {
                weights.push_back(std::exp(log_w - max_log_weight));
            }
            else
            {
                weights.push_back(0.0);
            }
        }

        std::vector<double> bins(weights.size(), 0.0);
        std::partial_sum(weights.begin(), weights.end(), bins.begin());

        if (reset_on_low_likelihood_)
        {
            // Convert the best particle score into a relative score by
            // accounting for the number of particles (log-domain normalization).
            const double relative_log_likelihood_per_beam =
                (max_log_weight - std::log(static_cast<double>(std::max(1, particle_count_)))) /
                static_cast<double>(kMeasurementBeams);

            if (relative_log_likelihood_per_beam < low_likelihood_log_threshold_)
            {
                low_likelihood_streak_ += 1;
            }
            else
            {
                low_likelihood_streak_ = 0;
            }

            if (low_likelihood_streak_ >= low_likelihood_patience_)
            {
                reset_particles();
                low_likelihood_streak_ = 0;
                reset_happened_ = true;
                return;
            }
        }

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

    bool consume_reset_flag()
    {
        const bool reset = reset_happened_;
        reset_happened_ = false;
        return reset;
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

        Eigen::MatrixXd coords = particles_.leftCols(2);

        nb::module_ sklearn_cluster = nb::module_::import_("sklearn.cluster");
        nb::object dbscan = sklearn_cluster.attr("DBSCAN")("eps"_a = 0.2, "min_samples"_a = 5);

        nb::object fit_res = dbscan.attr("fit")(coords);
                nb::list py_labels = nb::cast<nb::list>(fit_res.attr("labels_").attr("tolist")());

        std::vector<int> labels;
        labels.reserve(n);
        for (auto item : py_labels)
        {
            labels.push_back(nb::cast<int>(item));
        }

        std::vector<int> valid_labels;
        for (int lab : labels)
        {
            if (lab != -1 && std::find(valid_labels.begin(), valid_labels.end(), lab) == valid_labels.end())
                valid_labels.push_back(lab);
        }

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

            localized = true;
            particle_count_ = 100;
        }
        else
        {
            if (static_cast<int>(valid_labels.size()) < n_prev_clusters_)
            {
                particle_count_ = std::max(100, static_cast<int>(valid_labels.size() * 200));
            }

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

    double compute_intercept(double slope, double x, double y) const {
        return y - slope * x;
    }

    double compute_slope(double x0, double y0, double x1, double y1) const {
        double slope = std::numeric_limits<double>::infinity();
        if (std::fabs(x1 - x0) >= 1e-6)
            slope = (y1 - y0) / (x1 - x0);
        return slope;
    }

    void intersect(
        double* xi, double* yi, 
        double x0, double y0, double x1, double y1, 
        double x2, double y2, double x3, double y3
    ) const {
        double m1 = compute_slope(x0, y0, x1, y1);
        double m2 = compute_slope(x2, y2, x3, y3);
        double b1, b2;
        *xi = std::numeric_limits<double>::quiet_NaN();
        *yi = std::numeric_limits<double>::quiet_NaN();

        if (!std::isinf(m1) && !std::isinf(m2)) {
            b1 = compute_intercept(m1, x0, y0);
            b2 = compute_intercept(m2, x2, y2);

            if (std::fabs(m1 - m2) >= 1e-6) {
                *xi = (b2 - b1) / (m1 - m2);
            }

            *yi = m1 * *xi + b1;
        }
        else if (std::isinf(m1) && !std::isinf(m2)) {
            b2 = compute_intercept(m2, x2, y2);
            *xi = x0;
            *yi = m2 * *xi + b2;
        }
        else if (std::isinf(m2) && !std::isinf(m1)) {
            b1 = compute_intercept(m1, x0, y0);
            *xi = x2;
            *yi = m1 * *xi + b1;
        }
    }

    double round6(double x) const {
        return std::round(x * 1e6) / 1e6; 
    }

    bool segment_intersect(
        double* xi, double* yi, 
        double x0, double y0, double x1, double y1, 
        double x2, double y2, double x3, double y3
    ) const {
        intersect(xi, yi, x0, y0, x1, y1, x2, y2, x3, y3);

        if (std::isnan(*xi) || std::isnan(*yi))
            return false;

        double xr0 = round6(x0);  
        double xr1 = round6(x1);
        double xr2 = round6(x2);
        double xr3 = round6(x3);
        double xri = round6(*xi);
        double yr0 = round6(y0);  
        double yr1 = round6(y1);
        double yr2 = round6(y2);
        double yr3 = round6(y3);
        double yri = round6(*yi);

        if (xr0 < xr1) {
            if (xri < xr0 || xri > xr1) return false;
        } else {
            if (xri > xr0 || xri < xr1) return false;
        }

        if (yr0 < yr1) {
            if (yri < yr0 || yri > yr1) return false;
        } else {
            if (yri > yr0 || yri < yr1) return false;
        }

        if (xr2 < xr3) {
            if (xri < xr2 || xri > xr3) return false;
        } else {
            if (xri > xr2 || xri < xr3) return false;
        }

        if (yr2 < yr3) {
            if (yri < yr2 || yri > yr3) return false;
        } else {
            if (yri > yr2 || yri < yr3) return false;
        }

        return true;
    }

    std::tuple<std::optional<Point2D>, double> check_collision(const Segment2D &segment,
                                                               bool compute_distance) const
    {
        std::vector<Point2D> intersections;
        
        for (const auto& map_seg : map_segments_) {
            double xi = std::numeric_limits<double>::quiet_NaN();
            double yi = std::numeric_limits<double>::quiet_NaN();
            if (segment_intersect(&xi, &yi,
                segment[0].x, segment[0].y, segment[1].x, segment[1].y,
                map_seg[0].x, map_seg[0].y, map_seg[1].x, map_seg[1].y)) {
                intersections.push_back({xi, yi});
            }
        }

        double distance = std::numeric_limits<double>::quiet_NaN();
        std::optional<Point2D> closest_intersection = std::nullopt;

        if (!intersections.empty()) {
            if (compute_distance || intersections.size() > 1) {
                double min_dist = std::numeric_limits<double>::infinity();
                for (const auto& pt : intersections) {
                    double dist = std::hypot(pt.x - segment[0].x, pt.y - segment[0].y);
                    if (dist < min_dist) {
                        min_dist = dist;
                        closest_intersection = pt;
                    }
                }
                distance = min_dist;
            } else {
                closest_intersection = intersections[0];
            }
        }

        return {closest_intersection, distance};
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

    double gaussian_log_probability(double mu, double sigma, double x) const
    {
        const double s = std::max(sigma, 1e-9);
        const double delta = x - mu;
        return -std::log(std::sqrt(2.0 * M_PI) * s) - (delta * delta) / (2.0 * s * s);
    }

    double measurement_log_probability(const std::vector<float> &measurements,
                                       const std::array<double, 3> &particle) const
    {
        double log_probability = 0.0;

        if (measurements.empty())
            return log_probability;

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

        const int step = std::max(1, static_cast<int>(fixed_measurements.size()) / kMeasurementBeams);

        for (int i = 0; i < kMeasurementBeams; ++i)
        {
            const int idx = std::min(i * step, static_cast<int>(fixed_measurements.size()) - 1);
            log_probability += gaussian_log_probability(z_hat[i], sigma_z_, fixed_measurements[idx]);
        }

        return log_probability;
    }

    void reset_particles()
    {
        particle_count_ = initial_particle_count_;
        particles_ = init_particles(
            initial_particle_count_,
            true,
            initial_pose_,
            initial_pose_sigma_);
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
    bool reset_on_low_likelihood_;
    double low_likelihood_log_threshold_;
    int low_likelihood_patience_;
    bool global_localization_;
    std::tuple<double, double, double> initial_pose_;
    std::tuple<double, double, double> initial_pose_sigma_;
    int low_likelihood_streak_;
    bool reset_happened_;

    nb::object map_obj_;
    std::vector<Segment2D> map_segments_;
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
                 bool,
                 bool,
                 double,
                 int>(),
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
             "simulation"_a = false,
             "reset_on_low_likelihood"_a = true,
             "low_likelihood_log_threshold"_a = -6.0,
             "low_likelihood_patience"_a = 3)
        .def("move", &ParticleFilterCore::move)
        .def("resample", &ParticleFilterCore::resample)
        .def("compute_pose", &ParticleFilterCore::compute_pose)
        .def("consume_reset_flag", &ParticleFilterCore::consume_reset_flag)
        .def("particles", &ParticleFilterCore::particles);
}