#pragma once

#include <array>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

namespace amr_localization_cpp
{

    struct Point2D
    {
        double x;
        double y;
    };

    using Segment2D = std::array<Point2D, 2>;

    class PyMapBridge
    {
    public:
        PyMapBridge(
            const std::string &map_path,
            double sensor_range,
            double safety_distance = 0.0,
            bool use_regions = true,
            bool compiled_intersect = true);

        ~PyMapBridge();

        std::tuple<double, double, double, double> bounds() const;
        bool contains(const Point2D &point) const;

        std::pair<std::optional<Point2D>, double> check_collision(
            const Segment2D &segment,
            bool compute_distance) const;

    private:
        class Impl;
        Impl *impl_;
    };

} // namespace amr_localization_cpp