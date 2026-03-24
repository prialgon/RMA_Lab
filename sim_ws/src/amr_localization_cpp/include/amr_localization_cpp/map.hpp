#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace amr_localization_cpp
{

    struct Point2D
    {
        double x;
        double y;
    };

    struct Segment2D
    {
        Point2D p1;
        Point2D p2;
    };

    class Map
    {
    public:
        Map(
            const std::string &json_file,
            double sensor_range,
            double safety_distance = 0.0,
            bool use_regions = false);

        std::tuple<double, double, double, double> bounds() const;

        std::pair<std::optional<Point2D>, double> check_collision(
            const Segment2D &segment,
            bool compute_distance = false) const;

        bool contains(const Point2D &point) const;
        bool crosses(const Segment2D &segment) const;

        const std::vector<std::vector<int8_t>> &grid_map() const { return grid_map_; }

    private:
        void load_json(const std::string &filepath);

        bool point_in_polygon(const Point2D &pt, const std::vector<Point2D> &poly) const;
        bool point_in_free_space(const Point2D &point) const;

        std::optional<Point2D> segment_intersect(const Segment2D &s1, const Segment2D &s2) const;
        double point_to_segment_distance(const Point2D &p, const Segment2D &s) const;

    private:
        double sensor_range_;
        double safety_distance_;
        bool use_regions_;

        double x_min_;
        double y_min_;
        double x_max_;
        double y_max_;

        std::vector<Point2D> boundary_vertices_;
        std::vector<std::vector<Point2D>> holes_vertices_;
        std::vector<Segment2D> map_segments_;
        std::vector<std::vector<int8_t>> grid_map_;
    };

} // namespace amr_localization_cpp