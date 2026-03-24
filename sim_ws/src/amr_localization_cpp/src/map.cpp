#include "amr_localization_cpp/map.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

#include "nlohmann/json.hpp"

using json = nlohmann::json;

namespace amr_localization_cpp
{

    Map::Map(const std::string &json_file, double sensor_range, double safety_distance, bool use_regions)
        : sensor_range_(sensor_range),
          safety_distance_(safety_distance),
          use_regions_(use_regions),
          x_min_(std::numeric_limits<double>::max()),
          y_min_(std::numeric_limits<double>::max()),
          x_max_(std::numeric_limits<double>::lowest()),
          y_max_(std::numeric_limits<double>::lowest())
    {
        load_json(json_file);
    }

    void Map::load_json(const std::string &filepath)
    {
        std::ifstream file(filepath);
        if (!file.is_open())
        {
            throw std::runtime_error("Could not open map file: " + filepath);
        }

        json data;
        file >> data;

        auto parse_polygon = [&](const json &points, std::vector<Point2D> &vertices)
        {
            if (points.empty())
            {
                return;
            }

            for (size_t i = 0; i + 1 < points.size(); ++i)
            {
                Point2D p1{points[i][0].get<double>(), points[i][1].get<double>()};
                Point2D p2{points[i + 1][0].get<double>(), points[i + 1][1].get<double>()};

                vertices.push_back(p1);
                map_segments_.push_back({p1, p2});

                x_min_ = std::min({x_min_, p1.x, p2.x});
                y_min_ = std::min({y_min_, p1.y, p2.y});
                x_max_ = std::max({x_max_, p1.x, p2.x});
                y_max_ = std::max({y_max_, p1.y, p2.y});
            }

            Point2D p_last{points.back()[0].get<double>(), points.back()[1].get<double>()};
            Point2D p_first{points.front()[0].get<double>(), points.front()[1].get<double>()};

            vertices.push_back(p_last);
            map_segments_.push_back({p_last, p_first});

            x_min_ = std::min({x_min_, p_last.x, p_first.x});
            y_min_ = std::min({y_min_, p_last.y, p_first.y});
            x_max_ = std::max({x_max_, p_last.x, p_first.x});
            y_max_ = std::max({y_max_, p_last.y, p_first.y});
        };

        parse_polygon(data["metric"]["boundary"], boundary_vertices_);

        for (const auto &hole : data["metric"]["holes"])
        {
            std::vector<Point2D> hole_vertices;
            parse_polygon(hole, hole_vertices);
            holes_vertices_.push_back(hole_vertices);
        }

        if (data.contains("grid"))
        {
            int rows = data["grid"]["size"][0].get<int>();
            int cols = data["grid"]["size"][1].get<int>();
            grid_map_.assign(rows, std::vector<int8_t>(cols, 0));

            for (const auto &obs : data["grid"]["obstacles"])
            {
                int r = obs[0].get<int>();
                int c = obs[1].get<int>();

                if (r >= 0 && r < rows && c >= 0 && c < cols)
                {
                    grid_map_[r][c] = 1;
                }
            }
        }
    }

    std::tuple<double, double, double, double> Map::bounds() const
    {
        return {x_min_, y_min_, x_max_, y_max_};
    }

    bool Map::point_in_polygon(const Point2D &pt, const std::vector<Point2D> &poly) const
    {
        if (poly.empty())
        {
            return false;
        }

        bool inside = false;
        for (size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++)
        {
            const bool crosses =
                ((poly[i].y > pt.y) != (poly[j].y > pt.y)) &&
                (pt.x < (poly[j].x - poly[i].x) * (pt.y - poly[i].y) / (poly[j].y - poly[i].y) + poly[i].x);

            if (crosses)
            {
                inside = !inside;
            }
        }

        return inside;
    }

    bool Map::point_in_free_space(const Point2D &point) const
    {
        if (!point_in_polygon(point, boundary_vertices_))
        {
            return false;
        }

        for (const auto &hole : holes_vertices_)
        {
            if (point_in_polygon(point, hole))
            {
                return false;
            }
        }

        return true;
    }

    bool Map::contains(const Point2D &point) const
    {
        // Match Python behavior:
        // return self._safe_map_polygon.contains(pt)
        // We approximate that by:
        // 1) point must be in free space
        // 2) point must be at least safety_distance away from every wall/hole segment
        if (!point_in_free_space(point))
        {
            return false;
        }

        if (safety_distance_ <= 0.0)
        {
            return true;
        }

        for (const auto &seg : map_segments_)
        {
            if (point_to_segment_distance(point, seg) < safety_distance_)
            {
                return false;
            }
        }

        return true;
    }

    std::optional<Point2D> Map::segment_intersect(const Segment2D &s1, const Segment2D &s2) const
    {
        const double s1_x = s1.p2.x - s1.p1.x;
        const double s1_y = s1.p2.y - s1.p1.y;
        const double s2_x = s2.p2.x - s2.p1.x;
        const double s2_y = s2.p2.y - s2.p1.y;

        const double denom = -s2_x * s1_y + s1_x * s2_y;
        if (std::abs(denom) < 1e-10)
        {
            return std::nullopt;
        }

        const double s = (-s1_y * (s1.p1.x - s2.p1.x) + s1_x * (s1.p1.y - s2.p1.y)) / denom;
        const double t = (s2_x * (s1.p1.y - s2.p1.y) - s2_y * (s1.p1.x - s2.p1.x)) / denom;

        if (s >= 0.0 && s <= 1.0 && t >= 0.0 && t <= 1.0)
        {
            return Point2D{s1.p1.x + t * s1_x, s1.p1.y + t * s1_y};
        }

        return std::nullopt;
    }

    double Map::point_to_segment_distance(const Point2D &p, const Segment2D &s) const
    {
        const double l2 =
            (s.p2.x - s.p1.x) * (s.p2.x - s.p1.x) +
            (s.p2.y - s.p1.y) * (s.p2.y - s.p1.y);

        if (l2 == 0.0)
        {
            return std::hypot(p.x - s.p1.x, p.y - s.p1.y);
        }

        const double t = std::max(
            0.0,
            std::min(
                1.0,
                ((p.x - s.p1.x) * (s.p2.x - s.p1.x) + (p.y - s.p1.y) * (s.p2.y - s.p1.y)) / l2));

        const Point2D proj{
            s.p1.x + t * (s.p2.x - s.p1.x),
            s.p1.y + t * (s.p2.y - s.p1.y)};

        return std::hypot(p.x - proj.x, p.y - proj.y);
    }

    std::pair<std::optional<Point2D>, double> Map::check_collision(const Segment2D &segment, bool compute_distance) const
    {
        std::optional<Point2D> closest_pt = std::nullopt;
        double min_dist = std::numeric_limits<double>::max();

        for (const auto &map_seg : map_segments_)
        {
            auto pt = segment_intersect(segment, map_seg);
            if (pt.has_value())
            {
                const double dist = std::hypot(pt->x - segment.p1.x, pt->y - segment.p1.y);
                if (dist < min_dist)
                {
                    min_dist = dist;
                    closest_pt = pt;
                }
            }
        }

        if (!closest_pt.has_value())
        {
            return {std::nullopt, std::numeric_limits<double>::quiet_NaN()};
        }

        if (!compute_distance)
        {
            return {closest_pt, std::numeric_limits<double>::quiet_NaN()};
        }

        return {closest_pt, min_dist};
    }

    bool Map::crosses(const Segment2D &segment) const
    {
        // Conservative Python-like behavior:
        // if the segment intersects a map segment, it crosses a wall.
        for (const auto &map_seg : map_segments_)
        {
            if (segment_intersect(segment, map_seg).has_value())
            {
                return true;
            }
        }
        return false;
    }

} // namespace amr_localization_cpp