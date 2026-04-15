#include "amr_localization_cpp/py_map_bridge.hpp"

#include <pybind11/embed.h>
#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace amr_localization_cpp
{

    class PyMapBridge::Impl
    {
    public:
        Impl(
            const std::string &map_path,
            double sensor_range,
            double safety_distance,
            bool use_regions,
            bool compiled_intersect)
        {
            if (!Py_IsInitialized())
            {
                guard_ = std::make_unique<py::scoped_interpreter>();
            }

            py::gil_scoped_acquire gil;

            try
            {
                py::module_ sys = py::module_::import("sys");
                sys.attr("path").attr("insert")(0, ".");

                py::module_ maps_mod = py::module_::import("amr_localization_cpp.maps");
                py::object map_cls = maps_mod.attr("Map");

                map_obj_ = map_cls(
                    map_path,
                    sensor_range,
                    safety_distance,
                    use_regions,
                    compiled_intersect);
            }
            catch (const py::error_already_set &e)
            {
                throw std::runtime_error(std::string("Failed to initialize Python Map: ") + e.what());
            }
        }

        std::tuple<double, double, double, double> bounds() const
        {
            py::gil_scoped_acquire gil;
            py::tuple t = map_obj_.attr("bounds")().cast<py::tuple>();
            return {
                t[0].cast<double>(),
                t[1].cast<double>(),
                t[2].cast<double>(),
                t[3].cast<double>()};
        }

        bool contains(const Point2D &point) const
        {
            py::gil_scoped_acquire gil;
            py::tuple pt = py::make_tuple(point.x, point.y);
            return map_obj_.attr("contains")(pt).cast<bool>();
        }

        std::pair<std::optional<Point2D>, double> check_collision(
            const Segment2D &segment,
            bool compute_distance) const
        {
            py::gil_scoped_acquire gil;

            py::list seg;
            seg.append(py::make_tuple(segment[0].x, segment[0].y));
            seg.append(py::make_tuple(segment[1].x, segment[1].y));

            py::tuple result = map_obj_.attr("check_collision")(seg, compute_distance).cast<py::tuple>();

            py::object py_intersection = result[0];
            double dist = result[1].cast<double>();

            if (py_intersection.is_none())
            {
                return {std::nullopt, dist};
            }

            if (py::isinstance<py::list>(py_intersection) && py::len(py_intersection) == 0)
            {
                return {std::nullopt, dist};
            }

            py::tuple pt = py_intersection.cast<py::tuple>();
            return {Point2D{pt[0].cast<double>(), pt[1].cast<double>()}, dist};
        }

    private:
        std::unique_ptr<py::scoped_interpreter> guard_;
        py::object map_obj_;
    };

    PyMapBridge::PyMapBridge(
        const std::string &map_path,
        double sensor_range,
        double safety_distance,
        bool use_regions,
        bool compiled_intersect)
        : impl_(new Impl(map_path, sensor_range, safety_distance, use_regions, compiled_intersect))
    {
    }

    PyMapBridge::~PyMapBridge()
    {
        delete impl_;
    }

    std::tuple<double, double, double, double> PyMapBridge::bounds() const
    {
        return impl_->bounds();
    }

    bool PyMapBridge::contains(const Point2D &point) const
    {
        return impl_->contains(point);
    }

    std::pair<std::optional<Point2D>, double> PyMapBridge::check_collision(
        const Segment2D &segment,
        bool compute_distance) const
    {
        return impl_->check_collision(segment, compute_distance);
    }

} // namespace amr_localization_cpp