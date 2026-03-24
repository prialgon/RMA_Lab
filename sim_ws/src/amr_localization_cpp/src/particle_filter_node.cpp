#include <cmath>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "ament_index_cpp/get_package_share_directory.hpp"

#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "message_filters/synchronizer.h"

#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "tf2/LinearMath/Quaternion.h"

#include "amr_msgs/msg/allow_move.hpp"
#include "amr_msgs/msg/pose_stamped.hpp"

#include "amr_localization_cpp/particle_filter.hpp"

using std::placeholders::_1;
using std::placeholders::_2;
using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace amr_localization_cpp
{

    class ParticleFilterNode : public rclcpp_lifecycle::LifecycleNode
    {
    public:
        ParticleFilterNode()
            : rclcpp_lifecycle::LifecycleNode("particle_filter")
        {
            this->declare_parameter("dt", 0.05);
            this->declare_parameter("enable_plot", false);
            this->declare_parameter("global_localization", true);
            this->declare_parameter("initial_pose", std::vector<double>{0.0, 0.0, 0.0});
            this->declare_parameter("initial_pose_sigma", std::vector<double>{0.05, 0.05, 5.0 * M_PI / 180.0});
            this->declare_parameter("particles", 100);
            this->declare_parameter("sigma_v", 0.1);
            this->declare_parameter("sigma_w", 0.1);
            this->declare_parameter("sigma_z", 0.1);
            this->declare_parameter("simulation", false);
            this->declare_parameter("steps_btw_sense_updates", 10);
            this->declare_parameter("world", "lab03");
        }

        CallbackReturn on_configure(const rclcpp_lifecycle::State &state) override
        {
            RCLCPP_INFO(
                this->get_logger(),
                "Transitioning from '%s' to 'inactive' state.",
                state.label().c_str());

            try
            {
                // Parameters
                const double dt = this->get_parameter("dt").as_double();
                const bool global_localization = this->get_parameter("global_localization").as_bool();
                const auto init_pose_vec = this->get_parameter("initial_pose").as_double_array();
                const auto init_sigma_vec = this->get_parameter("initial_pose_sigma").as_double_array();
                const int particles = this->get_parameter("particles").as_int();
                const double sigma_v = this->get_parameter("sigma_v").as_double();
                const double sigma_w = this->get_parameter("sigma_w").as_double();
                const double sigma_z = this->get_parameter("sigma_z").as_double();
                simulation_ = this->get_parameter("simulation").as_bool();
                const std::string world = this->get_parameter("world").as_string();

                if (init_pose_vec.size() != 3)
                {
                    RCLCPP_ERROR(
                        this->get_logger(),
                        "Parameter 'initial_pose' must contain exactly 3 values, got %zu",
                        init_pose_vec.size());
                    return CallbackReturn::ERROR;
                }

                if (init_sigma_vec.size() != 3)
                {
                    RCLCPP_ERROR(
                        this->get_logger(),
                        "Parameter 'initial_pose_sigma' must contain exactly 3 values, got %zu",
                        init_sigma_vec.size());
                    return CallbackReturn::ERROR;
                }

                RCLCPP_INFO(this->get_logger(), "on_configure: parameters loaded");
                RCLCPP_INFO(this->get_logger(), "  dt=%.6f", dt);
                RCLCPP_INFO(this->get_logger(), "  global_localization=%s", global_localization ? "true" : "false");
                RCLCPP_INFO(this->get_logger(), "  particles=%d", particles);
                RCLCPP_INFO(this->get_logger(), "  sigma_v=%.6f", sigma_v);
                RCLCPP_INFO(this->get_logger(), "  sigma_w=%.6f", sigma_w);
                RCLCPP_INFO(this->get_logger(), "  sigma_z=%.6f", sigma_z);
                RCLCPP_INFO(this->get_logger(), "  simulation=%s", simulation_ ? "true" : "false");
                RCLCPP_INFO(this->get_logger(), "  world=%s", world.c_str());

                // Resolve map path
                const std::string map_path = resolve_map_path(world);
                RCLCPP_INFO(this->get_logger(), "  map_path=%s", map_path.c_str());

                // State initialization
                initial_pose_ = {init_pose_vec[0], init_pose_vec[1], init_pose_vec[2]};
                last_pose_ = initial_pose_;
                localized_ = true;
                finished_flag_ = true;
                have_last_odom_time_ = false;
                z_scan_.clear();
                z_odom_.clear();

                RCLCPP_INFO(this->get_logger(), "on_configure: before ParticleFilter construction");

                pf_ = std::make_unique<ParticleFilter>(
                    dt,
                    map_path,
                    particles,
                    sigma_v,
                    sigma_w,
                    sigma_z,
                    8.0,
                    0.16,
                    global_localization,
                    initial_pose_,
                    std::make_tuple(init_sigma_vec[0], init_sigma_vec[1], init_sigma_vec[2]));

                RCLCPP_INFO(this->get_logger(), "on_configure: ParticleFilter constructed successfully");

                // Publishers
                pose_pub_ = this->create_publisher<amr_msgs::msg::PoseStamped>("pose", 10);
                allow_move_pub_ = this->create_publisher<amr_msgs::msg::AllowMove>("allow_move", 10);

                RCLCPP_INFO(this->get_logger(), "on_configure: publishers created");

                auto scan_qos = rclcpp::SensorDataQoS();

                // Subscribers / timers
                if (simulation_)
                {
                    RCLCPP_INFO(this->get_logger(), "on_configure: configuring simulation subscribers");

                    odom_sub_.subscribe(this, "odometry");
                    scan_sub_.subscribe(this, "scan", scan_qos.get_rmw_qos_profile());

                    sync_ = std::make_shared<message_filters::Synchronizer<ApproxSyncPolicy>>(
                        ApproxSyncPolicy(10), odom_sub_, scan_sub_);

                    sync_->registerCallback(std::bind(&ParticleFilterNode::compute_pose_callback, this, _1, _2));

                    RCLCPP_INFO(this->get_logger(), "on_configure: simulation subscribers configured");
                }
                else
                {
                    RCLCPP_INFO(this->get_logger(), "on_configure: configuring real-robot subscriptions and timer");

                    laser_scan_subscriber_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
                        "scan",
                        scan_qos,
                        std::bind(&ParticleFilterNode::compute_scan_callback, this, _1));

                    odometry_subscriber_ = this->create_subscription<nav_msgs::msg::Odometry>(
                        "odometry",
                        10,
                        std::bind(&ParticleFilterNode::compute_odometry_callback, this, _1));

                    timer_ = this->create_wall_timer(
                        std::chrono::seconds(5),
                        std::bind(&ParticleFilterNode::timer_callback, this));

                    RCLCPP_INFO(this->get_logger(), "on_configure: real-robot subscriptions and timer configured");
                }

                RCLCPP_INFO(this->get_logger(), "on_configure completed successfully");
                return CallbackReturn::SUCCESS;
            }
            catch (const std::exception &e)
            {
                RCLCPP_ERROR(this->get_logger(), "Exception in on_configure: %s", e.what());
                return CallbackReturn::ERROR;
            }
            catch (...)
            {
                RCLCPP_ERROR(this->get_logger(), "Unknown exception in on_configure");
                return CallbackReturn::ERROR;
            }
        }

        CallbackReturn on_activate(const rclcpp_lifecycle::State &state) override
        {
            pose_pub_->on_activate();
            allow_move_pub_->on_activate();
            return rclcpp_lifecycle::LifecycleNode::on_activate(state);
        }

    private:
        using ApproxSyncPolicy =
            message_filters::sync_policies::ApproximateTime<nav_msgs::msg::Odometry, sensor_msgs::msg::LaserScan>;

        std::unique_ptr<ParticleFilter> pf_;

        std::shared_ptr<rclcpp_lifecycle::LifecyclePublisher<amr_msgs::msg::PoseStamped>> pose_pub_;
        std::shared_ptr<rclcpp_lifecycle::LifecyclePublisher<amr_msgs::msg::AllowMove>> allow_move_pub_;

        message_filters::Subscriber<nav_msgs::msg::Odometry, rclcpp_lifecycle::LifecycleNode> odom_sub_;
        message_filters::Subscriber<sensor_msgs::msg::LaserScan, rclcpp_lifecycle::LifecycleNode> scan_sub_;
        std::shared_ptr<message_filters::Synchronizer<ApproxSyncPolicy>> sync_;

        rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr laser_scan_subscriber_;
        rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_subscriber_;
        rclcpp::TimerBase::SharedPtr timer_;

        bool simulation_{false};
        bool localized_{true};
        bool finished_flag_{true};
        bool have_last_odom_time_{false};

        std::vector<double> z_scan_;
        std::vector<std::pair<double, double>> z_odom_;

        std::tuple<double, double, double> initial_pose_{0.0, 0.0, 0.0};
        std::tuple<double, double, double> last_pose_{0.0, 0.0, 0.0};

        rclcpp::Time last_odometry_time_;

        std::string resolve_map_path(const std::string &world) const
        {
            const std::filesystem::path install_share_path =
                ament_index_cpp::get_package_share_directory("amr_localization_cpp");
            const std::filesystem::path map_file = world + ".json";

            std::vector<std::filesystem::path> candidate_paths;
            candidate_paths.emplace_back(install_share_path / "maps" / map_file);

            // Support local source-tree runs if installed maps are missing/outdated.
            std::filesystem::path workspace_root = install_share_path;
            for (int i = 0; i < 4 && workspace_root.has_parent_path(); ++i)
            {
                workspace_root = workspace_root.parent_path();
            }
            candidate_paths.emplace_back(
                workspace_root / "src" / "amr_localization_cpp" / "maps" / map_file);

            for (const auto &candidate : candidate_paths)
            {
                if (std::filesystem::exists(candidate))
                {
                    return candidate.string();
                }
            }

            std::ostringstream oss;
            oss << "Could not open map file for world '" << world << "'. Tried: ";
            for (size_t i = 0; i < candidate_paths.size(); ++i)
            {
                if (i > 0)
                {
                    oss << ", ";
                }
                oss << candidate_paths[i].string();
            }
            throw std::runtime_error(oss.str());
        }

        void compute_scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr scan_msg)
        {
            z_scan_ = std::vector<double>(scan_msg->ranges.begin(), scan_msg->ranges.end());
        }

        void compute_odometry_callback(const nav_msgs::msg::Odometry::SharedPtr odom_msg)
        {
            const double z_v = odom_msg->twist.twist.linear.x;
            const double z_w = odom_msg->twist.twist.angular.z;

            // Match Python exactly
            if (std::abs(z_v) >= 0.1 && std::abs(z_w) >= 0.1)
            {
                z_odom_.push_back({z_v, z_w});
            }

            const rclcpp::Time now_time = this->now();

            if (have_last_odom_time_)
            {
                const double delta_time = (now_time - last_odometry_time_).seconds();

                auto [x_last, y_last, th_last] = last_pose_;
                const double x_h = x_last + z_v * std::cos(th_last + z_w * delta_time / 2.0) * delta_time;
                const double y_h = y_last + z_v * std::sin(th_last + z_w * delta_time / 2.0) * delta_time;
                const double th_h = th_last + z_w * delta_time;

                last_pose_ = {x_h, y_h, th_h};
                publish_pose(x_h, y_h, th_h, true);
            }

            last_odometry_time_ = now_time;
            have_last_odom_time_ = true;
        }

        void timer_callback()
        {
            if (finished_flag_ && !z_odom_.empty())
            {
                finished_flag_ = false;

                amr_msgs::msg::AllowMove allow_msg;
                allow_msg.header.stamp = this->now();
                allow_msg.allowed = false;
                allow_move_pub_->publish(allow_msg);

                for (const auto &[z_v, z_w] : z_odom_)
                {
                    pf_->move(z_v, z_w);
                }

                auto [loc, pose] = [&]()
                {
                    pf_->resample(z_scan_);
                    return pf_->compute_pose();
                }();

                localized_ = loc;
                last_pose_ = pose;

                allow_msg.header.stamp = this->now();
                allow_msg.allowed = true;
                allow_move_pub_->publish(allow_msg);

                publish_pose(std::get<0>(pose), std::get<1>(pose), std::get<2>(pose), localized_);

                z_odom_.clear();
                last_odometry_time_ = this->now();
                have_last_odom_time_ = true;
                finished_flag_ = true;
            }
        }

        void compute_pose_callback(
            const nav_msgs::msg::Odometry::ConstSharedPtr &odom_msg,
            const sensor_msgs::msg::LaserScan::ConstSharedPtr &scan_msg)
        {
            amr_msgs::msg::AllowMove allow_msg;
            allow_msg.header.stamp = this->now();
            allow_msg.allowed = true;
            allow_move_pub_->publish(allow_msg);

            const double z_v = odom_msg->twist.twist.linear.x;
            const double z_w = odom_msg->twist.twist.angular.z;
            const std::vector<double> z_scan(scan_msg->ranges.begin(), scan_msg->ranges.end());

            pf_->move(z_v, z_w);
            pf_->resample(z_scan);

            auto [loc, pose] = pf_->compute_pose();
            localized_ = loc;

            publish_pose(std::get<0>(pose), std::get<1>(pose), std::get<2>(pose), localized_);
        }

        void publish_pose(double x, double y, double theta, bool localized)
        {
            amr_msgs::msg::PoseStamped msg;
            msg.header.stamp = this->now();
            msg.localized = localized;

            if (localized)
            {
                msg.pose.position.x = x;
                msg.pose.position.y = y;

                tf2::Quaternion q;
                q.setRPY(0.0, 0.0, theta);
                msg.pose.orientation.x = q.x();
                msg.pose.orientation.y = q.y();
                msg.pose.orientation.z = q.z();
                msg.pose.orientation.w = q.w();
            }

            pose_pub_->publish(msg);
        }
    };

} // namespace amr_localization_cpp

int main(int argc, char **argv)
{
    setvbuf(stdout, nullptr, _IONBF, 0);

    rclcpp::init(argc, argv);

    auto node = std::make_shared<amr_localization_cpp::ParticleFilterNode>();
    rclcpp::spin(node->get_node_base_interface());

    rclcpp::shutdown();
    return 0;
}