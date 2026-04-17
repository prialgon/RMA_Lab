#pragma once

#include "amr_localization_cpp/particle_filter.hpp"

#include <amr_msgs/msg/allow_move.hpp>
#include <amr_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <rclcpp_lifecycle/lifecycle_publisher.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <std_msgs/msg/empty.hpp>

#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace amr_localization_cpp
{

    class ParticleFilterNode : public rclcpp_lifecycle::LifecycleNode
    {
    public:
        using CallbackReturn =
            rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

        explicit ParticleFilterNode(
            const rclcpp::NodeOptions &options = rclcpp::NodeOptions());

        CallbackReturn on_configure(const rclcpp_lifecycle::State &state) override;
        CallbackReturn on_activate(const rclcpp_lifecycle::State &state) override;
        CallbackReturn on_deactivate(const rclcpp_lifecycle::State &state) override;
        CallbackReturn on_cleanup(const rclcpp_lifecycle::State &state) override;
        CallbackReturn on_shutdown(const rclcpp_lifecycle::State &state) override;

    private:
        using PoseMsg = amr_msgs::msg::PoseStamped;
        using AllowMoveMsg = amr_msgs::msg::AllowMove;
        using OdomMsg = nav_msgs::msg::Odometry;
        using ScanMsg = sensor_msgs::msg::LaserScan;
        using EmptyMsg = std_msgs::msg::Empty;

        using ApproxPolicy =
            message_filters::sync_policies::ApproximateTime<OdomMsg, ScanMsg>;
        using Sync = message_filters::Synchronizer<ApproxPolicy>;

        void compute_scan_callback(const ScanMsg::SharedPtr scan_msg);
        void compute_odometry_callback(const OdomMsg::SharedPtr odom_msg);
        void timer_callback();
        void compute_pose_callback(
            const OdomMsg::ConstSharedPtr &odom_msg,
            const ScanMsg::ConstSharedPtr &scan_msg);
        void reset_localization_callback(const EmptyMsg::SharedPtr msg);
        void rebuild_particle_filter();

        std::tuple<double, double, double> execute_measurement_step(
            const std::vector<float> &z_scan);
        void execute_motion_step(double z_v, double z_w);
        void publish_pose_estimate(double x_h, double y_h, double theta_h);

        static geometry_msgs::msg::Quaternion yaw_to_quaternion(double yaw);

        bool enable_plot_{false};
        bool localized_{true};
        bool simulation_{false};
        bool finished_flag_{true};

        int steps_{0};
        int steps_btw_sense_updates_{10};

        std::tuple<double, double, double> initial_pose_{0.0, 0.0, 0.0};
        std::tuple<double, double, double> initial_pose_sigma_{0.05, 0.05, 5.0 * 3.14159265358979323846 / 180.0};
        std::string map_path_;
        std::tuple<double, double, double> last_pose_{0.0, 0.0, 0.0};

        std::optional<std::chrono::steady_clock::time_point> last_odometry_callback_;

        std::vector<float> z_scan_;
        std::vector<std::pair<double, double>> z_odom_;

        std::unique_ptr<ParticleFilter> particle_filter_;

        int particle_count_{100};
        double dt_{0.05};
        double sigma_v_{0.1};
        double sigma_w_{0.1};
        double sigma_z_{0.1};
        bool global_localization_{true};
        std::string reset_topic_{"particle_filter/reset"};

        rclcpp_lifecycle::LifecyclePublisher<PoseMsg>::SharedPtr pose_publisher_;
        rclcpp_lifecycle::LifecyclePublisher<PoseMsg>::SharedPtr particle_filter_pose_publisher_;
        rclcpp_lifecycle::LifecyclePublisher<AllowMoveMsg>::SharedPtr allowed_movement_publisher_;

        rclcpp::Subscription<ScanMsg>::SharedPtr laser_scan_subscriber_;
        rclcpp::Subscription<EmptyMsg>::SharedPtr reset_subscriber_;
        rclcpp::Subscription<OdomMsg>::SharedPtr odometry_subscriber_;
        rclcpp::TimerBase::SharedPtr timer_;

        std::unique_ptr<message_filters::Subscriber<OdomMsg>> odom_filter_sub_;
        std::unique_ptr<message_filters::Subscriber<ScanMsg>> scan_filter_sub_;
        std::shared_ptr<Sync> sync_;
    };

} // namespace amr_localization_cpp