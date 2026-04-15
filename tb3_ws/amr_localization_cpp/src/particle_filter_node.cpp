#include "amr_localization_cpp/particle_filter_node.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <cmath>
#include <filesystem>

namespace amr_localization_cpp
{

    ParticleFilterNode::ParticleFilterNode(const rclcpp::NodeOptions &options)
        : rclcpp_lifecycle::LifecycleNode("particle_filter", options)
    {
        this->declare_parameter<double>("dt", 0.05);
        this->declare_parameter<bool>("enable_plot", false);
        this->declare_parameter<bool>("global_localization", true);
        this->declare_parameter<std::vector<double>>("initial_pose", {0.0, 0.0, 0.0});
        this->declare_parameter<std::vector<double>>(
            "initial_pose_sigma", {0.05, 0.05, 5.0 * M_PI / 180.0});
        this->declare_parameter<int>("particles", 100);
        this->declare_parameter<double>("sigma_v", 0.1);
        this->declare_parameter<double>("sigma_w", 0.1);
        this->declare_parameter<double>("sigma_z", 0.1);
        this->declare_parameter<bool>("simulation", false);
        this->declare_parameter<int>("steps_btw_sense_updates", 10);
        this->declare_parameter<std::string>("world", "lab03");
        this->declare_parameter<std::string>("particle_filter_pose_topic", "particle_filter/pose");
        this->declare_parameter<std::string>("reset_topic", "particle_filter/reset");
    }

    auto ParticleFilterNode::on_configure(const rclcpp_lifecycle::State &state) -> CallbackReturn
    {
        RCLCPP_INFO(
            get_logger(),
            "Transitioning from '%s' to 'inactive' state.",
            state.label().c_str());

        try
        {
            const double dt = this->get_parameter("dt").as_double();
            dt_ = dt;
            enable_plot_ = this->get_parameter("enable_plot").as_bool();
            const bool global_localization = this->get_parameter("global_localization").as_bool();

            const auto initial_pose_vec = this->get_parameter("initial_pose").as_double_array();
            const auto initial_pose_sigma_vec = this->get_parameter("initial_pose_sigma").as_double_array();

            initial_pose_ = {
                initial_pose_vec.at(0),
                initial_pose_vec.at(1),
                initial_pose_vec.at(2)};

            const std::tuple<double, double, double> initial_pose_sigma = {
                initial_pose_sigma_vec.at(0),
                initial_pose_sigma_vec.at(1),
                initial_pose_sigma_vec.at(2)};

            const int particles = this->get_parameter("particles").as_int();
            particle_count_ = particles;
            const double sigma_v = this->get_parameter("sigma_v").as_double();
            sigma_v_ = sigma_v;
            const double sigma_w = this->get_parameter("sigma_w").as_double();
            sigma_w_ = sigma_w;
            const double sigma_z = this->get_parameter("sigma_z").as_double();
            sigma_z_ = sigma_z;
            steps_btw_sense_updates_ = this->get_parameter("steps_btw_sense_updates").as_int();
            simulation_ = this->get_parameter("simulation").as_bool();
            const std::string world = this->get_parameter("world").as_string();
            const std::string particle_filter_pose_topic =
                this->get_parameter("particle_filter_pose_topic").as_string();
            reset_topic_ = this->get_parameter("reset_topic").as_string();
            global_localization_ = global_localization;
            initial_pose_sigma_ = initial_pose_sigma;

            localized_ = false;
            steps_ = 0;

            const std::string pkg_share =
                ament_index_cpp::get_package_share_directory("amr_localization_cpp");
            const std::string map_path = pkg_share + "/maps/" + world + ".json";
            map_path_ = map_path;

            rebuild_particle_filter();

            z_scan_.clear();
            z_odom_.clear();
            last_pose_ = initial_pose_;
            last_odometry_callback_.reset();

            if (enable_plot_)
            {
                particle_filter_->show("Initialization", true, false, false, true);
            }

            pose_publisher_ = this->create_publisher<PoseMsg>("pose", 10);
            particle_filter_pose_publisher_ =
                this->create_publisher<PoseMsg>(particle_filter_pose_topic, 10);
            allowed_movement_publisher_ = this->create_publisher<AllowMoveMsg>("allow_move", 10);
            if (simulation_)
            {
                reset_subscriber_ = this->create_subscription<EmptyMsg>(
                    reset_topic_,
                    10,
                    std::bind(&ParticleFilterNode::reset_localization_callback, this, std::placeholders::_1));
            }

            rclcpp::QoS scan_qos(10);
            scan_qos.best_effort();
            scan_qos.durability_volatile();

            if (simulation_)
            {
                odom_filter_sub_ = std::make_unique<message_filters::Subscriber<OdomMsg>>();
                scan_filter_sub_ = std::make_unique<message_filters::Subscriber<ScanMsg>>();

                odom_filter_sub_->subscribe(this, "odometry", rclcpp::QoS(10));
                scan_filter_sub_->subscribe(this, "scan", rclcpp::SensorDataQoS());

                sync_ = std::make_shared<Sync>(ApproxPolicy(10), *odom_filter_sub_, *scan_filter_sub_);
                sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(9.0));
                sync_->registerCallback(
                    std::bind(
                        &ParticleFilterNode::compute_pose_callback,
                        this,
                        std::placeholders::_1,
                        std::placeholders::_2));
            }
            else
            {
                laser_scan_subscriber_ = this->create_subscription<ScanMsg>(
                    "scan",
                    scan_qos,
                    std::bind(&ParticleFilterNode::compute_scan_callback, this, std::placeholders::_1));

                odometry_subscriber_ = this->create_subscription<OdomMsg>(
                    "odometry",
                    10,
                    std::bind(&ParticleFilterNode::compute_odometry_callback, this, std::placeholders::_1));

                timer_ = this->create_wall_timer(
                    std::chrono::seconds(5),
                    std::bind(&ParticleFilterNode::timer_callback, this));

                finished_flag_ = true;
            }
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(get_logger(), "%s", e.what());
            return CallbackReturn::ERROR;
        }
        catch (...)
        {
            RCLCPP_ERROR(get_logger(), "Unknown error in on_configure");
            return CallbackReturn::ERROR;
        }

        return CallbackReturn::SUCCESS;
    }

    auto ParticleFilterNode::on_activate(const rclcpp_lifecycle::State &state) -> CallbackReturn
    {
        RCLCPP_INFO(
            get_logger(),
            "Transitioning from '%s' to 'active' state.",
            state.label().c_str());

        pose_publisher_->on_activate();
        particle_filter_pose_publisher_->on_activate();
        allowed_movement_publisher_->on_activate();

        return CallbackReturn::SUCCESS;
    }

    auto ParticleFilterNode::on_deactivate(const rclcpp_lifecycle::State &state) -> CallbackReturn
    {
        (void)state;
        if (pose_publisher_)
        {
            pose_publisher_->on_deactivate();
        }
        if (allowed_movement_publisher_)
        {
            allowed_movement_publisher_->on_deactivate();
        }
        if (particle_filter_pose_publisher_)
        {
            particle_filter_pose_publisher_->on_deactivate();
        }
        return CallbackReturn::SUCCESS;
    }

    auto ParticleFilterNode::on_cleanup(const rclcpp_lifecycle::State &state) -> CallbackReturn
    {
        (void)state;
        timer_.reset();
        laser_scan_subscriber_.reset();
        odometry_subscriber_.reset();
        sync_.reset();
        odom_filter_sub_.reset();
        scan_filter_sub_.reset();
        pose_publisher_.reset();
        particle_filter_pose_publisher_.reset();
        allowed_movement_publisher_.reset();
        particle_filter_.reset();
        z_scan_.clear();
        z_odom_.clear();
        last_odometry_callback_.reset();
        return CallbackReturn::SUCCESS;
    }

    auto ParticleFilterNode::on_shutdown(const rclcpp_lifecycle::State &state) -> CallbackReturn
    {
        (void)state;
        return CallbackReturn::SUCCESS;
    }

    void ParticleFilterNode::compute_scan_callback(const ScanMsg::SharedPtr scan_msg)
    {

        z_scan_ = scan_msg->ranges;
    }

    void ParticleFilterNode::compute_odometry_callback(const OdomMsg::SharedPtr odom_msg)
    {


        const double z_v = odom_msg->twist.twist.linear.x;
        const double z_w = odom_msg->twist.twist.angular.z;

        if (std::abs(z_v) >= 0.01 || std::abs(z_w) >= 0.1)
        {
            z_odom_.push_back({z_v, z_w});
        }

        if (last_odometry_callback_.has_value())
        {
            const auto now = std::chrono::steady_clock::now();
            const double delta_time =
                std::chrono::duration<double>(now - *last_odometry_callback_).count();

            const double x_last = std::get<0>(last_pose_);
            const double y_last = std::get<1>(last_pose_);
            const double th_last = std::get<2>(last_pose_);

            const double x_h = x_last + z_v * std::cos(th_last + z_w * delta_time / 2.0) * delta_time;
            const double y_h = y_last + z_v * std::sin(th_last + z_w * delta_time / 2.0) * delta_time;
            const double th_h = th_last + z_w * delta_time;

            last_pose_ = {x_h, y_h, th_h};

            PoseMsg msg;
            msg.header.stamp = this->get_clock()->now();
            msg.localized = localized_;
            msg.pose.position.x = x_h;
            msg.pose.position.y = y_h;
            msg.pose.orientation = yaw_to_quaternion(th_h);

            pose_publisher_->publish(msg);
        }

        last_odometry_callback_ = std::chrono::steady_clock::now();
    }

    void ParticleFilterNode::timer_callback()
    {


        if (finished_flag_ && !z_odom_.empty())
        {
            finished_flag_ = false;

            AllowMoveMsg allow_msg;
            allow_msg.header.stamp = this->get_clock()->now();
            allow_msg.allowed = false;
            allowed_movement_publisher_->publish(allow_msg);

            for (const auto &[z_v, z_w] : z_odom_)
            {
                execute_motion_step(z_v, z_w);
                steps_ += 1;
            }

            const auto [x_h, y_h, theta_h] = execute_measurement_step(z_scan_);

            last_odometry_callback_ = std::chrono::steady_clock::now();
            last_pose_ = {x_h, y_h, theta_h};

            AllowMoveMsg allow_msg2;
            allow_msg2.header.stamp = this->get_clock()->now();
            allow_msg2.allowed = true;
            allowed_movement_publisher_->publish(allow_msg2);

            publish_pose_estimate(x_h, y_h, theta_h);
            finished_flag_ = true;
            z_odom_.clear();
        }
    }

    void ParticleFilterNode::compute_pose_callback(
        const OdomMsg::ConstSharedPtr &odom_msg,
        const ScanMsg::ConstSharedPtr &scan_msg)
    {


        AllowMoveMsg msg;
        msg.header.stamp = this->get_clock()->now();
        msg.allowed = true;
        allowed_movement_publisher_->publish(msg);

        const double z_v = odom_msg->twist.twist.linear.x;
        const double z_w = odom_msg->twist.twist.angular.z;
        const std::vector<float> z_scan = scan_msg->ranges;

        execute_motion_step(z_v, z_w);
        steps_ += 1;

        double x_h = std::numeric_limits<double>::infinity();
        double y_h = std::numeric_limits<double>::infinity();
        double theta_h = std::numeric_limits<double>::infinity();

        if (localized_ || (steps_btw_sense_updates_ > 0 && steps_ % steps_btw_sense_updates_ == 0))
        {
            std::tie(x_h, y_h, theta_h) = execute_measurement_step(z_scan);
        }

        publish_pose_estimate(x_h, y_h, theta_h);
    }

    std::tuple<double, double, double> ParticleFilterNode::execute_measurement_step(
        const std::vector<float> &z_scan)
    {
        std::tuple<double, double, double> pose{
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity()};



        {
            const auto start = std::chrono::steady_clock::now();
            particle_filter_->resample(z_scan);
            const double sense_time =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            RCLCPP_INFO(get_logger(), "Sense step time: %6.3f s", sense_time);
        }

        if (enable_plot_)
        {
            particle_filter_->show("Sense", true, false, false, true);
        }

        {
            const auto start = std::chrono::steady_clock::now();
            auto result = particle_filter_->compute_pose();
            localized_ = std::get<0>(result);
            pose = std::get<1>(result);
            const double clustering_time =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            RCLCPP_INFO(get_logger(), "Clustering time: %6.3f s", clustering_time);
        }

        return pose;
    }

    void ParticleFilterNode::execute_motion_step(double z_v, double z_w)
    {


        const auto start = std::chrono::steady_clock::now();
        particle_filter_->move(z_v, z_w);
        const double move_time =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

        // RCLCPP_INFO(get_logger(), "Move step time: %7.3f s", move_time);

        if (enable_plot_)
        {
            particle_filter_->show("Move", true, false, false, true);
        }
    }

    void ParticleFilterNode::publish_pose_estimate(double x_h, double y_h, double theta_h)
    {
        PoseMsg msg;
        msg.header.stamp = this->get_clock()->now();
        msg.localized = localized_;

        if (localized_)
        {
            msg.pose.position.x = x_h;
            msg.pose.position.y = y_h;
            msg.pose.orientation = yaw_to_quaternion(theta_h);
        }

        pose_publisher_->publish(msg);
        if (localized_)
        {
            particle_filter_pose_publisher_->publish(msg);
        }
    }

    void ParticleFilterNode::rebuild_particle_filter()
    {
        particle_filter_ = std::make_unique<ParticleFilter>(
            dt_,
            map_path_,
            particle_count_,
            sigma_v_,
            sigma_w_,
            sigma_z_,
            8.0,
            0.16,
            global_localization_,
            initial_pose_,
            initial_pose_sigma_,
            simulation_);
    }

    void ParticleFilterNode::reset_localization_callback(const EmptyMsg::SharedPtr)
    {
        if (!simulation_)
        {
            return;
        }

        RCLCPP_WARN(get_logger(), "Restarting particle filter because localization error exceeded the threshold.");
        rebuild_particle_filter();
        localized_ = false;
        steps_ = 0;
        z_scan_.clear();
        z_odom_.clear();
        last_pose_ = initial_pose_;
        last_odometry_callback_.reset();

        if (enable_plot_)
        {
            particle_filter_->show("After reset", true, false, false, true);
        }

        AllowMoveMsg allow_msg;
        allow_msg.header.stamp = this->get_clock()->now();
        allow_msg.allowed = true;
        allowed_movement_publisher_->publish(allow_msg);

        publish_pose_estimate(
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity());
    }

    geometry_msgs::msg::Quaternion ParticleFilterNode::yaw_to_quaternion(double yaw)
    {
        tf2::Quaternion q;
        q.setRPY(0.0, 0.0, yaw);
        return tf2::toMsg(q);
    }

} // namespace amr_localization_cpp