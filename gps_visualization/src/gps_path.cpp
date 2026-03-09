#include <memory>
#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/nav_sat_fix.hpp"
#include "gps_visualization/msg/nav_sat_fix_array.hpp"

using namespace std::chrono_literals;

class GPSFixtoArray : public rclcpp::Node
{
public:
    gps_visualization::msg::NavSatFixArray gps_fixes_msg = gps_visualization::msg::NavSatFixArray();
    GPSFixtoArray()
    : Node("gps_fix_to_array")
    {
    subscription_ =
        this->create_subscription<sensor_msgs::msg::NavSatFix>(
            "/spot1/gps/ublox_gps_node/fix",
            10,
            std::bind(&GPSFixtoArray::navsat_callback, this, std::placeholders::_1)
        );
    navsatarray_pub_ =
        this->create_publisher<gps_visualization::msg::NavSatFixArray>("/spot1/gps_path", 10);
    }

    void navsat_callback(const sensor_msgs::msg::NavSatFix &msg)
    {
        RCLCPP_INFO(this->get_logger(), "Received GPS data: Latitude: %f, Longitude: %f, Altitude: %f, Fix: %d, Status: %d",
                     msg.latitude, msg.longitude, msg.altitude, msg.status.status, msg.status.service);
        this->gps_fixes_msg.gps_fixes.push_back(msg);
        navsatarray_pub_->publish(this->gps_fixes_msg);
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr subscription_;
    rclcpp::Publisher<gps_visualization::msg::NavSatFixArray>::SharedPtr navsatarray_pub_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GPSFixtoArray>());
    rclcpp::shutdown();
    return 0;
}