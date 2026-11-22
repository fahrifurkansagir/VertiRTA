cmake_minimum_required(VERSION 3.22)
project(vertirta_msgs VERSION 1.0.0 LANGUAGES C CXX)

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/SlzPrior.msg"
  "msg/SlzPriorArray.msg"
  "msg/WireCapsuleArray.msg"
  "msg/IntruderArray.msg"
  "msg/RtaStateSnapshot.msg"
  "srv/InjectFault.srv"
  "action/ExecuteContingency.action"
  DEPENDENCIES builtin_interfaces geometry_msgs std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
