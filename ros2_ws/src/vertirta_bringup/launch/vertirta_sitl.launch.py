"""ROS 2 Jazzy SITL bringup launch file for VertiRTA-X8-240 (Plan 1A Task 12)."""

from __future__ import annotations

FORBIDDEN_PORTS = frozenset({4173, 4174, 8765, 9222, 9223})
DEFAULT_FOXGLOVE_PORT = 8891
DEFAULT_MICRO_XRCE_PORT = 8888


def validate_sitl_ports(*, foxglove_port: int, micro_xrce_port: int) -> None:
    """Reject occupied/forbidden workspace ports before spawning bridge daemons."""
    for name, port in (("foxglove_port", foxglove_port), ("micro_xrce_port", micro_xrce_port)):
        if port in FORBIDDEN_PORTS:
            raise ValueError(
                f"{name}={port} is in forbidden occupied ports {sorted(FORBIDDEN_PORTS)}"
            )
    if foxglove_port == micro_xrce_port:
        raise ValueError(f"foxglove_port and micro_xrce_port must differ (both {foxglove_port})")


def _launch_setup(context, *_args, **_kwargs):
    from launch.actions import ExecuteProcess, SetEnvironmentVariable
    from launch.substitutions import LaunchConfiguration
    from launch_ros.actions import Node

    use_indi = LaunchConfiguration("use_indi").perform(context)
    world = LaunchConfiguration("world").perform(context)
    vehicle_namespace = LaunchConfiguration("vehicle_namespace").perform(context)
    gz_partition = LaunchConfiguration("gz_partition").perform(context)
    micro_xrce_port = int(LaunchConfiguration("micro_xrce_port").perform(context))
    foxglove_port = int(LaunchConfiguration("foxglove_port").perform(context))

    validate_sitl_ports(foxglove_port=foxglove_port, micro_xrce_port=micro_xrce_port)

    return [
        SetEnvironmentVariable("VERTIRTA_USE_INDI", use_indi),
        SetEnvironmentVariable("PX4_GZ_WORLD", world),
        SetEnvironmentVariable("GZ_PARTITION", gz_partition),
        ExecuteProcess(
            cmd=["MicroXRCEAgent", "udp4", "-p", str(micro_xrce_port)],
            output="screen",
        ),
        Node(
            package="foxglove_bridge",
            executable="foxglove_bridge",
            name="foxglove_bridge",
            parameters=[{"port": foxglove_port, "send_buffer_limit": 10000000}],
            output="screen",
        ),
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            name="vertirta_gz_bridge",
            arguments=[
                "/vertirta/lidar_16@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
                "/vertirta/fault/motor_mask@std_msgs/msg/UInt32]gz.msgs.UInt32",
                "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            ],
            output="screen",
        ),
        Node(
            package="vertirta_px4_bridge",
            executable="vertirta_px4_bridge_node",
            name="vertirta_px4_bridge",
            namespace=vehicle_namespace,
            parameters=[{"vehicle_namespace": vehicle_namespace}],
            output="screen",
        ),
        Node(
            package="vertirta_rta_node",
            executable="vertirta_rta_node",
            name="vertirta_rta_node",
            namespace=vehicle_namespace,
            output="screen",
        ),
        Node(
            package="vertirta_map_node",
            executable="vertirta_map_node",
            name="vertirta_map_node",
            namespace=vehicle_namespace,
            output="screen",
        ),
        Node(
            package="vertirta_fault_injector",
            executable="vertirta_fault_injector_node",
            name="vertirta_fault_injector",
            namespace=vehicle_namespace,
            output="screen",
        ),
    ]


def generate_launch_description():
    """Build ROS 2 Jazzy LaunchDescription for VertiRTA SITL stack."""
    from launch import LaunchDescription
    from launch.actions import DeclareLaunchArgument, OpaqueFunction

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_indi", default_value="1"),
            DeclareLaunchArgument("world", default_value="kartal_coast"),
            DeclareLaunchArgument("vehicle_namespace", default_value=""),
            DeclareLaunchArgument("gz_partition", default_value="vertirta_live"),
            DeclareLaunchArgument("micro_xrce_port", default_value=str(DEFAULT_MICRO_XRCE_PORT)),
            DeclareLaunchArgument("foxglove_port", default_value=str(DEFAULT_FOXGLOVE_PORT)),
            OpaqueFunction(function=_launch_setup),
        ]
    )
