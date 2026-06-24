"""ROS 2 node: slipping trajectory-tracking control on TurtleBot3.

Subscribes to the robot pose (wheel odometry or AMCL), runs the paper's
kinematic backstepping + adaptive slip controller, and publishes cmd_vel.

The reference trajectory is anchored to the robot's start position so tracking
begins with ~zero position error (and stays inside the Gazebo free space).
"""
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray

from wmr_tb3.controller_core import KinematicAdaptiveController, TB3Params, pose_error
from wmr_tb3.trajectories import make_trajectory


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class TrackingNode(Node):
    def __init__(self):
        super().__init__('wmr_tracking')

        # --- parameters ---
        self.declare_parameter('trajectory', 'circle')   # 'circle' | 'line'
        self.declare_parameter('mode', 'paper')          # 'paper' | 'nocomp'
        self.declare_parameter('pose_source', 'odom')    # 'odom' | 'amcl'
        self.declare_parameter('control_rate', 20.0)     # [Hz]
        self.declare_parameter('duration', 0.0)          # 0 -> run forever
        # trajectory shape
        self.declare_parameter('circle_radius', 1.0)
        self.declare_parameter('circle_wr', 0.12)        # -> v = R*wr
        self.declare_parameter('line_speed', 0.12)
        self.declare_parameter('line_alpha', 0.0)
        # controller gains (override defaults if desired)
        self.declare_parameter('Hx', TB3Params.Hx)
        self.declare_parameter('Hy', TB3Params.Hy)
        self.declare_parameter('Hs', TB3Params.Hs)

        gp = self.get_parameter
        self.traj_name = gp('trajectory').value
        self.mode = gp('mode').value
        self.pose_source = gp('pose_source').value
        self.rate = float(gp('control_rate').value)
        self.duration = float(gp('duration').value)

        p = TB3Params()
        p.Hx = float(gp('Hx').value)
        p.Hy = float(gp('Hy').value)
        p.Hs = float(gp('Hs').value)
        self.p = p

        if self.traj_name == 'circle':
            self.traj = make_trajectory('circle',
                                        R=float(gp('circle_radius').value),
                                        wr=float(gp('circle_wr').value))
        else:
            self.traj = make_trajectory('line',
                                        vr=float(gp('line_speed').value),
                                        alpha=float(gp('line_alpha').value))

        self.ctrl = KinematicAdaptiveController(p, adaptive=(self.mode == 'paper'))

        # --- state ---
        self.pose = None          # (x, y, theta)
        self.w_meas = 0.0         # measured body angular velocity
        self.t0 = None            # start time
        self.p_start = (0.0, 0.0)  # robot start position
        self.ref0 = (0.0, 0.0)     # reference start position
        self.dth = 0.0             # heading offset (align ref to robot)

        # --- I/O ---
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.err_pub = self.create_publisher(Float32MultiArray, 'tracking_error', 10)
        if self.pose_source == 'amcl':
            self.create_subscription(PoseWithCovarianceStamped, 'amcl_pose',
                                     self._amcl_cb, 10)
        elif self.pose_source == 'truth':
            self.create_subscription(Odometry, 'odom_truth', self._odom_cb, 10)
        else:
            self.create_subscription(Odometry, 'odom', self._odom_cb, 10)

        self.dt = 1.0 / self.rate
        self.timer = self.create_timer(self.dt, self._control_step)
        self.get_logger().info(
            f"wmr_tracking: traj={self.traj_name} mode={self.mode} "
            f"pose={self.pose_source} rate={self.rate}Hz")

    # ------------------------------------------------------------------ #
    def _odom_cb(self, msg: Odometry):
        po = msg.pose.pose
        self.pose = (po.position.x, po.position.y, yaw_from_quat(po.orientation))
        self.w_meas = msg.twist.twist.angular.z

    def _amcl_cb(self, msg: PoseWithCovarianceStamped):
        po = msg.pose.pose
        self.pose = (po.position.x, po.position.y, yaw_from_quat(po.orientation))

    # ------------------------------------------------------------------ #
    def _ref(self, t):
        # rigid transform: place the reference's start at the robot's start pose
        # and rotate by dth so the initial heading error is zero too.
        (xr, yr, thr), vr, wr = self.traj.state(t)
        dx = xr - self.ref0[0]
        dy = yr - self.ref0[1]
        c, s = math.cos(self.dth), math.sin(self.dth)
        rx = self.p_start[0] + c * dx - s * dy
        ry = self.p_start[1] + s * dx + c * dy
        from wmr_tb3.controller_core import wrap
        return (rx, ry, wrap(thr + self.dth)), vr, wr

    def _control_step(self):
        if self.pose is None:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.t0 is None:
            self.t0 = now
            # anchor the trajectory's start pose to the robot's start pose
            from wmr_tb3.controller_core import wrap
            (xr0, yr0, thr0), _, _ = self.traj.state(0.0)
            self.p_start = (self.pose[0], self.pose[1])
            self.ref0 = (xr0, yr0)
            self.dth = wrap(self.pose[2] - thr0)
            self.get_logger().info(
                f"anchored at {self.pose[:2]}, heading offset {self.dth:.2f} rad")
        t = now - self.t0

        if self.duration > 0.0 and t > self.duration:
            self.pub.publish(Twist())   # stop
            return

        ref, vr, wr = self._ref(t)
        e = pose_error(self.pose, ref)
        vc, wc = self.ctrl.virtual_velocity(e, vr, wr, self.w_meas)
        self.ctrl.adapt(e, vc, wc, self.dt)
        v, w = self.ctrl.cmd_vel(vc, wc)

        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(w)
        self.pub.publish(cmd)

        sL, sR = self.ctrl.slip_estimate
        self.err_pub.publish(Float32MultiArray(
            data=[float(e[0]), float(e[1]), float(e[2]), float(sL), float(sR)]))


def main(args=None):
    rclpy.init(args=args)
    node = TrackingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub.publish(Twist())   # stop the robot
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
