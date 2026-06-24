"""Longitudinal slip injector (paper slip model) for TurtleBot3 in Gazebo.

Gazebo's Coulomb wheel friction is force-saturating and gives only a narrow,
uncontrollable slip window on the light, slow TB3 -- and it is a different model
from the paper's proportional slip ratio s where v_wheel = r*w*(1-s).  To
reproduce the paper's slip faithfully and controllably on the real stack, this
node sits between the controller and the robot:

    /cmd_vel_raw (controller)  ->  apply per-wheel (1-s)  ->  /cmd_vel (robot)

So the wheels "spin" at the controller's command but the body only achieves the
slipped velocity -- exactly the plant used in the validated simulation.  Slip
ratios step at t_right / t_left (default 30 s / 50 s), as in the paper.
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray


class SlipInjector(Node):
    def __init__(self):
        super().__init__('slip_injector')
        self.declare_parameter('r', 0.033)
        self.declare_parameter('b', 0.08)
        self.declare_parameter('sL0', 0.10)
        self.declare_parameter('sR0', 0.10)
        self.declare_parameter('sL1', 0.25)
        self.declare_parameter('sR1', 0.20)
        self.declare_parameter('t_left', 50.0)
        self.declare_parameter('t_right', 30.0)
        g = self.get_parameter
        self.r = g('r').value
        self.b = g('b').value
        self.sL0, self.sR0 = g('sL0').value, g('sR0').value
        self.sL1, self.sR1 = g('sL1').value, g('sR1').value
        self.t_left, self.t_right = g('t_left').value, g('t_right').value

        self.t0 = None
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.slip_pub = self.create_publisher(Float32MultiArray, 'true_slip', 10)
        self.create_subscription(Twist, 'cmd_vel_raw', self._cb, 10)
        self.get_logger().info(
            f"slip_injector: s0=({self.sL0},{self.sR0}) -> "
            f"s1=({self.sL1},{self.sR1}) at t_right={self.t_right}, "
            f"t_left={self.t_left}")

    def _slip(self, t):
        sR = self.sR0 if t < self.t_right else self.sR1
        sL = self.sL0 if t < self.t_left else self.sL1
        return sL, sR

    def _cb(self, msg: Twist):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.t0 is None:
            self.t0 = now
        t = now - self.t0
        sL, sR = self._slip(t)

        v, w = msg.linear.x, msg.angular.z
        wL = (v - self.b * w) / self.r        # commanded wheel angular speeds
        wR = (v + self.b * w) / self.r
        vL = self.r * wL * (1.0 - sL)          # achieved ground speeds (eq.1)
        vR = self.r * wR * (1.0 - sR)
        out = Twist()
        out.linear.x = 0.5 * (vR + vL)
        out.angular.z = (vR - vL) / (2.0 * self.b)
        self.pub.publish(out)
        self.slip_pub.publish(Float32MultiArray(data=[float(sL), float(sR)]))


def main(args=None):
    rclpy.init(args=args)
    node = SlipInjector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
