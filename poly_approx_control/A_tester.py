import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Float32MultiArray
import numpy as np
import matplotlib.pyplot as plt

class ATester(Node):
    def __init__(self):
        super().__init__('a_tester')
        self.subscription = self.create_subscription(
            Float64MultiArray,
            'A_init',
            self.listener_callback,
            10
        )

        self.create_subscription(
            Float32MultiArray,
            'cosserat_shape',
            self.shape_cb,
            10

        )
        self.subscription  # prevent unused variable 
        
        self.A = None

        self.s = None

    def listener_callback(self, msg):
        self.A = np.array(msg.data).reshape(153,12)

    def shape_cb(self, msg):
        self.s = np.array(msg.data).reshape(-1, )

        self.get_logger().info(f"Received shape data: {self.s}")

        if self.A is not None :

            drs = np.eye(12) * 0.01
            for dr in drs:
                self.get_logger().info(f"Computing ds for dr: {dr.transpose()}")
                ds = self.A @ dr.transpose()
                s1 = self.s + ds

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

                def plot_cable(s,color='b'):
                    """
                    Plots a 3D cable given its points.
                    
                    Parameters:
                        s (numpy.ndarray): A (51, 3) array representing the cable points (x, y, z).
                    """
                    if s.shape != (51, 3):
                        s = s.reshape(-1, 3)  # Ensure s is reshaped to (51, 3)
                    

                    
                    ax.plot(s[:, 0], s[:, 1], s[:, 2], marker='o', label='Cable', color=color)
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    ax.set_title('3D Cable Plot')
                    ax.legend()

                plot_cable(s1, color='r')
                plot_cable(self.s, color='b')
                plt.show()


       



def main(args=None):
    rclpy.init(args=args)
    node = ATester()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()