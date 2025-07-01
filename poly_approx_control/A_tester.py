import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Float32MultiArray
import numpy as np
import matplotlib.pyplot as plt
from rclpy.wait_for_message import wait_for_message
from scipy.spatial.transform import Rotation as R




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

            dx = 0.05
            da = 0.1

            def plot_jac_3d(A,s,dx=0.05,da=0.5):

                drs = np.diag([dx,dx,dx,da,da,da,dx,dx,dx,da,da,da]) 
                fig, axs = plt.subplots(3, 4, subplot_kw={'projection': '3d'}, figsize=(15, 10))
                axs = axs.flatten()  # Flatten the 3x4 grid for easier indexing

                for i, dr in enumerate(drs):
                    self.get_logger().info(f"Computing ds for dr: {dr.transpose()}")
                    ds = A @ dr.transpose()
                    s1 = s + ds

                    def plot_cable(ax, s, color='b', label='Cable'):
                        """
                        Plots a 3D cable given its points on a specific subplot.
                        
                        Parameters:
                            ax (matplotlib.axes._subplots.Axes3DSubplot): The subplot to plot on.
                            s (numpy.ndarray): A (51, 3) array representing the cable points (x, y, z).
                            color (str): Color of the cable plot.
                            label (str): Label for the cable plot.
                        """
                        if s.shape != (51, 3):
                            s = s.reshape(-1, 3)  # Ensure s is reshaped to (51, 3)
                        
                        ax.plot(s[:, 0], s[:, 1], s[:, 2], marker='o', label=label, color=color)
                        ax.set_xlabel('X')
                        ax.set_ylabel('Y')
                        ax.set_zlabel('Z')
                        ax.set_title(f'Cable {i+1}')
                        ax.legend()
                        
                        # Set equal aspect ratio for isometric view
                        x_limits = ax.get_xlim()
                        y_limits = ax.get_ylim()
                        z_limits = ax.get_zlim()
                        max_range = max(
                            x_limits[1] - x_limits[0],
                            y_limits[1] - y_limits[0],
                            z_limits[1] - z_limits[0]
                        ) / 2.0

                        mid_x = (x_limits[0] + x_limits[1]) * 0.5
                        mid_y = (y_limits[0] + y_limits[1]) * 0.5
                        mid_z = (z_limits[0] + z_limits[1]) * 0.5

                        ax.set_xlim(mid_x - max_range, mid_x + max_range)
                        ax.set_ylim(mid_y - max_range, mid_y + max_range)
                        ax.set_zlim(mid_z - max_range, mid_z + max_range)

                    plot_cable(axs[i], s1, color='r', label='Modified Cable')
                    plot_cable(axs[i], s, color='b', label='Original Cable')

                plt.tight_layout()
                plt.show()

            def plot_jac_2d(A, s, dx=0.05, da=0.5):
                import matplotlib.pyplot as plt
                import numpy as np

                drs = np.diag([dx, dx, dx, da, da, da, dx, dx, dx, da, da, da])

                def plot_projection(ax, s_orig, s_mod, proj='zx', title=''):
                    """
                    Plots a 2D projection of the cable deformation.
                    
                    Parameters:
                        ax: matplotlib axis
                        s_orig: original cable shape (51,3)
                        s_mod: modified cable shape (51,3)
                        proj: 'zx' or 'zy'
                        title: title of the subplot
                    """
                    s_orig = s_orig.reshape(-1, 3)
                    s_mod = s_mod.reshape(-1, 3)

                    if proj == 'zx':
                        ax.plot(s_orig[:, 0], s_orig[:, 2], label='Original', color='b')
                        ax.plot(s_mod[:, 0], s_mod[:, 2], label='Modified', color='r')
                        ax.set_xlabel('X')
                        ax.set_ylabel('Z')
                    elif proj == 'zy':
                        ax.plot(s_orig[:, 1], s_orig[:, 2], label='Original', color='b')
                        ax.plot(s_mod[:, 1], s_mod[:, 2], label='Modified', color='r')
                        ax.set_xlabel('Y')
                        ax.set_ylabel('Z')

                    ax.set_title(title)
                    ax.grid(True)

                for window in range(2):
                    fig, axs = plt.subplots(2, 6, figsize=(18, 6))  # 2 rows: zx and zy, 6 columns
                    axs = axs.reshape(2, 6)

                    for i in range(6):
                        idx = i + window * 6
                        dr = drs[idx]
                        ds = A @ dr
                        s1 = s + ds

                        plot_projection(axs[0, i], s, s1, proj='zx', title=f'dr[{idx}] - Z vs X')
                        plot_projection(axs[1, i], s, s1, proj='zy', title=f'dr[{idx}] - Z vs Y')

                    plt.tight_layout()
                    plt.suptitle(f'Jacobian Columns {window*6 + 1} to {window*6 + 6}', fontsize=16)
                    plt.subplots_adjust(top=0.88)
                    plt.show()


            plot_jac_2d(self.A,self.s,dx=dx,da=da)

            plot_jac_3d(self.A,self.s,dx=dx,da=da)

            self.get_logger().info("Waiting fr target to plot example cmd ")

            _,msg = wait_for_message(Float32MultiArray,self,'/curve_target_6dof' )
            target = np.array(msg.data).reshape(-1,)


            def decompose_r(r):
                """Decompose the r vector into positions and rotation matrices for left and right end effectors"""
                
                # Extraire les positions
                r_pos = np.array(r[0:3], dtype=np.float64)
                vr = np.array(r[3:6], dtype=np.float64)
                l_pos = np.array(r[6:9], dtype=np.float64)
                vl = np.array(r[9:12], dtype=np.float64)
                
                # Convertir les vecteurs de rotation en matrices de rotation
                r_R = R.from_rotvec(vr.transpose()).as_matrix()
                l_R = R.from_rotvec(vl.transpose()).as_matrix()
                
                return r_pos, l_pos, vr, vl
            
            def test_cmd(s):

                ds = target - s

                J = np.linalg.pinv(self.A)

                dr = J @ ds

                dr_pos, dl_pos, dvr, dvl = decompose_r(dr)

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')


                def plot_cable2(s,color='b'):
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
    


                rp = s[0:3]
                lp = s[-3:]


                # Plot position vectors
                ax.quiver(rp[0],rp[1],rp[2], dr_pos[0], dr_pos[1], dr_pos[2], color='red', label='dr_pos')
                ax.quiver(lp[0],lp[1],lp[2], dl_pos[0], dl_pos[1], dl_pos[2], color='red', label='dl_pos')

                # Plot rotation vectors
                ax.quiver(rp[0], rp[1], rp[2], dvr[0], dvr[1], dvr[2], color='blue', label='dvr')
                ax.quiver(lp[0], lp[1], lp[2], dvl[0], dvl[1], dvl[2], color='blue', label='dvl')
                ax.legend()
                # Example usage
                plot_cable2(s)
                plot_cable2(target, color='green')

                plt.show()

            
            test_cmd(self.s)

            P = 1

            def read_txt_file(file_path):
                with open(file_path, 'r') as file:
                    lines = file.readlines()

                r_pos, l_pos, r_R, l_R = [], [], [], []
                points_between_tcp = []
                curves = []

                i=0

                for line in lines:
                    line = line.strip()
                    if line.startswith("Right TCP:"):
                        i+=1

                        if i%P == 0 :r_pos.append(np.array([float(x) for x in line.split(":")[1].split(",")]))

                    elif line.startswith("Left TCP:"):
                        if i%P == 0 :l_pos.append(np.array([float(x) for x in line.split(":")[1].split(",")]))

                        if i%P == 0 :curves.append(points_between_tcp)
                        points_between_tcp = []  # Reset for the next curve

                    elif line.startswith("Right Orientation:"):
                        r_quat = [float(x) for x in line.split(":")[1].split(",")]
                        if i%P == 0 :r_R.append(R.from_quat(r_quat).as_matrix())

                    elif line.startswith("Left Orientation:"):
                        l_quat = [float(x) for x in line.split(":")[1].split(",")]
                        if i%P == 0 :l_R.append(R.from_quat(l_quat).as_matrix())
                    elif line.startswith("(") and line.endswith(")"):
                        point = [float(x) for x in line.strip("()").split(",")]
                        if i%P == 0 :points_between_tcp.append(point)

                curves = np.array(curves)

                return r_pos, l_pos, r_R, l_R, curves
            
            r_poses, l_poses, r_Rs, l_Rs, points_between_tcpes =read_txt_file("curve_points_datas.txt")

            self.get_logger().info(f"Read {len(points_between_tcpes)} curves from file.")

            for i, s in enumerate(points_between_tcpes):
                self.get_logger().info(f"s[-1]: {s[-1]}, s[0]: {s[0]}, r_poses[{i}]: {r_poses[i]}, l_poses[{i}]: {l_poses[i]}")
                print("------------------------")
                print(f"dist right : {np.linalg.norm(s[0]-r_poses[i])}")
                print(f"dist left : {np.linalg.norm(s[-1]-l_poses[i])}")

                for j in range(1, 6):
                    print(f"dist right s{j} : {np.linalg.norm(s[j]-r_poses[i])}")
                    print(f"dist left s{j}  : {np.linalg.norm(s[-j-1]-l_poses[i])}")

                print("------------------------")

            for s in points_between_tcpes:
                test_cmd(np.array(s).reshape(-1,))

            self.destroy_node()


       



def main(args=None):
    rclpy.init(args=args)
    node = ATester()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()