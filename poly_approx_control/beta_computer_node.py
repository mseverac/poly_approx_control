import numpy as np
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import TransformStamped
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float64MultiArray,MultiArrayDimension,Float32MultiArray
import time

M = 3 # degree of polynomial
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

def trans_to_matrix(trans: TransformStamped):
  
    pos = trans.transform.translation
    ori = trans.transform.rotation

    r_curr = R.from_quat([ori.x, ori.y, ori.z, ori.w])

    return r_curr.as_matrix(),np.array([pos.x, pos.y, pos.z])

def trans_to_vecs(trans: TransformStamped):
  
    pos = trans.transform.translation
    ori = trans.transform.rotation

    r_curr = R.from_quat([ori.x, ori.y, ori.z, ori.w])

    return r_curr.as_rotvec(),np.array([pos.x, pos.y, pos.z])

def compute_r_from_transes(tr :TransformStamped,tl :TransformStamped):
    """Compute the r vector from the transforms of the left and right end effectors"""
    r_left, pos_left = trans_to_vecs(tl)
    r_right, pos_right = trans_to_vecs(tr)

    r = np.array([pos_right[0],pos_right[1],pos_right[2],
                  r_right[0],r_right[1],r_right[2],
                  pos_left[0],pos_left[1],pos_left[2],
                  r_left[0],r_left[1],r_left[2]], dtype=np.float64)
    
    return r

def compute_r(r_pos, l_pos, r_R, l_R):
    """Compute the r vector from the positions and orientations of the left and right end effectors"""

    vr = R.from_matrix(r_R).as_rotvec()
    vl = R.from_matrix(l_R).as_rotvec()
    r = np.array([r_pos[0], r_pos[1], r_pos[2],
                vr[0], vr[1], vr[2],
                l_pos[0], l_pos[1], l_pos[2],
                vl[0], vl[1], vl[2]]
                , dtype=np.float64)
    
    return r

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


def compute_w(r):
    """r : iterable of scalar dof of effectors"""
    w = np.array([[]])
    for ri in r:
        w = np.hstack([w,np.array([[ ri**j for j in range(M) ]])])
    
    return w

def compute_beta(r_poses, l_poses, r_Rs, l_Rs, points_between_tcpes):
    """Compute the beta coefficients for the polynomial approximation of the curves between the end effectors"""
    
    
    wv = np.array([compute_w(compute_r(r_poses[i], l_poses[i], r_Rs[i], l_Rs[i])) for i in range(len(r_poses))])
    wv = wv.reshape(-1,12*M)


    N = len(r_poses)

    points_between_tcpes = np.array(points_between_tcpes).reshape(N,-1)


    Beta = []

    for i in range(153):
        siv = points_between_tcpes[:,i]

        #print(f"siv shape: {siv.shape}")
        """print("--------------")
        print(f"s{i}v : {siv[:10]}")
        print("**********************")
        print(f"wv xr: {wv[:,1+i*M][:10]}")
        print("--------------")"""

        #print(f"wv shape: {wv.shape}")

        wp = np.linalg.pinv(wv)
        #print(f"wp shape: {wp.shape}")
        beta_i = wp @ siv

        #print(f"beta_{i} shape: {beta_i.shape}")

        Beta.append(beta_i)


        #print(f"beta_{i} : {beta_i.reshape(12,M)}")

        """a = compute_a(r0, beta_i)
        print(f"a shape: {a.shape}")
        print(f"a_{i}: {a}")"""

    return np.array(Beta)

def compute_a(r, beta):
    "compute a row of the A matrix for a given r and beta_i (shape : 12*M,)"

    b = beta.reshape(12, M)  
    a = np.zeros(12, dtype=np.float32)
    for j in range(12):
        s = 0
        for i in range(1,M):
            s += b[j, i] * i * r[i] ** (i - 1)
        a[j] = s

    return a

def compute_A(B,r):

    A = []
    for beta_i in B:
        a = compute_a(r, beta_i)
        A.append(a)
    A = np.array(A)
    return A
        

class BetaComputerNode(Node):
    def __init__(self):
        super().__init__('beta_computer_node')
        
        self.A_pub = self.create_publisher(Float64MultiArray, '/A_init', 10)


        r_poses, l_poses, r_Rs, l_Rs, points_between_tcpes =read_txt_file("curve_points_datas.txt")

        self.get_logger().info(f"computing beta for {len(r_poses)} curves with {len(points_between_tcpes[0])} points each.")

        self.Beta = compute_beta(r_poses, l_poses, r_Rs, l_Rs, points_between_tcpes)
        self.get_logger().info(f"finished computing beta : Beta shape: {self.Beta.shape}")

        self.create_subscription(
            TransformStamped,
            '/tcp_left',
            self.tcp_left_callback,
            10
        )

        self.create_subscription(
            TransformStamped,
            '/tcp_right',
            self.tcp_right_callback,
            10
        )

        self.create_timer(0.1, self.publish_A)  
        
    def tcp_right_callback(self, msg):
        self.tcp_right = msg

    def tcp_left_callback(self, msg):
        self.tcp_left = msg


    def publish_A(self):
        if hasattr(self, 'tcp_left') and hasattr(self, 'tcp_right'):
            r = compute_r_from_transes(self.tcp_right, self.tcp_left)
            A = compute_A(self.Beta, r)

            A_msg = Float64MultiArray()
            A_msg.data = A.flatten().tolist()
            self.A_pub.publish(A_msg)
            self.get_logger().info(f"Published A matrix with shape: {A.shape}")
            self.get_logger().info(f"First row of A: {A[0]}")
            self.get_logger().info(f"Second row of A: {A[1]}")
            self.get_logger().info(f"Third row of A: {A[2]}")
            self.get_logger().info(f"-3 row of A: {A[-3]}")
            self.get_logger().info(f"-2 row of A: {A[-2]}")
            self.get_logger().info(f"-1 row of A: {A[-1]}")

        else:
            self.get_logger().info("Waiting for tcp_left and tcp_right transforms to compute A matrix.")
            time.sleep(1)


def main(args=None):
    rclpy.init(args=args)
    node = BetaComputerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('BetaComputerNode has been stopped.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

    


   

    
