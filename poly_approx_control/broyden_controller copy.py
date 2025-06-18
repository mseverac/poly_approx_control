from geometry_msgs.msg import Point,Pose,PoseArray,Twist,Vector3
import rclpy
from rclpy.node import Node
import numpy as np
from ur_msgs.msg import PointArray
from pipy.tf import Rotation
from std_msgs.msg import Float64, Float64MultiArray,MultiArrayDimension,Float32MultiArray




M=4 # degree of polynomial
d=2 # dof of each point
k_angles = 100 #gain pour la commande des angles
file_path = "curve_poses_log.txt"

def read_curve_points(file_path):
    data = []
    current_curve = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line == '---':  # Separator between curves
                if current_curve:
                    data.append(np.array(current_curve))
                    current_curve = []
            elif line:  # Non-empty line
                values = list(map(float, line.split(',')))
                if len(values) == 4:  
                    x,y,z,w = values
                    values = Rotation.quaternion(x,y,z,w).get_rpy()
                    values = k_angles*np.array(values,dtype=np.float64)
                current_curve.append(values)
        if current_curve:  # Add the last curve if it exists
            data.append(np.array(current_curve))

    N = len(data)
    return np.array(data, dtype=object)  # Use dtype=object for jagged arrays

def computeWi(r):
    """r : iterable of scalar dof of effectors"""
    w = np.matrix([[]])
    for ri in r:
        w = np.hstack([w,np.matrix([[ ri**j for j in range(M) ]])])
    zero = np.zeros_like(w)
    W = np.vstack([np.hstack([w,zero]),np.hstack([zero,w])])    
    
    return W

def compute_dWi(r):
    """r : iterable of scalar dof of effectors"""

    dW = np.matrix([[]])

    for ri in r:

        dWdri = np.matrix([[j*ri**(j-1) for j in range(M)]])
        dW = np.hstack([dW, dWdri])

    return dW

def computeA(r, beta, i):
    beta_i = beta[i]
    m = len(r)

    A = None

    for k in range(d):
        a = np.matrix([[]], dtype=np.float64)
        for l in range(m):
            a = np.hstack([a, np.matrix([sum([j * beta_i[j + l * M + m * k * M] * r[l] ** (j - 1) for j in range(M)])], dtype=np.float64)])
        if A is None:
            A = a
        else:
            A = np.vstack([A, a])

    return A


def computeBetai(curve_points,j):

    s1v = np.array([curve_points[i,j,:2] for i in range(len(curve_points))]).flatten()
    W1v = np.vstack([computeWi(compute_r(curve)) for curve in curve_points]) 

    invW1v = np.linalg.pinv(W1v)
    betai = (invW1v @ s1v).transpose()

    return betai



def compute_r(curve):
    r = np.array([curve[0,0],curve[0,1],curve[1,1],curve[-2,0],curve[-2,1],curve[-1,1]]).flatten()
    return r


class PointController(Node):
    def __init__(self):
        super().__init__('point_controller')

        self.tcp_left,self.tcp_right = None, None

        self.publisher_r = self.create_publisher(Twist, '/right/vis_vel_cmd_6dof', 10)
        self.publisher_l = self.create_publisher(Twist, '/left/vis_vel_cmd_6dof', 10)

         
        self.create_subscription(
            Float32MultiArray,
            'points3d',
            self.points3d_callback,
            10
        )
        
        self.beta_sub = self.create_subscription(Float64MultiArray,"/beta",self.beta_cb,1)
        

        

        self.taget_sub = self.create_subscription(PointArray,"/curve_target",self.target_cb,1)
        
        
        self.sstar = None

        self.point_index = 5

        self.ka = 0.0003

        self.beta = None

        self.prev_error = np.zeros(6)
        self.integral_error = np.zeros(6)
        self.last_time = self.get_clock().now()

        self.Kp = 1.0
        self.Ki = 0.2
        self.Kd = 0.05


        self.points3d_data = None
        self.last_points3d_time = self.get_clock().now()



    def points3d_callback(self, msg):

        self.points3d_data = np.array(msg.data).reshape(-1, 3)
        #self.get_logger().info(f"Points3D data shape : {self.points3d_data.shape}")

        self.last_points3d_time = self.get_clock().now()
        #self.get_logger().info(f"last_points3d_time: {self.last_points3d_time}")



    def beta_cb(self, msg: Float64MultiArray):
        # Vérifie qu'on a bien 3 dimensions dans le layout
        if len(msg.layout.dim) != 3:
            self.get_logger().warn(f"beta_cb: unexpected layout dimensions: {len(msg.layout.dim)}")
            return

        # Récupération des tailles depuis le layout
        size_dim0 = msg.layout.dim[0].size
        size_dim1 = msg.layout.dim[1].size
        size_dim2 = msg.layout.dim[2].size

        total_expected_size = size_dim0 * size_dim1 * size_dim2

        if len(msg.data) != total_expected_size:
            self.get_logger().warn(f"beta_cb: data size mismatch: expected {total_expected_size}, got {len(msg.data)}")
            return

        # Conversion en tableau numpy
        beta_np = np.array(msg.data, dtype=np.float64).reshape((size_dim0, size_dim1, size_dim2))

        # Stockage
        self.beta = beta_np

    def target_cb(self,msg):

        self.sstar = np.array([[msg.points[i].x,msg.points[i].y] for i in range(len(msg.points))])
    
            
    def curve_cb(self,msg : PoseArray):
        
        if self.sstar is None:
            self.get_logger().info("sstar is not set")
            return
        
        if self.beta is None:
            self.get_logger().info("beta is not set")
            return
        
        
        orientation = msg.poses[0].orientation
        x,y,z,w = orientation.x,orientation.y,orientation.z,orientation.w
        values = Rotation.quaternion(x,y,z,w).get_rpy()
        values = k_angles*np.array(values,dtype=np.float64)

        self.tcp_left = np.array([msg.poses[0].position.x,msg.poses[0].position.y,values[1]])

        orientation = msg.poses[-1].orientation
        x,y,z,w = orientation.x,orientation.y,orientation.z,orientation.w
        values = Rotation.quaternion(x,y,z,w).get_rpy()
        values = k_angles*np.array(values,dtype=np.float64)



        self.tcp_right = np.array([msg.poses[-1].position.x,msg.poses[-1].position.y,values[1]])
        


        """file_path = "curve_poses_log.txt"
        curve_points = read_curve_points(file_path)
        beta = np.array([computeBetai(curve_points,j) for j in range(2,20)])

        if beta.all==self.beta.all:
            self.get_logger().info("beta is the same")
        else:
            self.get_logger().info("beta is not the same")
            self.get_logger().info(f"self.beta shape: {self.beta.shape}")
            self.get_logger().info(f"beta shape: {beta.shape}")
            for i in range(self.beta.shape[0]):
                for j in range(self.beta.shape[1]):
                    for k in range(self.beta.shape[2]):
                        self.get_logger().info(f"self.beta[{i}][{j}][{k}] = {self.beta[i][j][k]}, beta[{i}][{j}][{k}] = {beta[i][j][k]}")

                        if self.beta[i][j][k] != beta[i][j][k]:
                            self.get_logger().error(f"self.beta[{i}][{j}][{k}] != beta[{i}][{j}][{k}]")
                            break"""


            

        r= np.array([self.tcp_left,self.tcp_right]).flatten()

        dr = np.zeros_like(r)

        normds = 0


        with open("data.txt", "a") as file:
            for index, sstari in enumerate(self.sstar):
                Ai = computeA(r, self.beta, index)
                invAi = np.linalg.pinv(Ai)

                point = msg.poses[index + 1].position
                s = (point.x, point.y)
                ds = -np.matrix([[s[0] - sstari[0]], [s[1] - sstari[1]]])
                normds += np.linalg.norm(ds)

                dri = invAi @ ds
                dr += np.array(dri).flatten()

                # Log ds and dr with timestamp
                timestamp = self.get_clock().now().to_msg()
                file.write(f"Timestamp: {timestamp.sec}.{timestamp.nanosec}, i: {index}, ds: {ds.tolist()}, dr: {np.array(dri).flatten().tolist()}\n")

        dr = np.array(dr).flatten() / len(self.sstar)

        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds * 1e-9
        self.last_time = current_time


        error = dr
        self.integral_error += error * dt
        self.integral_error = np.clip(self.integral_error, -np.array([50,50,5,50,50,5]),np.array([50,50,5,50,50,5]))  # Limiter l'erreur intégrale
        derivative_error = (error - self.prev_error) / dt
        self.prev_error = error

        # Commande PID
        u = self.Kp * error + self.Ki * self.integral_error + self.Kd * derivative_error




        dr = u


        cmd_right = Vector3(x=self.ka*float(dr[1]),y=self.ka*float(dr[0]),z=0.0)
        cmd_left = Vector3(x=self.ka*float(dr[4]),y=self.ka*float(dr[3]),z=0.0)

        cmd_right = Twist(linear=cmd_right,angular=Vector3(x=0.0,y=0.0,z=-10*self.ka*float(dr[2])))
        cmd_left = Twist(linear=cmd_left,angular=Vector3(x=0.0,y=0.0,z=10*self.ka*float(dr[5])))

        self.publisher_l.publish(cmd_left)
        self.publisher_r.publish(cmd_right)

        self.get_logger().info(f"tcp_left : {self.tcp_left}")
        self.get_logger().info(f"tcp_right : {self.tcp_right}")

        self.get_logger().info(f"r : {r}")

        self.get_logger().info(f"error : {error}")
        self.get_logger().info(f"differential error : {derivative_error}")
        self.get_logger().info(f"integral error : {self.integral_error}")
        self.get_logger().info(f"cmd PID : {dr}")
        self.get_logger().info(f"norme ds : {normds}")
        self.get_logger().info(f"cmd_left : {cmd_left}")
        self.get_logger().info(f"cmd_right : {cmd_right}")
        self.get_logger().info(f"updated")

        



def main(args=None):
    rclpy.init(args=args)

    point_controller = PointController()

    rclpy.spin(point_controller)

    point_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()











        










    




    



