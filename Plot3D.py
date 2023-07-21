"""
A Python library, inspired by Wolfram Mathematica, that facilitates the 
plotting of objects in 3D space, particularly useful for implementing 
camera models and epipolar geometry.
"""
import numpy as np 
import matplotlib.pyplot as plt 
import plotly.graph_objects as go

# ========================================================
# 3D objects
# ========================================================
class Figure:

    def __init__(self):
        self.fig_data = []

    def get_fig_data(self):
        return self.fig_data

class Arrow(Figure):

    def __init__(self, start, end, color='black', name=None):
        super().__init__()
        start = start.flatten()
        end = end.flatten()
        head_scale = 0.3 # adjusts the cone size scaling 

        body = go.Scatter3d(x = [start[0], end[0]],
                            y = [start[1], end[1]],
                            z = [start[2], end[2]], 
                            mode = "lines",
                            line = dict(color=color), 
                            showlegend = False)
        
        head = go.Cone(x = [end[0]], y = [end[1]], z = [end[2]],
                    u = [end[0]-start[0]], v = [end[1]-start[1]], w = [end[2]-start[2]],
                    sizemode = "scaled",
                    sizeref = head_scale,
                    colorscale = [[0, color], [1, color]],
                    showscale=False,
                    showlegend=False)
        
        dist = 0.4
        text = go.Scatter3d(x = [-dist * start[0] + (1 + dist) * end[0]],
                            y = [-dist * start[1] + (1 + dist) * end[1]],
                            z = [-dist * start[2] + (1 + dist) * end[2]],
                            mode = "text",
                            text = [name],
                            textposition = 'middle center',
                            showlegend = False,
                            textfont = dict(color=color))
        self.fig_data = [body, head, text]

class Frame(Figure):
    
    def __init__(self, pose=np.eye(3), center=np.zeros(3), color='black'):
        super().__init__()
        # original center and pose 
        x_axis = np.array([1, 0, 0])
        y_axis = np.array([0, 1, 0])
        z_axis = np.array([0, 0, 1])

        # transformed center and pose  
        x_axis = pose @ x_axis
        y_axis = pose @ y_axis
        z_axis = pose @ z_axis
        o = center

        fig_data = []
        fig_data.extend(Arrow(o, o + x_axis, color=color, name='x').get_fig_data())
        fig_data.extend(Arrow(o, o + y_axis, color=color, name='y').get_fig_data())
        fig_data.extend(Arrow(o, o + z_axis, color=color, name='z').get_fig_data())
        self.fig_data = fig_data

# Line segment
class Segment(Figure):

    def __init__(self, start, end, color='black', opacity=1):
        super().__init__()
        start, end = self.input_broadcast(start, end)
        fig_data = []
        for _start, _end in zip(start, end):
            fig_data.append(go.Scatter3d(
                x = [_start[0], _end[0]], 
                y = [_start[1], _end[1]], 
                z = [_start[2], _end[2]], 
                mode = "lines", line = dict(color=color), opacity=opacity, showlegend = False))
        self.fig_data = fig_data

    # ===== to allow 1-to-1, 1-to-N, N-to-1, N-to-M inputs =====
    # case 1: a: (3, ), b: (3, ) ->  (1, 3), b: (1, 3)
    # case 2: a: (N, 3), b: (3, ) -> (N, 3), b: (N, 3)
    # case 3: a: (3, ), b: (N, 3) -> (N, 3), b: (N, 3)
    # case 4: a: (N, 3), b: (M, 3) -> (N*M, 3), b: (N*M, 3)
    def input_broadcast(self, a, b):
        a, b, = a.reshape(-1, 3), b.reshape(-1, 3)
        Na, Nb = a.shape[0], b.shape[0]
        new_a = np.repeat(a, Nb, axis=0)
        new_b = np.repeat(b, Na, axis=0)
        new_b = new_b.reshape(Nb, Na, 3).transpose(1, 0, 2).reshape(-1, 3)
        return new_a, new_b
    
# plot "infinitely" lone line
class Line(Segment):

    def __init__(self, start, end, color='black', opacity=1):
        start, end = super().input_broadcast(start, end)
        
        a_big_num = 20
        length = np.linalg.norm(start - end, axis=-1) + 1e-15
        coef = a_big_num / length
        coef = coef.reshape(-1, 1)
        start2end = end - start 
        # basically extend the line segment to "infinitely" long
        start = start - coef * start2end
        end = end + coef * start2end
        super().__init__(start, end, color, opacity)

# include image plane and the 4 lines from focal point to the 4 corners of the image plane
class ImagePlane(Figure):

    def __init__(self, pose, center, focal_len, img_width, img_height, color='black'):
        super().__init__()
        x_axis = np.array([1, 0, 0])
        y_axis = np.array([0, 1, 0])
        z_axis = np.array([0, 0, 1])
        x_axis = pose @ x_axis 
        y_axis = pose @ y_axis
        z_axis = pose @ z_axis
        o = center # center of the camera coordinate frame
        img_center = o + z_axis * focal_len # center of the image plane 

        # 4 corners of the image plane 
        corner1 = img_center - (img_width / 2) * x_axis - (img_height / 2) * y_axis
        corner2 = img_center + (img_width / 2) * x_axis - (img_height / 2) * y_axis
        corner3 = img_center + (img_width / 2) * x_axis + (img_height / 2) * y_axis
        corner4 = img_center - (img_width / 2) * x_axis + (img_height / 2) * y_axis
        corners = np.vstack([corner1, corner2, corner3, corner4])
        
        opcty = 0.3 # opacity of the image plane and lines
        data = [go.Mesh3d(x=corners[:, 0], y=corners[:, 1], z=corners[:, 2], 
                        i=[0, 0],  # vertices of first triangle
                        j=[1, 2],  # vertices of second triangle
                        k=[2, 3],  # vertices of third triangle
                        opacity=opcty, color=color)]
        data.extend(Segment(o, corner1, color=color, opacity=opcty).get_fig_data())
        data.extend(Segment(o, corner2, color=color, opacity=opcty).get_fig_data())
        data.extend(Segment(o, corner3, color=color, opacity=opcty).get_fig_data())
        data.extend(Segment(o, corner4, color=color, opacity=opcty).get_fig_data())

        self.fig_data = data

# cannot be capture by camera, I guess...
class Point(Figure):
    def __init__(self, pts, color='black', size=2):
        super().__init__()
        fig_data = go.Scatter3d(
            x = [pts[0]], 
            y = [pts[1]], 
            z = [pts[2]], 
            mode = "markers", marker = dict(color=color, size=size), showlegend = False)
        self.fig_data = [fig_data]
    
# quite similar to Point class, but can store different point sets with different colors
class PointCloud:
    
    # data = [pts, "color"]
    def __init__(self, size=2):
        self.pts_data = []
        self.size = size
        
    def add_data(self, data):
        # default color is black
        if data:
            if len(data) <= 1:
                data.append('black')
            data[0] = data[0].reshape(-1, 3)
        self.pts_data.append(data) 
        
    def get_fig_data(self):
        fig_data = []
        if self.pts_data:
            for pts, color in self.pts_data:
                fig_data.append(go.Scatter3d(
                    x = pts[:, 0],
                    y = pts[:, 1],
                    z = pts[:, 2],
                    mode = 'markers',
                    marker = dict(size = self.size, color = color, opacity=1),
                    showlegend = False))
        
        return fig_data
    
# show text in 3D space
class Text(Figure):

    def __init__(self, pts, names, color='black'):
        # INPUT
        #  pts   : list of (3, ) arrays
        #  names : list of strings
        assert len(pts)==len(names)
        super().__init__()
        self.pts = pts
        self.names = names
        self.color = color 
        self.fig_data = []
        for pt, name in zip(self.pts, self.names):
            self.add_name(pt, name)

    def add_name(self, pt, name):
        fig_data = go.Scatter3d(x = [pt[0]], 
                                y = [pt[1]], 
                                z = [pt[2]],
                                text = [name],
                                mode = 'text', 
                                marker = dict(size = 2, color = self.color, opacity=1), 
                                showlegend = False)
        self.fig_data.append(fig_data)

class Camera:

    def __init__(self, pose, center, name=None, show_frame=True):
        self.pose = pose
        self.center = center
        self.focal_len = 1
        self.img_width = 4
        self.img_height = 2
        self.name = name
        self.fig_data = []
        
        # coordinate frame 
        if show_frame:
            self.fig_data.extend(Frame(self.pose, self.center).get_fig_data())
        # image plane 
        self.fig_data.extend(ImagePlane(self.pose, self.center, self.focal_len, self.img_width, self.img_height).get_fig_data())    
        # add name 
        if self.name is not None:
            self.add_name()

        # data points in represented by different coordinate frames
        self.pts_data = None # pts_data = [[pts1, color1], [pts2, color2], ...]
        self.world_pts = None 
        self.camera_pts = None 
        self.image_pts = None 

        # store the color of the points
        self.pts_color = None

        # camera pose matrix R (using the convention in Hartley's book)
        self.R = self.pose.T

        # camera calibration matrix
        self.K = np.array([[self.focal_len, 0, self.img_width/2],
                           [0, self.focal_len, self.img_height/2],
                           [0, 0, 1]])
        
        # camera projection matrix
        self.P = self.K @ self.R @ np.hstack([np.eye(3), -self.center[:, None]])

    def get_fig_data(self):
        return self.fig_data

    def plot_camera(self):
        # plot coordinate frame 
        fig = go.Figure(data=self.fig_data)
        fig.show()

    def capture(self, pts):
        self.pts_data = pts.pts_data

        self.world_pts = np.vstack([d[0] for d in self.pts_data])
        self.pts_color = np.array([d[1] for d in self.pts_data for _ in range(d[0].shape[0])])

        # transform the data to the camera coordinate frame
        # remember the formula: X_camera = pose.T @ (X_world - center)
        camera_pts = (self.world_pts - self.center) @ self.pose
        self.camera_pts = camera_pts[camera_pts[:, -1] > self.focal_len] # remove points behind the camera
        self.pts_color = self.pts_color[camera_pts[:, -1] > self.focal_len] # remove points (color) behind the camera
        
        # transform the data to the image coordinate frame
        image_pts = self.camera_pts @ self.K.T 
        self.image_pts = image_pts[:, :-1] / image_pts[:, -1][:, None]
        #self.image_pts = homo2eucl(eucl2homo(self.world_pts) @ self.P.T)
        
    def show_image(self):
        # remove the point that is outside of the image plane 
        idx = (self.image_pts[:, 1] > 0) * (self.image_pts[:, 1] < self.img_height) * (self.image_pts[:, 0] < self.img_width) * (self.image_pts[:, 0] > 0)
        if True in idx:
            image_pts = self.image_pts[idx]
            camera_pts = self.camera_pts[idx]
            dist = np.linalg.norm(camera_pts - np.array([0, 0, self.focal_len]), axis = 1)
            scaling = 5 * self.focal_len / dist
            plt.figure()
            plt.scatter(image_pts[:, 0], image_pts[:, 1], c=self.pts_color[idx], s=scaling)
        else: 
            plt.figure()
            
        plt.xlim([0, self.img_width])
        plt.ylim([0, self.img_height])
        plt.gca().set_aspect('equal')
        plt.title(self.name)
        plt.show()

    # add name for this camera
    def add_name(self):
        fig_data = go.Scatter3d(x = [self.center[0]], 
                                y = [self.center[1]], 
                                z = [self.center[2]],
                                text = [self.name],
                                mode = 'text', 
                                marker = dict(size = 2, color = 'black', opacity=1), 
                                showlegend = False)
        self.fig_data.append(fig_data)

# plot a point(2D) on a camera image plane(3D)
class CameraPoint(Point):

    def __init__(self, pts, camera, color='black', size=2):
        self.R = camera.R
        self.center = camera.center
        self.K = camera.K
        world_pts = self.img2world(pts)
        super().__init__(world_pts, color, size)
        self.data = world_pts

    # transform image points to world coordinates
    # (transformed points are on the image plane in 3D world, not 3D reconstruction)
    def img2world(self, x):
        xcam = np.linalg.inv(self.K) @ eucl2homo(x) # image frame to camera frame
        xworld = xcam @ self.R + self.center # R^T*xcam + C
        return xworld
    
class CameraLine(Line):
    
    def __init__(self, l, camera, color='black', opacity=1):
        # save the camera information
        self.R = camera.R
        self.center = camera.center 
        self.K = camera.K
        ximg1, ximg2 = self.find_two_img_pts(l)
        
        start = self.img2world(ximg1)
        end = self.img2world(ximg2)
        super().__init__(start, end, color, opacity)

    # transform image points to world coordinates
    # (transformed points are on the image plane in 3D world, not 3D reconstruction)
    def img2world(self, x):
        xcam =  eucl2homo(x) @ np.linalg.inv(self.K).T # image frame to camera frame
        xworld = xcam @ self.R + self.center # R^T*xcam + C
        return xworld
    
    # find two image points that l^T x = 0
    def find_two_img_pts(self, l):
        # let the two points be [x, 0, 1] and [0, y, 1]
        ximg1 = np.array([-l[2]/l[0], 0])
        ximg2 = np.array([0, -l[2]/l[1]])
        return ximg1, ximg2

_drange = 4 # set the plot size
_graph_center = np.array([2, 0, 1])
def Show(
    figs, 
    world=False, 
    pltrange=[[-_drange+_graph_center[0], _drange+_graph_center[0]], 
              [-_drange+_graph_center[1], _drange+_graph_center[1]],
              [-_drange+_graph_center[2], _drange+_graph_center[2]]],
    ):
    if type(figs) == dict: 
        figs = [figs[i] for i in figs] 
          
    if world == True:
        fig_data = Frame(color='gray').get_fig_data()
        for fig in figs:
            fig_data.extend(fig.get_fig_data())
        FIG = go.Figure(data=fig_data)
        FIG.update_layout(
            scene = dict(
                xaxis = dict(nticks = 4, range = pltrange[0]),
                yaxis = dict(nticks = 4, range = pltrange[1]),
                zaxis = dict(nticks = 4, range = pltrange[2]),
                # set aspectratio to 1:1:1
                aspectratio = dict(x = 1, y = 1, z = 1)
            ), 
            width=500,
            height=500,
        )
    else:
        fig_data = []
        for fig in figs:
            fig_data.extend(fig.get_fig_data())
        FIG = go.Figure(data=fig_data)
    FIG.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
    )

    FIG.show()
    return FIG

# ========================================================
# some useful functions
# ========================================================
# rotation matrix
Rx = lambda theta: np.array([
    [1, 0, 0],
    [0, np.cos(theta), -np.sin(theta)],
    [0, np.sin(theta), np.cos(theta)]])
Ry = lambda theta: np.array([
    [np.cos(theta), 0, np.sin(theta)],
    [0, 1, 0],
    [-np.sin(theta), 0, np.cos(theta)]])
Rz = lambda theta: np.array([
    [np.cos(theta), -np.sin(theta), 0], 
    [np.sin(theta), np.cos(theta), 0],
    [0, 0, 1]])

# skew symmetric matrix for cross product
skew = lambda v: np.array([
    [0, -v[2], v[1]],
    [v[2], 0, -v[0]],
    [-v[1], v[0], 0]])    

# homogeneous to euclidean, 1d or 2d data
def homo2eucl(data):
    if len(data.shape) == 1:
        return data[:-1] / data[-1]
    else:
        return data[:, :-1] / data[:, -1].reshape(-1, 1)

# euclidean to homogeneous, 1d or 2d data
def eucl2homo(data):
    if len(data.shape) == 1:
        return np.hstack([data, 1])
    else:
        return np.hstack([data, np.ones((data.shape[0], 1))])