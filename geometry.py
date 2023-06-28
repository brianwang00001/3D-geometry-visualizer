import numpy as np 
import plotly.graph_objects as go

# rotation matrix
Rx = lambda theta: np.array([[1, 0, 0],
                             [0, np.cos(theta), -np.sin(theta)],
                             [0, np.sin(theta), np.cos(theta)]])
Ry = lambda theta: np.array([[np.cos(theta), 0, np.sin(theta)],
                             [0, 1, 0],
                             [-np.sin(theta), 0, np.cos(theta)]])
Rz = lambda theta: np.array([[np.cos(theta), -np.sin(theta), 0], 
                             [np.sin(theta), np.cos(theta), 0], 
                             [0, 0, 1]])

# return data of an arrow object
def arrow(start, end, color='black', name=None):
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
    result = [body, head, text]
    return result 

# return data of a frame object
def frame(pose=np.eye(3), center=np.zeros(3), color='black'):
    # original center and pose 
    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])
    o = np.zeros(3)

    # transformed center and pose  
    x_axis = pose @ x_axis
    y_axis = pose @ y_axis
    z_axis = pose @ z_axis
    o = center

    fig_data = []
    fig_data.extend(arrow(o, o + x_axis, color=color, name='x'))
    fig_data.extend(arrow(o, o + y_axis, color=color, name='y'))
    fig_data.extend(arrow(o, o + z_axis, color=color, name='z'))
    return fig_data

def image_plane(pose, center, focal_len, img_width, img_height, color='black'):
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
    
    data = [go.Mesh3d(x=corners[:, 0], y=corners[:, 1], z=corners[:, 2], 
                      i=[0, 0],  # vertices of first triangle
                      j=[1, 2],  # vertices of second triangle
                      k=[2, 3],  # vertices of third triangle
                      opacity=.3, color=color)]
    return data

drange = 10 # set the plot size
def plot_all(figs, world=False, pltrange=[[-1, drange], [-1, drange], [-1, drange]]):
    if world == True:
        fig_data = frame(color='gray')
        for fig in figs:
            fig_data.extend(fig.get_fig_data())
        FIG = go.Figure(data=fig_data)
        FIG.update_layout(
            scene = dict(
                xaxis = dict(nticks = 4, range = pltrange[0]),
                yaxis = dict(nticks = 4, range = pltrange[1]),
                zaxis = dict(nticks = 4, range = pltrange[2]),
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=1)
            )
        )
    else:
        fig_data = []
        for fig in figs:
            fig_data.extend(fig.get_fig_data())
        FIG = go.Figure(data=fig_data)

    FIG.show()

class Camera():

    def __init__(self, pose, center):
        self.pose = pose
        self.center = center
        self.focal_len = 1
        self.img_width = 4
        self.img_height = 2

        # coordinate frame 
        self.fig_data = frame(self.pose, self.center)
        # image plane 
        self.fig_data.extend(image_plane(self.pose, self.center, self.focal_len, self.img_width, self.img_height))

        # data points in represented by different coordinate frames
        self.world_data = None
        self.camera_data = None 
        self.image_data = None 

    def get_fig_data(self):
        return self.fig_data

    def plot_camera(self):
        # plot coordinate frame 
        fig = go.Figure(data=self.fig_data)
        fig.show()

    def capture(self, data):
        self.world_data = data

# save the data in the camera coordinate frame
# and return the data as a list of go.Scatter3d objects        
class PointCloud():
    
    def __init__(self, data=None):
        self.data = None

    def add_data(self, data):
        if self.data is None:
            self.data = data
        else:
            self.data = np.vstack((self.data, data))

    def get_fig_data(self):
        if self.data is None:
            fig_data = []
        else:
            fig_data = [go.Scatter3d(x = self.data[:, 0], 
                                     y = self.data[:, 1], 
                                     z = self.data[:, 2],
                                     mode = 'markers', 
                                     marker = dict(size = 2, color = 'black', opacity=1), 
                                     showlegend = False)]
        return fig_data