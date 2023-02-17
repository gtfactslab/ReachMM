import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import shapely.geometry as sg
import shapely.ops as so

class Partition :
    def __init__(self, x_xh=None, x_xh_t=None, u_uh_t=None, tt=None) :
        if x_xh is None and x_xh_t is None :
            Exception('Need to define x_xh or x_xh_t')

        self.x_xh = x_xh
        self.x_xh_t = x_xh_t
        if x_xh is None and x_xh_t is not None :
            self.x_xh = x_xh_t[:,-1]
        
        self.u_uh_t = u_uh_t
        self.tt = tt if tt is not None else 0
    
    def sg_box (self, xi=0, yi=1) :
        h = len(self.x_xh) // 2
        Xl, Yl, Xu, Yu = \
            self.x_xh[xi],   self.x_xh[yi], \
            self.x_xh[xi+h], self.x_xh[yi+h]
        return sg.box(Xl,Yl,Xu,Yu)

    def rect_patch (self, xi=0, yi=1) :
        h = len(self.x_xh) // 2
        Xl, Yl, Xu, Yu = \
            self.x_xh[xi],   self.x_xh[yi], \
            self.x_xh[xi+h], self.x_xh[yi+h]
        return Rectangle((Xl,Yl),(Xu-Xl),(Yu-Yl), linewidth=0, alpha=0.2)

    def draw_rect(self, ax:plt.Axes, xi=0, yi=1) :
        h = len(self.x_xh) // 2
        Xl, Yl, Xu, Yu = \
            self.x_xh[xi],   self.x_xh[yi], \
            self.x_xh[xi+h], self.x_xh[yi+h]
        ax.add_patch(Rectangle((Xl,Yl),(Xu-Xl),(Yu-Yl), linewidth=0, alpha=0.2))
    
    def draw_cube(self, ax:plt.Axes, xi=0, yi=1, zi=2) :
        h = len(self.x_xh) // 2
        Xl, Yl, Zl, Xu, Yu, Zu = \
            self.x_xh[xi],   self.x_xh[yi],   self.x_xh[zi],\
            self.x_xh[xi+h], self.x_xh[yi+h], self.x_xh[zi+h]

    def full_partition (self) :
        parts = []
        len_x = len(self.x_xh) // 2
        part_avg = (self.x_xh[:len_x] + self.x_xh[len_x:]) / 2

        for part_i in range(2**len_x) :
            part = np.copy(self.x_xh)
            for ind in range (len_x) :
                part[ind + len_x*((part_i >> ind) % 2)] = part_avg[ind]
            parts.append(Partition(x_xh=part,x_xh_t=self.x_xh_t))
        return parts
    
    def width (self, scale=None) :
        len_x = len(self.x_xh) // 2
        width = self.x_xh[len_x:] - self.x_xh[:len_x]
        return width if scale is None else width / scale
    
    def __repr__(self) -> str:
        return self.x_xh.__repr__()

class ControlPartition (Partition):
    def __init__(self, x_xh=None, x_xh_t=None, u_uh_t=None, tt=None, integral_partitions=None):
        super().__init__(x_xh, x_xh_t, u_uh_t, tt)
        self.integral_partitions = [Partition(x_xh=)]
        if integral_partitions is None:
            
        else :
            self.integral_partitions = integral_partitions

    def add_integral_partition (self, partition):
        if self.integral_partitions is None :
            self.integral_partitions = []
        self.integral_partitions.append(partition)

    def create_integral_partitions(self, num_divisions=1):
        for div in range(num_divisions) :
            if self.integral_partitions is None :
                self.integral_partitions = self.full_partition()
            else:
                ip_i = 0
                while ip_i < len(self.integral_partitions) :
                    parts = self.integral_partitions[ip_i].full_partition()
                    del self.integral_partitions[ip_i]
                    self.integral_partitions[ip_i:ip_i] = parts
                    ip_i += len(parts)

    def full_partition (self) :
        parts = []
        len_x = len(self.x_xh) // 2
        part_avg = (self.x_xh[:len_x] + self.x_xh[len_x:]) / 2

        for part_i in range(2**len_x) :
            part = np.copy(self.x_xh)
            for ind in range (len_x) :
                part[ind + len_x*((part_i >> ind) % 2)] = part_avg[ind]
            parts.append(ControlPartition(x_xh=part,x_xh_t=self.x_xh_t))
        return parts
    
    def get_bounding_box(self) :
        if self.integral_partitions is None :
            return np.copy(self.x_xh)
        else :
            allx_xh = np.array([np.copy(ip.x_xh) for ip in self.integral_partitions])
            len_x = allx_xh.shape[1] // 2
            min_x = np.min(allx_xh[:,:len_x], axis=0)
            max_xh = np.max(allx_xh[:,len_x:], axis=0)
            return np.concatenate((min_x, max_xh))

    def get_bounding_box_t (self) :
        if self.integral_partitions is None :
            if self.x_xh_t is None :
                return np.copy(self.x_xh.reshape(-1,1))
            return np.copy(self.x_xh_t)
        else :
            allx_xh_t = np.array([np.copy(ip.x_xh_t) for ip in self.integral_partitions])
            len_x = allx_xh_t.shape[1] // 2
            min_x = np.min(allx_xh_t[:,:len_x,:], axis=0)
            max_xh = np.max(allx_xh_t[:,len_x:,:], axis=0)
            return np.concatenate((min_x, max_xh))
            
    def sg_boxes (self, xi=0, yi=1) :
        boxes = []
        if self.integral_partitions is None :
            boxes.append(self.sg_box(xi, yi))
        else :
            for ip in self.integral_partitions :
                boxes.append(ip.sg_box(xi, yi))
        return boxes

    def rect_patchs (self, xi=0, yi=1) :
        patches = []
        if self.integral_partitions is None :
            patches.append(self.rect_patch(xi, yi))
        else :
            for ip in self.integral_partitions :
                patches.append(ip.rect_patch(xi, yi))
        return patches

    def draw_rects (self, ax, xi=0, yi=1) :
        if self.integral_partitions is None :
            self.draw_rect(ax, xi, yi)
        else :
            for ip in self.integral_partitions :
                ip.draw_rect(ax, xi, yi)

    def __repr__(self) -> str:
        if self.integral_partitions is None :
            return self.x_xh.__repr__()
        else :
            return f'{self.x_xh.__repr__()} -> {self.integral_partitions}'

class ReachableSet : 
    def __init__(self, steps) :
        self.steps = steps
        self.partitions_i = [[] for s in range(steps)]

    def add_control_partition(self, partition, i=0) :
        self.partitions_i[i].append(partition)

    def create_control_partitions(self, num_divisions=1, i=0):
        for div in range(num_divisions) :
            cp_i = 0
            while cp_i < len(self.partitions_i[i]) :
                parts = self.partitions_i[i][cp_i].full_partition()
                del self.partitions_i[i][cp_i]
                self.partitions_i[i][cp_i:cp_i] = parts
                cp_i += len(parts)

    def create_partitions(self, num_control_divisions=1, num_integral_divisions=0, i=0):
        self.create_control_partitions(num_control_divisions, i)
        for cp in self.partitions_i[i] :
            cp.create_integral_partitions(num_integral_divisions)
    
    def repartition (self, num_control_divisions=1, num_integral_divisions=0, i=0) :
        bounding_x_xh = self.get_bounding_box(i)
        del self.partitions_i[i][:]
        self.add_control_partition(ControlPartition(x_xh=bounding_x_xh),i)
        self.create_partitions(num_control_divisions, num_integral_divisions, i)

    def get_bounding_box (self,i) :
        allx_xh = np.array([cp.get_bounding_box() for cp in self.partitions_i[i]])
        len_x = allx_xh.shape[1] // 2
        min_x = np.min(allx_xh[:,:len_x], axis=0)
        max_xh = np.max(allx_xh[:,len_x:], axis=0)
        return np.concatenate((min_x, max_xh))
    
    def get_bounding_box_t (self, i) :
        allx_xh_t = np.array([cp.get_bounding_box_t() for cp in self.partitions_i[i]])
        len_x = allx_xh_t.shape[1] // 2
        min_x = np.min(allx_xh_t[:,:len_x,:], axis=0)
        max_xh = np.max(allx_xh_t[:,len_x:,:], axis=0)
        return np.concatenate((min_x, max_xh))
    
    def sg_bounding_box (self, i, xi=0, yi=1) :
        x_xh = self.get_bounding_box(i)
        h = len(x_xh) // 2
        Xl, Yl, Xu, Yu = \
            x_xh[xi],   x_xh[yi], \
            x_xh[xi+h], x_xh[yi+h]
        return sg.box(Xl,Yl,Xu,Yu)
    
    def draw_sg_boxes (self, ax, xi=0, yi=1, color='tab:blue') :
        for i in range(len(self.partitions_i)) :
            boxes = []
            for cp in self.partitions_i[i] :
                boxes[-1:-1] = cp.sg_boxes(xi, yi)
            shape = so.unary_union(boxes)
            xs, ys = shape.exterior.xy    
            ax.fill(xs, ys, alpha=0.75, fc='none', ec=color)
            xsb, ysb = self.sg_bounding_box(i).exterior.xy
            ax.fill(xsb, ysb, alpha=0.5, fc='none', ec=color, linestyle='--')
    
    def draw_3d_boxes (self, ax, xi=0, yi=1, zi=2) :
        for i in range(len(self.partitions_i)) :
            x_xh = self.get_bounding_box(i)
            h = len(x_xh) // 2
            Xl, Yl, Zl, Xu, Yu, Zu = \
                x_xh[xi],   x_xh[yi],   x_xh[zi],\
                x_xh[xi+h], x_xh[yi+h], x_xh[zi+h]
            faces = [ \
                np.array([[Xl,Yl,Zl],[Xu,Yl,Zl],[Xu,Yu,Zl],[Xl,Yu,Zl],[Xl,Yl,Zl]]), \
                np.array([[Xl,Yl,Zu],[Xu,Yl,Zu],[Xu,Yu,Zu],[Xl,Yu,Zu],[Xl,Yl,Zu]]), \
                np.array([[Xl,Yl,Zl],[Xu,Yl,Zl],[Xu,Yl,Zu],[Xl,Yl,Zu],[Xl,Yl,Zl]]), \
                np.array([[Xl,Yu,Zl],[Xu,Yu,Zl],[Xu,Yu,Zu],[Xl,Yu,Zu],[Xl,Yu,Zl]]), \
                np.array([[Xl,Yl,Zl],[Xl,Yu,Zl],[Xl,Yu,Zu],[Xl,Yl,Zu],[Xl,Yl,Zl]]), \
                np.array([[Xu,Yl,Zl],[Xu,Yu,Zl],[Xu,Yu,Zu],[Xu,Yl,Zu],[Xu,Yl,Zl]]) ]
            for face in faces :
                ax.plot3D(face[:,0], face[:,1], face[:,2], color='tab:blue', alpha=0.75, lw=0.75)

    def plot_bounds_xy (self, ax, xi=0, yi=1, color='tab:blue') :
        x_xh_t = np.concatenate([self.get_bounding_box_t(i) for i in range(len(self.partitions_i))],axis=1)
        len_x = x_xh_t.shape[0] // 2
        ax.plot(x_xh_t[xi,:-1],x_xh_t[yi,:-1], color=color, linestyle='--')
        ax.plot(x_xh_t[xi+len_x,:-1],x_xh_t[yi+len_x,:-1], color=color, linestyle='--')

    def plot_bounds_t (self, ax, state, color='C0', label=None) :
        x_xh_t = np.concatenate([self.get_bounding_box_t(i) for i in range(len(self.partitions_i))],axis=1)
        tt = np.concatenate([np.atleast_1d(self.partitions_i[i][0].tt) for i in range(len(self.partitions_i))])
        len_x = x_xh_t.shape[0] // 2
        # ax.plot(tt[:-1], x_xh_t[state,:-1], alpha=0.75, linewidth=0.5, color=color)
        # ax.plot(tt[:-1], x_xh_t[state+len_x,:-1], alpha=0.75, linewidth=0.5, color=color)
        ax.fill_between(tt[:-1], x_xh_t[state,:-1], x_xh_t[state+len_x,:-1], alpha=1, color=color, label=label)

    # def draw_all_rects (self, ax, xi=0, yi=1):
    #     patches = []
    #     for i in range(len(self.partitions_i)) :
    #         for cp in self.partitions_i[i] :
    #             patches[-1:-1] = cp.rect_patchs(xi, yi)
    #     ax.add_collection(PatchCollection(patches, linewidths=0, facecolors=[0,0,1,0.2]))

    # def draw_all_rects (self, ax, xi=0, yi=1):
    #     for i in range(len(self.partitions_i)) :
    #         self.draw_rects(ax, i, xi, yi)

    # def draw_rects (self, ax, i=-1, xi=0, yi=1) :
    #     for cp in self.partitions_i[i] :
    #         cp.draw_rects(ax)

    def printall(self) :
        str = ''
        for step in range(self.steps) :
            str += f'Step {step}\n'
            str += self.partitions_i[step].__repr__() + '\n'
        return str
    def print(self) :
        print(self)
    def __repr__(self) :
        str = ''
        for cp in self.partitions_i[-1] :
            str += cp.__repr__() + '\n'
        return str

# r = ReachableSet(1)
# cp = ControlPartition(x_xh=np.array([1,3,2,4],dtype=np.float64))
# r.add_control_partition(cp)
# # r.partitions_i[0][0].create_integral_partitions(1)
# r.create_partitions(1,1)
# print(r)

