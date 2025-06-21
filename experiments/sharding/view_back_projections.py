import numpy as np
import mbirjax as mj

back_projection_normal = np.load("output/control_back_projection.npy")

title = 'Control Back Projection'
mj.slice_viewer(back_projection_normal, slice_axis=0, title=title)