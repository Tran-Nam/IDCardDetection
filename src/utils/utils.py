import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image 

def draw_paf(paf):
    h, w = paf.shape[:2]
    x = paf[:, :, 0]
    y = paf[:, :, 1]
    y = -y # decart coodinate to image coordinate
    #print(x)
    #print(y)
    a = np.arange(w)
    b = np.arange(h)
    u, v = np.meshgrid(a, b)
    m = np.hypot(x, y)
    fig, ax = plt.subplots(dpi=500)
    q = ax.quiver(u, v, x, y, m, pivot='mid', scale_units='xy', scale=1)
    plt.xlim(-1, w)
    plt.ylim(h, -1)
    plt.axis('off')
    #plt.show()
    return fig

def fig2im(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (w, h, 3)
    buf = np.roll(buf, 3, axis=2)
    im = Image.frombytes('RGB', (w, h), buf.tostring())
    return im

def paf2im(paf):
    im_h, im_w = paf.shape[:2]
    fig = draw_paf(paf)
    im = fig2im(fig)
    im.resize((im_w, im_h))
    im = np.array(im)
    return im


