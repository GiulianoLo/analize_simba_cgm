"""Simple scatter-plot animations with fading trails."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def create_animation(x, y, outfile='animation.gif', fps=30, interval=3000):
    """Create an animated scatter plot with colour-changing points.

    Points from earlier frames become more transparent while the
    colour shifts through viridis.

    Parameters
    ----------
    x, y : 2-D np.ndarray
        Coordinate arrays of shape ``(n_frames, n_points)``.
    outfile : str
        Output GIF path.
    fps : int
        Frames per second.
    interval : int
        Delay between frames in ms.
    """
    fig, ax = plt.subplots()
    scat = ax.scatter([], [], s=10)

    margin = 0.1 * (np.nanmax(x) - np.nanmin(x))
    ax.set_xlim(np.nanmin(x) - margin, np.nanmax(x) + margin)
    ax.set_ylim(np.nanmin(y) - margin, np.nanmax(y) + margin)

    all_x, all_y = [], []

    def update(frame):
        all_x.append(x[frame, :])
        all_y.append(y[frame, :])
        alpha_values = frame / max(x.shape[0] - 1, 1)

        combined_x = np.concatenate(all_x)
        combined_y = np.concatenate(all_y)
        scat.set_offsets(np.column_stack((combined_x, combined_y)))
        scat.set_alpha(alpha_values)
        color = plt.cm.viridis(frame / max(x.shape[0] - 1, 1))
        scat.set_color(color)
        return (scat,)

    ani = FuncAnimation(fig, update, frames=x.shape[0],
                        interval=interval, repeat=True)
    ani.save(outfile, writer=PillowWriter(fps=fps))
    plt.show()
