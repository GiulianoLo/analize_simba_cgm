import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

'''Code must be implemented in order to keep trace of teach frame 
'''


def create_animation(x, y):
    """
    Creates an animated scatter plot where points change color each frame,
    and old points become more transparent.

    Parameters:
    - x: 2D NumPy array for x-coordinates
    - y: 2D NumPy array for y-coordinates
    """
    # Create figure and axis
    fig, ax = plt.subplots()
    scat = ax.scatter([], [], s=10)  # Initialize empty scatter plot

    # Set axis limits based on the data
    min_x, max_x = np.nanmin(x), np.nanmax(x)
    min_y, max_y = np.nanmin(y), np.nanmax(y)
    margin = 0.1 * (max_x - min_x)  # Add some margin to the plot
    ax.set_xlim(min_x - margin, max_x + margin)
    ax.set_ylim(min_y - margin, max_y + margin)

    # Initialize storage for points and their properties
    all_x = []
    all_y = []
    all_alpha = []

    # Function to update the animation
    def update(frame):
        # Append current positions to the storage lists
        current_x = x[frame, :]
        current_y = y[frame, :]

        all_x.append(current_x)
        all_y.append(current_y)

        # Calculate alpha values for all points
        alpha_values = frame / (x.shape[0] - 1)

        # Concatenate all stored points
        combined_x = np.concatenate(all_x)
        combined_y = np.concatenate(all_y)

        # Update scatter plot with new points
        scat.set_offsets(np.column_stack((combined_x, combined_y)))
        scat.set_alpha(alpha_values)
        color = plt.cm.viridis(frame / (x.shape[0] - 1))  # Change color frame by frame
        scat.set_color(color)

        return scat,

    # Create animation
    ani = FuncAnimation(fig, update, frames=x.shape[0], interval=3000, repeat=True)

    # Save the animation as a GIF or MP4 file
    ani.save("animation.gif", writer=PillowWriter(fps=30))  # For GIF
    # ani.save("animation.mp4", writer='ffmpeg')  # Uncomment this line to save as MP4

    # Display the plot
    plt.show()