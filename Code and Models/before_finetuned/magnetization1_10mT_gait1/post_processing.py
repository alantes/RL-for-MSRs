import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib import cm
from tqdm import tqdm

from typing import Dict, Sequence


def plot_video_with_sphere(
    rods_history: Sequence[Dict],
    video_name="video.mp4",
    fps=60,
    step=1,
    vis2D=True,
    **kwargs,
):
    plt.rcParams.update({"font.size": 22})

    # 2d case <always 2d case for now>
    import matplotlib.animation as animation
    from matplotlib.patches import Circle
    from mpl_toolkits.mplot3d import proj3d, Axes3D

    # simulation time
    sim_time = np.array(rods_history[0]["time"])


    # Rod
    n_visualized_rods = len(rods_history)  # should be one for now
    # Rod info
    rod_history_unpacker = lambda rod_idx, t_idx: (
        rods_history[rod_idx]["position"][t_idx],
        rods_history[rod_idx]["radius"][t_idx],
    )
    # Rod center of mass
    com_history_unpacker = lambda rod_idx, t_idx: rods_history[rod_idx]["com"][time_idx]

    # video pre-processing
    print("plot scene visualization video")
    FFMpegWriter = animation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    dpi = kwargs.get("dpi", 100)

    xlim = kwargs.get("x_limits", (-1.0, 100.0))
    ylim = kwargs.get("y_limits", (-1.0, 100.0))
    zlim = kwargs.get("z_limits", (-1.0, 100.0))

    difference = lambda x: x[1] - x[0]
    max_axis_length = max(difference(xlim), difference(ylim))
    # The scaling factor from physical space to matplotlib space
    scaling_factor = (2 * 0.1) / max_axis_length  # Octopus head dimension
    scaling_factor *= 2.6e3  # Along one-axis

    fig = plt.figure(2, figsize=(10, 8), frameon=True, dpi=dpi)
    # ax = fig.add_subplot(111)
    ax = plt.axes(projection="3d")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)

    time_idx = 0
    rod_lines = [None for _ in range(n_visualized_rods)]
    rod_com_lines = [None for _ in range(n_visualized_rods)]
    rod_scatters = [None for _ in range(n_visualized_rods)]

    for rod_idx in range(n_visualized_rods):
        inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
        # TAG: need to resize the inst_radius to make it adapt to ax.scatter
        inst_position = 0.5 * (inst_position[..., 1:] + inst_position[..., :-1])
        # rod_lines[rod_idx] = ax.plot(inst_position[0], inst_position[1],inst_position[2], "r", lw=0.5)[0]
        inst_com = com_history_unpacker(rod_idx, time_idx)
        # rod_com_lines[rod_idx] = ax.plot(inst_com[0], inst_com[1],inst_com[2], "k--", lw=2.0)[0]

        rod_scatters[rod_idx] = ax.scatter(
            inst_position[2],
            inst_position[0],
            inst_position[1],
            s=np.pi * (inst_radius) ** 2,
        )


    # ax.set_aspect("equal")
    video_name = "2D_" + video_name

    with writer.saving(fig, video_name, dpi):
        with plt.style.context("seaborn-whitegrid"):
            for time_idx in tqdm(range(0, sim_time.shape[0], int(step))):

                for rod_idx in range(n_visualized_rods):
                    inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
                    inst_position = 0.5 * (
                        inst_position[..., 1:] + inst_position[..., :-1]
                    )

                    # rod_lines[rod_idx].set_xdata(inst_position[0])
                    # rod_lines[rod_idx].set_ydata(inst_position[1])
                    # rod_lines[rod_idx].set_zdata(inst_position[2])

                    com = com_history_unpacker(rod_idx, time_idx)
                    # rod_com_lines[rod_idx].set_xdata(com[0])
                    # rod_com_lines[rod_idx].set_ydata(com[1])
                    # rod_com_lines[rod_idx].set_zdata(com[2])

                    # rod_scatters[rod_idx].set_offsets(inst_position[:3].T)
                    rod_scatters[rod_idx]._offsets3d = (
                        inst_position[2],
                        inst_position[0],
                        inst_position[1],
                    )

                    rod_scatters[rod_idx].set_sizes(
                        np.pi * (scaling_factor * inst_radius) ** 2 * 0.1
                    )

                writer.grab_frame()

    # Be a good boy and close figures
    # https://stackoverflow.com/a/37451036
    # plt.close(fig) alone does not suffice
    # See https://github.com/matplotlib/matplotlib/issues/8560/
    plt.close(plt.gcf())


def plot_video_with_sphere_2D(
    rods_history: Sequence[Dict],
    video_name="video_2D.mp4",
    fps=60,
    step=1,
    vis2D=True,
    **kwargs,
):
    plt.rcParams.update({"font.size": 22})

    # 2d case <always 2d case for now>
    import matplotlib.animation as animation
    from matplotlib.patches import Circle

    # simulation time
    sim_time = np.array(rods_history[0]["time"])

    # Rod
    n_visualized_rods = len(rods_history)  # should be one for now
    # Rod info
    rod_history_unpacker = lambda rod_idx, t_idx: (
        rods_history[rod_idx]["position"][t_idx],
        rods_history[rod_idx]["radius"][t_idx],
    )
    # Rod center of mass
    com_history_unpacker = lambda rod_idx, t_idx: rods_history[rod_idx]["com"][time_idx]

    # video pre-processing
    print("plot scene visualization video")
    FFMpegWriter = animation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    dpi = kwargs.get("dpi", 100)

    xlim = kwargs.get("x_limits", (-1.0, 100.0))
    ylim = kwargs.get("y_limits", (-1.0, 100.0))
    zlim = kwargs.get("z_limits", (-1.0, 100.0))

    difference = lambda x: x[1] - x[0]
    max_axis_length = max(difference(xlim), difference(ylim))
    # The scaling factor from physical space to matplotlib space
    scaling_factor = (2 * 0.1) / max_axis_length  # Octopus head dimension
    scaling_factor *= 2.6e3  # Along one-axis

    fig = plt.figure(2, figsize=(10, 8), frameon=True, dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    time_idx = 0
    rod_lines = [None for _ in range(n_visualized_rods)]
    rod_com_lines = [None for _ in range(n_visualized_rods)]
    rod_scatters = [None for _ in range(n_visualized_rods)]

    for rod_idx in range(n_visualized_rods):
        inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
        # TAG: need to resize the inst_radius to make it adapt to ax.scatter
        inst_position = 0.5 * (inst_position[..., 1:] + inst_position[..., :-1])
        rod_lines[rod_idx] = ax.plot(inst_position[0], inst_position[1], "r", lw=0.5)[0]
        inst_com = com_history_unpacker(rod_idx, time_idx)
        rod_com_lines[rod_idx] = ax.plot(inst_com[0], inst_com[1], "k--", lw=2.0)[0]

        rod_scatters[rod_idx] = ax.scatter(
            inst_position[0],
            inst_position[1],
            s=np.pi * (inst_radius) ** 2,
        )

    ax.set_aspect("equal")
    video_name = "2D_" + video_name

    with writer.saving(fig, video_name, dpi):
        with plt.style.context("seaborn-whitegrid"):
            for time_idx in tqdm(range(0, sim_time.shape[0], int(step))):

                for rod_idx in range(n_visualized_rods):
                    inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
                    inst_position = 0.5 * (
                        inst_position[..., 1:] + inst_position[..., :-1]
                    )

                    rod_lines[rod_idx].set_xdata(inst_position[0])
                    rod_lines[rod_idx].set_ydata(inst_position[1])

                    com = com_history_unpacker(rod_idx, time_idx)
                    rod_com_lines[rod_idx].set_xdata(com[0])
                    rod_com_lines[rod_idx].set_ydata(com[1])

                    rod_scatters[rod_idx].set_offsets(inst_position[:2].T)
                    rod_scatters[rod_idx].set_sizes(
                        np.pi * (scaling_factor * inst_radius) ** 2
                    )

                writer.grab_frame()
