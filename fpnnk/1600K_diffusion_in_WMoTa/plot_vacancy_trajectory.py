import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def parse_dump(filename):
    positions = []
    box = None
    with open(filename) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == "ITEM: TIMESTEP":
            i += 2  # skip timestep value
        elif line == "ITEM: NUMBER OF ATOMS":
            i += 2  # skip count
        elif line.startswith("ITEM: BOX BOUNDS"):
            xlo, xhi = map(float, lines[i+1].split())
            ylo, yhi = map(float, lines[i+2].split())
            zlo, zhi = map(float, lines[i+3].split())
            if box is None:
                box = [xlo, xhi, ylo, yhi, zlo, zhi]
            i += 4
        elif line.startswith("ITEM: ATOMS"):
            i += 1
            # read one atom per config
            parts = lines[i].split()
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            positions.append([x, y, z])
            i += 1
        else:
            i += 1

    return np.array(positions), box


def adjusted_box(positions, box, pad=20.0):
    """Cubic box with side = max trajectory span + 2*pad, centered on trajectory."""
    spans = positions.max(axis=0) - positions.min(axis=0)
    side = spans.max() + 2 * pad
    center = (positions.min(axis=0) + positions.max(axis=0)) / 2
    lo = center - side / 2
    hi = center + side / 2
    return [lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]]


def draw_box(ax, box):
    xlo, xhi, ylo, yhi, zlo, zhi = box
    corners = np.array([
        [xlo, ylo, zlo], [xhi, ylo, zlo],
        [xhi, yhi, zlo], [xlo, yhi, zlo],
        [xlo, ylo, zhi], [xhi, ylo, zhi],
        [xhi, yhi, zhi], [xlo, yhi, zhi],
    ])
    edges = [
        (0,1),(1,2),(2,3),(3,0),  # bottom face
        (4,5),(5,6),(6,7),(7,4),  # top face
        (0,4),(1,5),(2,6),(3,7),  # verticals
    ]
    for a, b in edges:
        ax.plot(*zip(corners[a], corners[b]), color='black', lw=2.0, ls='-')


def main():
    positions, box = parse_dump("res_dir/vacancy_configs_unwrap.dump")
    box = adjusted_box(positions, box)

    n = len(positions)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    # color trajectory by time
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, n - 1))

    # draw line segments colored by timestep
    segments = [positions[i:i+2] for i in range(n - 1)]
    lc = Line3DCollection(segments, colors=colors, linewidths=1.5, zorder=2)
    ax.add_collection3d(lc)

    # scatter points
    sc = ax.scatter(
        positions[:, 0], positions[:, 1], positions[:, 2],
        c=np.arange(n), cmap='viridis', s=5, zorder=3, depthshade=True
    )

    # arrows at midpoint of each segment, colored by timestep
    for i in range(n - 1):
        mid = positions[i] + 0.5 * (positions[i+1] - positions[i])
        d = positions[i+1] - positions[i]
        color = cmap(i / (n - 1))
        ax.quiver(
            mid[0], mid[1], mid[2],
            d[0], d[1], d[2],
            length=0.4, normalize=True,
            color=color, arrow_length_ratio=0.6,
            linewidth=0, zorder=4
        )

    draw_box(ax, box)

    cbar = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.6)
    cbar.set_label('Timestep')

    # remove all background panes, grid, ticks, labels, and axes
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor('none')
    ax.grid(False)
    ax.set_axis_off()

    # equal axis ranges so box looks cubic
    xlo, xhi, ylo, yhi, zlo, zhi = box
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_zlim(zlo, zhi)
    ax.set_box_aspect([1, 1, 1])

    # white background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    plt.tight_layout()
    plt.savefig('vacancy_trajectory.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    print(f"Plot Completed! Generating vacancy trajectory figure saved in path: results/vacancy_trajectory.png with {n} frames")


if __name__ == "__main__":
    main()

