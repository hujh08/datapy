layout of axes

# Motivation
Many constraints should be considered when putting axes in figure, like ax size, or separation between axes to avoid to cover axis label.

To handle these constraint generally, layout of axes is viewed as colletion of rectangles, each for an ax. A rectangle could be specified by coordinates of its bottom-left and top-right corner, (x0, y0) and (x1, y1) respectively. These four numbers (x0, x1, y0, y1) could be splitted into two pairs, (x0, x1) and (y0, y1), and each could be seen as points in a 1d line, along x direction or y direction. Meanwhile, constraint to ax size or separation could be fully transferred to constraint at distance between these pairs, without any loss.

Starting from this, all these 1d points of axes, including two independent sets (xs, ...) and (ys, ...), are managed in a directed graph-like method. Corresponding graph edge is the distance between point pair. And all distances are also managed as graph. Constraint is then implemented as the graph edge.

For distances, besides one between point pair, some absolute distances are also considered, like unit `pt`, `inch`. And constraints on distance are all considered relatively to another distance, pair one or absolute unit. Distances in x and y direction are considered together, connected by absolute unit or aspect of ax, e.g. `equal`.

# Point

# Distance