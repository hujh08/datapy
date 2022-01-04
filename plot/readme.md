layout of axes

# Motivation

This module aims to provide flexible method to create figure and put axes in it.

For figure creating, main parameters to determine are figure size, `(width, height)` in **inch**es, and resolution `dpi`, dots (**pixel**) per inch. These two properties together give the number of pixels in a figure. Another unit of size is **point**. In typography, a point is 1/72 inches. It uses same setup in `matplotlib` (see [transforms tutorial](https://matplotlib.org/stable/tutorials/advanced/transforms_tutorial.html) or the code `text:_AnnotationBase._get_xy_transform`, coping with `xycoords = 'figure points'`). Many elements in `matplotlib` is specified for size in points, e.g. line, text and marker, et. al. (***NOTE***: marker size in `scatter` has unit `points**2`, which corresponds to marker area, instead of 1d size).

Layout of axes in figure means essentially to decide its position with respect to figure. During this process, many constraints should be considered, like axes size in order to hold neccessary elements, and separation between axes to avoid to cover axis label.

In `matplotlib`, layout via `gridspec` needs to preceding calculation, considering these constraints, for some parameters, e.g. `hspace`, `wspace`. Although this calculation may be simple, in actual case, computing procedure should be carefully designly.

To get around these inconveniences, this module provides a flexible way to cope with these constraints generally. It is raised from thought that position of axes is exactly set by 4 variables `(x0, y0, x1, y1)`, the coordinates of bottom-left and top-right corner, and (almost) all constraints are linear for these variables. A typical constraint is ratio between distance. For example, width of two axes, written as `a0` and `a1`, with coordinates `(a0x0, a0y0, a0x1, a0y1)` and `(a1x0, a1y0, a1x1, a1y1)` respectively. Then widths of them are `a0x1-a0x0` and `a1x1-a1x0`. If we require a ratio `1:k`, constraint is `a0x1-a0x0 = (a1x1-a1x0)/k`, which is linear. Another frequently-used constraint is to align two axes in some axis, e.g. `x-axis`. Using previous example, aligning left of axes `a0` and `a1` is represented as `a0x0 = a1x0`. In this module, coordinate variables and linear constraints are traced and maintained dynamically.
