#!/usr/bin/env python3

import numpy as np
import pyvista as pv


def geom_analysis(mesh):
    ma = mesh.compute_cell_sizes()
    res = ma.cell_data["resin"]
    volumes = np.abs(ma.cell_data["Volume"])
    total_volume = volumes.sum()
    resin_volume = volumes[res == 1].sum()
    core_volume = volumes[res == 0].sum()
    resin_vf = resin_volume / total_volume
    core_domain = ma.extract_cells((res == 0).nonzero()[0])
    core_surface = core_domain.extract_surface().compute_cell_sizes()
    csc = core_surface.cell_centers().points
    xmin, xmax, ymin, ymax, zmin, zmax = ma.bounds
    dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
    core_xyedges = (
        (csc[:, 0] == xmin)
        | (csc[:, 0] == xmax)
        | (csc[:, 1] == ymin)
        | (csc[:, 1] == ymax)
    )
    ca = core_surface.cell_data["Area"]
    core_area_notxy = (ca * (core_xyedges == 0)).sum()
    corearea_nogrooves = dx * dy * 2
    area_increase = core_area_notxy / corearea_nogrooves
    return {"area_increase": float(area_increase), "resin_vf": float(resin_vf)}


if __name__ == "__main__":
    m = pv.read("m30_yc1.vts")
    geom_analysis(m)
