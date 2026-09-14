#!/usr/bin/env python3

import argparse
import json

import numpy as np
import pyvista as pv


def postprocess(results, datfiles, thickness):
    output = {}
    for i in zip(results, datfiles, strict=False):
        res = i[0]
        res_cell = res.point_data_to_cell_data(pass_point_data=True)
        if res_cell.bounds[5] > thickness * 1e-3 + 1e-9:
            centers = res_cell.cell_centers().points
            keep = np.where(centers[:, 2] < thickness * 1e-3)[0]
            res_cell = res_cell.extract_cells(keep)
        # Find dynamic keys for DISP and FORC
        disp_key = next(
            (k for k in res_cell.point_data.keys() if k.startswith("DISP")), None
        )
        forc_key = next(
            (k for k in res_cell.point_data.keys() if k.startswith("FORC")), None
        )
        if disp_key is None or forc_key is None:
            raise KeyError("Required data arrays (DISP or FORC) not found in VTU")
        # Temporarily commented out due to missing 'material' in VTU
        # for j in np.unique(res_cell.cell_data["material"]):
        #     output[f"stress_{lc}_{j}"] = float(
        #         (
        #             (res_cell.cell_data["material"] == j)
        #             * res_cell.cell_data["mises_stress"]
        #         ).max()
        #     )
        #     output[f"strain_{lc}_{j}"] = float(
        #         (
        #             (res_cell.cell_data["material"] == j)
        #             * res_cell.cell_data["mises_strain"]
        #         ).max()
        #     )
        xmin, xmax, ymin, ymax, zmin, zmax = res_cell.bounds
        dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
        tol = 1e-7
        xminp = np.where(res_cell.points[:, 0] < xmin + tol)[0]
        xmaxp = np.where(res_cell.points[:, 0] > xmax - tol)[0]
        yminp = np.where(res_cell.points[:, 1] < ymin + tol)[0]
        ymaxp = np.where(res_cell.points[:, 1] > ymax - tol)[0]
        zminp = np.where(res_cell.points[:, 2] < zmin + tol)[0]
        zmaxp = np.where(res_cell.points[:, 2] > zmax - tol)[0]
        xmindisp = res_cell.point_data[disp_key][xminp]
        xmaxdisp = res_cell.point_data[disp_key][xmaxp]
        ymindisp = res_cell.point_data[disp_key][yminp]
        ymaxdisp = res_cell.point_data[disp_key][ymaxp]
        zmindisp = res_cell.point_data[disp_key][zminp]
        zmaxdisp = res_cell.point_data[disp_key][zmaxp]
        ddx = (xmaxdisp - xmindisp).mean(axis=0)
        ddy = (ymaxdisp - ymindisp).mean(axis=0)
        ddz = (zmaxdisp - zmindisp).mean(axis=0)
        if i[1].find("xy") != -1:
            stress = res_cell.point_data[forc_key][yminp, 0].sum() / (dx * dz)
            strain = -ddx[1] / dx
            shear_modulus = stress / strain
            output["Gxy"] = float(shear_modulus)
        elif i[1].find("xz") != -1:
            stress = res_cell.point_data[forc_key][zminp, 0].sum() / (dx * dy)
            strain = -ddz[0] / dz
            shear_modulus = stress / strain
            output["Gxz"] = float(shear_modulus)
        elif i[1].find("yz") != -1:
            stress = res_cell.point_data[forc_key][zminp, 1].sum() / (dx * dy)
            strain = -ddz[1] / dz
            shear_modulus = stress / strain
            output["Gyz"] = float(shear_modulus)
        elif i[1].find("xx") != -1:
            xstr = res_cell.point_data[forc_key][xminp, 0].sum() / (dy * dz)
            strain = ddx[0] / dx
            strainy = -ddy[1] / dy
            strainz = -ddz[2] / dz
            e_modulus = np.fabs(xstr) / strain
            nuxy = strainy / strain
            nuxz = strainz / strain
            output["Exx"] = float(e_modulus)
            output["nuxy"] = float(nuxy)
            output["nuxz"] = float(nuxz)
        elif i[1].find("yy") != -1:
            stress = res_cell.point_data[forc_key][yminp, 1].sum() / (dx * dz)
            strain = ddy[1] / dy
            strainx = -ddx[0] / dx
            strainz = -ddz[2] / dz
            nuyx = strainx / strain
            nuyz = strainz / strain
            e_modulus = np.fabs(stress) / strain
            output["Eyy"] = float(e_modulus)
            output["nuyx"] = float(nuyx)
            output["nuyz"] = float(nuyz)
        elif i[1].find("zz") != -1:
            stress = res_cell.point_data[forc_key][zminp, 2].sum() / (dx * dy)
            strain = ddz[2] / dz
            strainx = -ddx[0] / dx
            strainy = -ddy[1] / dy
            nuzx = strainx / strain
            nuzy = strainy / strain
            e_modulus = np.fabs(stress) / strain
            output["Ezz"] = float(e_modulus)
            output["nuzx"] = float(nuzx)
            output["nuzy"] = float(nuzy)
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results", nargs="+")
    p.add_argument("--output", default="__eprop.json", help="output json file")
    args = p.parse_args()
    out = postprocess(
        [pv.read(i) for i in args.results],
        [i.replace(".vtu", ".dat") for i in args.results],
        thickness=30.0,
    )
    json.dump(out, open(args.output, "w"), indent=4)


if __name__ == "__main__":
    main()
