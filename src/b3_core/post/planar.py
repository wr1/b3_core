#!/usr/bin/env python3

import pyvista as pv
import numpy as np
import argparse
import json


def postprocess_planar(results):
    output = {}
    for i in results:
        res = pv.read(i)
        # Find dynamic keys for disp and force
        disp_key = next(
            (k for k in res.point_data.keys() if k.startswith("DISP")), None
        )
        forc_key = next(
            (k for k in res.point_data.keys() if k.startswith("FORC")), None
        )
        if disp_key is None or forc_key is None:
            raise KeyError("Required data arrays (DISP or FORC) not found in VTU")
        dispmax = res.point_data[disp_key].max()
        xmin, xmax, ymin, ymax, zmin, zmax = res.bounds
        dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
        if i.find("xy") != -1:
            stress = (
                (res.points[:, 1] == ymin) * res.point_data[forc_key][:, 0]
            ).sum() / (dx * dz)
            strain = -dispmax / dx
            shear_modulus = stress / strain
            output["Gxy"] = float(shear_modulus)
        elif i.find("xz") != -1:
            stress = (
                (res.points[:, 2] == zmin) * res.point_data[forc_key][:, 0]
            ).sum() / (dx * dy)
            strain = -dispmax / dz
            shear_modulus = stress / strain
            output["Gxz"] = float(shear_modulus)
        elif i.find("yz") != -1:
            stress = (
                (res.points[:, 2] == zmin) * res.point_data[forc_key][:, 1]
            ).sum() / (dx * dy)
            strain = -dispmax / dz
            shear_modulus = stress / strain
            output["Gyz"] = float(shear_modulus)
        elif i.find("xx") != -1:
            xstr = (
                (res.points[:, 0] == xmin) * res.point_data[forc_key][:, 0]
            ).sum() / (dy * dz)
            strain = res.point_data[disp_key][:, 0].max() / dx
            strainy = (
                res.point_data[disp_key][:, 1].max()
                - res.point_data[disp_key][:, 1].min()
            ) / dy
            strainz = (
                res.point_data[disp_key][:, 2].max()
                - res.point_data[disp_key][:, 2].min()
            ) / dz
            e_modulus = np.fabs(xstr) / strain
            nuxy = strainy / strain
            nuxz = strainz / strain
            output["Exx"] = float(e_modulus)
            output["nuxy"] = float(nuxy)
            output["nuxz"] = float(nuxz)
        elif i.find("yy") != -1:
            stress = (
                (res.points[:, 1] == ymin) * res.point_data[forc_key][:, 1]
            ).sum() / (dx * dz)
            strain = res.point_data[disp_key][:, 1].max() / dy
            strainx = (
                res.point_data[disp_key][:, 0].max()
                - res.point_data[disp_key][:, 0].min()
            ) / dx
            strainz = (
                res.point_data[disp_key][:, 2].max()
                - res.point_data[disp_key][:, 2].min()
            ) / dz
            nuyx = strainx / strain
            nuyz = strainz / strain
            e_modulus = np.fabs(stress) / strain
            output["Eyy"] = float(e_modulus)
            output["nuyx"] = float(nuyx)
            output["nuyz"] = float(nuyz)
        elif i.find("zz") != -1:
            stress = (
                (res.points[:, 2] == zmin) * res.point_data[forc_key][:, 2]
            ).sum() / (dx * dy)
            strain = res.point_data[disp_key][:, 2].max() / dz
            strainx = (
                res.point_data[disp_key][:, 0].max()
                - res.point_data[disp_key][:, 0].min()
            ) / dx
            strainy = (
                res.point_data[disp_key][:, 1].max()
                - res.point_data[disp_key][:, 1].min()
            ) / dy
            nuzx = strainx / strain
            nuzy = strainy / strain
            e_modulus = np.fabs(stress) / strain
            output["Ezz"] = float(e_modulus)
            output["nuzx"] = float(nuzx)
            output["nuzy"] = float(nuzy)
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("results", nargs="+")
    p.add_argument("--output", default="__eprop_planar.json", help="output json file")
    args = p.parse_args()
    output = postprocess_planar(args.results)
    json.dump(output, open(args.output, "w"), indent=4)
