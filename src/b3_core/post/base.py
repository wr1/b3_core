#!/usr/bin/env python3

import pyvista as pv
import numpy as np
import argparse
import json


def postprocess(results, datfiles):
    output = {}
    for i in zip(results, datfiles):
        res = i[0]
        lc = i[1].split("_")[-1].split(".")[0]
        res_cell = res.point_data_to_cell_data()
        for j in np.unique(res_cell.cell_data["material"]):
            output[f"stress_{lc}_{j}"] = float(
                (
                    (res_cell.cell_data["material"] == j)
                    * res_cell.cell_data["mises_stress"]
                ).max()
            )
            output[f"strain_{lc}_{j}"] = float(
                (
                    (res_cell.cell_data["material"] == j)
                    * res_cell.cell_data["mises_strain"]
                ).max()
            )
        datfile = i[1]
        dat = np.array(
            [j.split()[1:] for j in open(datfile, "r").readlines()[-3:]]
        ).astype(float)
        xmin, xmax, ymin, ymax, zmin, zmax = res.bounds
        dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmax
        if i[1].find("xy") != -1:
            stress = (
                (res.points[:, 1] == ymin) * res.point_data["force"][:, 0]
            ).sum() / (dx * dz)
            strain = -dat[0, 1] / dx
            shear_modulus = stress / strain
            output["Gxy"] = float(shear_modulus)
        elif i[1].find("xz") != -1:
            stress = (
                (res.points[:, 2] == zmin) * res.point_data["force"][:, 0]
            ).sum() / (dx * dy)
            strain = -dat[2, 0] / dz
            shear_modulus = stress / strain
            output["Gxz"] = float(shear_modulus)
        elif i[1].find("yz") != -1:
            stress = (
                (res.points[:, 2] == zmin) * res.point_data["force"][:, 1]
            ).sum() / (dx * dy)
            strain = -dat[2, 1] / dz
            shear_modulus = stress / strain
            output["Gyz"] = float(shear_modulus)
        elif i[1].find("xx") != -1:
            xstr = (
                (res.points[:, 0] == xmin) * res.point_data["force"][:, 0]
            ).sum() / (dy * dz)
            strain = dat[0, 0] / dx
            strainy = -dat[1, 1] / dy
            strainz = -dat[2, 2] / dz
            e_modulus = np.fabs(xstr) / strain
            nuxy = strainy / strain
            nuxz = strainz / strain
            output["Exx"] = float(e_modulus)
            output["nuxy"] = float(nuxy)
            output["nuxz"] = float(nuxz)
        elif i[1].find("yy") != -1:
            stress = (
                (res.points[:, 1] == ymin) * res.point_data["force"][:, 1]
            ).sum() / (dx * dz)
            strain = dat[1, 1] / dy
            strainx = -dat[0, 0] / dx
            strainz = -dat[2, 2] / dz
            nuyx = strainx / strain
            nuyz = strainz / strain
            e_modulus = np.fabs(stress) / strain
            output["Eyy"] = float(e_modulus)
            output["nuyx"] = float(nuyx)
            output["nuyz"] = float(nuyz)
        elif i[1].find("zz") != -1:
            stress = (
                (res.points[:, 2] == zmin) * res.point_data["force"][:, 2]
            ).sum() / (dx * dy)
            strain = dat[2, 2] / dz
            strainx = -dat[0, 0] / dx
            strainy = -dat[1, 1] / dy
            nuzx = strainx / strain
            nuzy = strainy / strain
            e_modulus = np.fabs(stress) / strain
            output["Ezz"] = float(e_modulus)
            output["nuzx"] = float(nuzx)
            output["nuzy"] = float(nuzy)
    print(output)
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results", nargs="+")
    p.add_argument("--output", default="__eprop.json", help="output json file")
    args = p.parse_args()
    out = postprocess(
        [pv.read(i) for i in args.results],
        [i.replace(".vtu", ".dat") for i in args.results],
    )
    json.dump(out, open(args.output, "w"), indent=4)


if __name__ == "__main__":
    main()
