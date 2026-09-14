#!/usr/bin/env python3

import argparse

import numpy as np
import pyvista as pv


def nset(nds, name):
    out = f"*nset,nset={name}\n"
    for i in np.array_split(nds + 1, np.ceil(len(nds) / 16.0)):
        out += ",".join(i.astype(str).tolist()) + "\n"
    return out


def _sort_face(face, points, free_axes):
    coords = points[face][:, free_axes]
    order = np.lexsort([coords[:, i] for i in reversed(range(coords.shape[1]))])
    return face[order]


def periodic_bcs(msh, added_nodes):
    bnds = msh.GetBounds()
    pts = msh.points
    xmin = _sort_face(np.where(pts[:, 0] == bnds[0])[0], pts, [1, 2])
    xmax = _sort_face(np.where(pts[:, 0] == bnds[1])[0], pts, [1, 2])
    ymin = _sort_face(np.where(pts[:, 1] == bnds[2])[0], pts, [0, 2])
    ymax = _sort_face(np.where(pts[:, 1] == bnds[3])[0], pts, [0, 2])
    zmin = _sort_face(np.where(pts[:, 2] == bnds[4])[0], pts, [0, 1])
    zmax = _sort_face(np.where(pts[:, 2] == bnds[5])[0], pts, [0, 1])
    nsets = [
        nset(i[0], i[1])
        for i in [
            (xmin, "xmin"),
            (xmax, "xmax"),
            (ymin, "ymin"),
            (ymax, "ymax"),
            (zmin, "zmin"),
            (zmax, "zmax"),
        ]
    ]
    eqs = np.stack(
        [
            np.concatenate([xmax, ymax, zmax]),
            np.concatenate([xmin, ymin, zmin]),
            np.concatenate(
                [np.zeros_like(xmax), 1 * np.ones_like(ymax), 2 * np.ones_like(zmax)]
            ),
        ]
    ).T
    eql = []
    for n, i in enumerate(eqs):
        if i[1] in eqs[:n, 0]:
            prev = np.where(eqs[:n, 0] == i[1])
            eq = list(i[0:3:2])
            for k in prev[0][-1:]:
                eq += eql[k][1:]
            eql.append(eq)
        else:
            eql.append(list(i))
    ss = []
    buf = ""
    for n in nsets:
        buf += n
    for _, i in enumerate(reversed(eql)):
        if i[0] not in ss:
            for j in range(3):
                if len(i) == 3:
                    buf += (
                        "*equation\n3\n"
                        + f"{i[0] + 1},{j + 1},-1,"
                        + f"{i[1] + 1},{j + 1},1,"
                        + f"{added_nodes[i[2]]},{j + 1},1\n"
                    )
                elif len(i) == 4:
                    buf += (
                        "*equation\n4\n"
                        + f"{i[0] + 1},{j + 1},-1,"
                        + f"{added_nodes[i[1]]},{j + 1},1,"
                        + f"{i[2] + 1},{j + 1},1,"
                        + f"{added_nodes[i[3]]},{j + 1},1\n"
                    )
                elif len(i) == 5:
                    buf += (
                        "*equation\n5\n"
                        + f"{i[0] + 1},{j + 1},-1,"
                        + f"{added_nodes[i[1]]},{j + 1},1,"
                        + f"{added_nodes[i[2]]},{j + 1},1,"
                        + f"{i[3] + 1},{j + 1},1\n"
                        + f"{added_nodes[i[4]]},{j + 1},1\n"
                    )
        ss.append(i[0])
    return buf


_CCX_TO_VTK_C3D8 = [0, 3, 2, 1, 4, 7, 6, 5]
_CCX_TO_VTK_C3D20 = [
    0,
    3,
    2,
    1,
    4,
    7,
    6,
    5,
    11,
    10,
    9,
    8,
    15,
    14,
    13,
    12,
    16,
    19,
    18,
    17,
]


def vtstoccx(mesh, outputname, resin, core, face=None, element_type="C3D8"):
    grd = mesh
    sigrd = grd.scale((1e-3, 1e-3, 1e-3), inplace=False)
    if element_type == "C3D8":
        nodes_per_cell = 8
        ccx_to_vtk = _CCX_TO_VTK_C3D8
    elif element_type == "C3D20":
        import vtk

        ug = (
            sigrd.cast_to_unstructured_grid()
            if hasattr(sigrd, "cast_to_unstructured_grid")
            else sigrd
        )
        f = vtk.vtkLinearToQuadraticCellsFilter()
        f.SetInputData(ug)
        f.Update()
        sigrd = pv.wrap(f.GetOutput())
        nodes_per_cell = 20
        ccx_to_vtk = _CCX_TO_VTK_C3D20
    else:
        raise ValueError(f"unsupported element_type: {element_type!r}")
    ids = (
        np.array(
            [
                [sigrd.GetCell(i).GetPointIds().GetId(j) for j in range(nodes_per_cell)]
                for i in range(sigrd.GetNumberOfCells())
            ]
        )
        + 1
    )
    out = ""
    out += "*node,nset=nall\n"
    for n, i in enumerate(sigrd.points):
        out += f"{n + 1},{i[0]},{i[1]},{i[2]}\n"
    added_nodes = []
    for i in range(3):
        added_nodes.append(n + i + 2)
        out += f"{added_nodes[-1]},-100,-100,-100\n"
    if "E" in core:
        out += f"*material,name=core\n*elastic,type=iso\n{core['E']},{core['nu']}\n"
    elif "E11" in core:
        out += "*material,name=core\n*elastic,type=engineering constants\n"
        out += f"{core['E11']}, {core['E22']}, {core['E33']}, {core['nu12']}, {core['nu13']}, {core['nu23']}, {core['G12']}, {core['G13']}\n{core['G23']},\n"
    out += f"*material,name=resin\n*elastic,type=iso\n{resin['E']},{resin['nu']}\n"
    out += "*material,name=face\n*elastic,type=iso\n12000000000,0.3\n"
    for n, i in enumerate(ids):
        out += f"*element,type={element_type},ELSET=e{n + 1}\n"
        reordered = [str(i[k]) for k in ccx_to_vtk]
        if nodes_per_cell <= 15:
            out += f"{n + 1}," + ",".join(reordered) + "\n"
        else:
            out += ",".join([str(n + 1), *reordered[:15]]) + ",\n"
            out += ",".join(reordered[15:]) + "\n"
        is_resin = sigrd.cell_data["resin"][n]
        is_face = sigrd.cell_data["face"][n]
        if bool(is_resin):
            out += f"*solid section,material=resin,elset=e{n + 1}\n"
        elif bool(is_face):
            out += f"*solid section,material=face,elset=e{n + 1}\n"
        else:
            out += f"*solid section,material=core,elset=e{n + 1}\n"
    perbc = periodic_bcs(sigrd, added_nodes)
    out += perbc
    stheader = "*STEP\n*STATIC\n" + "*boundary,op=new\n"
    stfooter = "*NODE OUTPUT,NSET=nall\nU,RF\n*ELEMENT OUTPUT\nS,E\n*END STEP\n"
    base_constraints = "1,1,3,0\n"
    xx, yy, zz, xy, xz, yz = [
        outputname.replace(".inp", f"_{i}.inp") for i in "xx,yy,zz,xy,xz,yz".split(",")
    ]
    open(xx, "w").write(
        out + stheader + base_constraints + f"{added_nodes[0]},1,1,1e-3\n" + stfooter
    )
    open(yy, "w").write(
        out + stheader + base_constraints + f"{added_nodes[1]},2,2,1e-3\n" + stfooter
    )
    open(zz, "w").write(
        out + stheader + base_constraints + f"{added_nodes[2]},3,3,1e-3\n" + stfooter
    )
    open(xy, "w").write(
        out
        + stheader
        + base_constraints
        + f"{added_nodes[0]},2,2,1e-3\n"
        + f"{added_nodes[1]},1,1,0\n"
        + stfooter
    )
    open(xz, "w").write(
        out
        + stheader
        + base_constraints
        + f"{added_nodes[0]},3,3,0\n"
        + f"{added_nodes[2]},1,1,1e-3\n"
        + stfooter
    )
    open(yz, "w").write(
        out
        + stheader
        + base_constraints
        + f"{added_nodes[1]},3,3,0\n"
        + f"{added_nodes[2]},2,2,1e-3\n"
        + stfooter
    )
    o = (xx, yy, zz, xy, xz, yz)
    print(f"Written outputs to {','.join(o)}")
    return o


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mesh")
    p.add_argument(
        "--Eresin", default=4e9, type=float, help="Elastic modulus for resin domain"
    )
    p.add_argument(
        "--nuresin", default=0.3, type=float, help="Poisson's ratio for resin domain"
    )
    p.add_argument(
        "--Ecore", default=4e9, type=float, help="Elastic modulus for core domain"
    )
    p.add_argument(
        "--nucore", default=0.3, type=float, help="Poisson's ratio for core domain"
    )
    p.add_argument("--output", default="__temp.inp")
    args = p.parse_args()
    vtstoccx(
        pv.read(args.mesh),
        args.output,
        {"E": args.Eresin, "nu": args.nuresin},
        {"E": args.Ecore, "nu": args.nucore},
    )


if __name__ == "__main__":
    main()
