#!/usr/bin/env python3

import pyvista as pv
import numpy as np
import argparse


def write_mpc(vtkids):
    lst = vtkids + 1
    done = [lst[int(len(lst) / 5)], lst[int(len(lst) / 1.5)], lst[int(len(lst) / 3)]]
    bf = f"*mpc\nplane,{done[0]},{done[1]},{done[2]},"
    st = 4
    for i in lst:
        if i not in done:
            bf += f"{i}"
            st += 1
            if st % 16 == 0:
                bf += "\n"
            else:
                bf += ","
    return bf if bf.endswith("\n") else bf + "\n", done


def write_nodeset(lst, name):
    bf = f"*nset,nset={name}\n"
    st = 0
    for i in lst:
        bf += f"{i}"
        st += 1
        if st % 16 == 0:
            bf += "\n"
        else:
            bf += ","
    return bf if bf.endswith("\n") else bf + "\n"


def planar_mpcs(msh, added_nodes):
    bnds = msh.GetBounds()
    xmin = np.where(msh.points[:, 0] == bnds[0])[0]
    xmax = np.where(msh.points[:, 0] == bnds[1])[0]
    ymin = np.where(msh.points[:, 1] == bnds[2])[0]
    ymax = np.where(msh.points[:, 1] == bnds[3])[0]
    zmin = np.where(msh.points[:, 2] == bnds[4])[0]
    zmax = np.where(msh.points[:, 2] == bnds[5])[0]
    xplan = ""
    for i in xmin:
        xplan += f"*equation\n2\n{i + 1},1,-1,{added_nodes[0]},1,1\n"
    for i in xmax:
        xplan += f"*equation\n2\n{i + 1},1,-1,{added_nodes[1]},1,1\n"
    yplan = ""
    for i in ymin:
        yplan += f"*equation\n2\n{i + 1},2,-1,{added_nodes[0]},2,1\n"
    for i in ymax:
        yplan += f"*equation\n2\n{i + 1},2,-1,{added_nodes[1]},2,1\n"
    zplan = ""
    for i in zmin:
        zplan += f"*equation\n2\n{i + 1},3,-1,{added_nodes[0]},3,1\n"
    for i in zmax:
        zplan += f"*equation\n2\n{i + 1},3,-1,{added_nodes[1]},3,1\n"
    return xplan, yplan, zplan


def shearbcs(msh, disp):
    bnds = msh.GetBounds()
    xmin = np.where(msh.points[:, 0] == bnds[0])[0]
    xmax = np.where(msh.points[:, 0] == bnds[1])[0]
    ymin = np.where(msh.points[:, 1] == bnds[2])[0]
    ymax = np.where(msh.points[:, 1] == bnds[3])[0]
    zmin = np.where(msh.points[:, 2] == bnds[4])[0]
    zmax = np.where(msh.points[:, 2] == bnds[5])[0]
    allbounds = np.unique(np.concatenate([xmin, xmax, ymin, ymax, zmin, zmax]))
    yy = msh.points[allbounds, 1]
    multy = disp / yy.max()
    zz = msh.points[allbounds, 2]
    multz = disp / zz.max()
    shearxy = ""
    for i in zip(allbounds, yy):
        shearxy += f"{i[0] + 1},1,1,{multy * i[1]}\n"
        shearxy += f"{i[0] + 1},2,3,0\n"
    shearxz = ""
    for i in zip(allbounds, zz):
        shearxz += f"{i[0] + 1},1,1,{multz * i[1]}\n"
        shearxz += f"{i[0] + 1},2,3,0\n"
    shearyz = ""
    for i in zip(allbounds, zz):
        shearyz += f"{i[0] + 1},2,2,{multz * i[1]}\n"
        shearyz += f"{i[0] + 1},1,1,0\n"
        shearyz += f"{i[0] + 1},3,3,0\n"
    return shearxy, shearxz, shearyz


def vtstoccx_planar(mesh, outputname, resin, core):
    grd = mesh
    sigrd = grd.scale((1e-3, 1e-3, 1e-3), inplace=False)
    ids = (
        np.array(
            [
                [sigrd.GetCell(i).GetPointIds().GetId(j) for j in range(8)]
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
    for i in range(2):
        added_nodes.append(n + i + 2)
        out += f"{added_nodes[-1]},-100,-100,-100\n"
    out += f"*material,name=core\n*elastic,type=iso\n{core['E']},{core['nu']}\n"
    out += f"*material,name=resin\n*elastic,type=iso\n{resin['E']},{resin['nu']}\n"
    for n, i in enumerate(ids):
        out += f"*element,type=C3D8,ELSET=e{n + 1}\n"
        out += f"{n + 1},{i[0]},{i[3]},{i[2]},{i[1]},{i[4]},{i[7]},{i[6]},{i[5]}\n"
        is_resin = sigrd.cell_data["resin"][n]
        if bool(is_resin):
            out += f"*solid section,material=resin,elset=e{n + 1}\n"
        else:
            out += f"*solid section,material=core,elset=e{n + 1}\n"
    xplan, yplan, zplan = planar_mpcs(sigrd, added_nodes)
    shearxy, shearxz, shearyz = shearbcs(sigrd, disp=1e-3)
    stheader = "*STEP\n*STATIC\n" + "*boundary,op=new\n"
    stfooter = "*NODE FILE,NSET=nall\nU,RF\n*EL FILE\nS,E\n*node print,nset=nall\nrf,U\n*END STEP\n"
    base_constraints = ""
    open(outputname.replace(".inp", "_xx.inp"), "w").write(
        out
        + xplan
        + yplan
        + zplan
        + stheader
        + base_constraints
        + f"{added_nodes[0]},1,1,0\n"
        + f"{added_nodes[1]},1,1,1e-3\n"
        + stfooter
    )
    open(outputname.replace(".inp", "_yy.inp"), "w").write(
        out
        + xplan
        + yplan
        + zplan
        + stheader
        + base_constraints
        + f"{added_nodes[0]},2,2,0\n"
        + f"{added_nodes[1]},2,2,1e-3\n"
        + stfooter
    )
    open(outputname.replace(".inp", "_zz.inp"), "w").write(
        out
        + xplan
        + yplan
        + zplan
        + stheader
        + base_constraints
        + f"{added_nodes[0]},3,3,0\n"
        + f"{added_nodes[1]},3,3,1e-3\n"
        + stfooter
    )
    open(outputname.replace(".inp", "_xy.inp"), "w").write(
        out + stheader + base_constraints + shearxy + stfooter
    )
    open(outputname.replace(".inp", "_xz.inp"), "w").write(
        out + stheader + base_constraints + shearxz + stfooter
    )
    open(outputname.replace(".inp", "_yz.inp"), "w").write(
        out + stheader + base_constraints + shearyz + stfooter
    )
    print(f"Written output to {outputname}")


if __name__ == "__main__":
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
    vtstoccx_planar(
        pv.read(args.mesh),
        args.output,
        {"E": args.Eresin, "nu": args.nuresin},
        {"E": args.Ecore, "nu": args.nucore},
    )
