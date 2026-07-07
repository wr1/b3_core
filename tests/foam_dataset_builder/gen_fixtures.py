"""Generate 3 synthetic FEA result fixtures for Phase 1 dataset builder testing."""
import numpy as np
import json
import os

WORKSPACE = "/home/wr/.hermes/kanban/boards/blade3/workspaces/t_2c562aac"

def engineering_to_stiffness(Exx, Eyy, Ezz, Gxy, Gxz, Gyz, nuxy, nuxz, nuyz):
    """Convert 9 engineering constants to 6x6 stiffness tensor (Voigt notation)."""
    S = np.zeros((6, 6))
    S[0, 0] = 1.0 / Exx
    S[1, 1] = 1.0 / Eyy
    S[2, 2] = 1.0 / Ezz
    S[3, 3] = 1.0 / Gxy
    S[4, 4] = 1.0 / Gxz
    S[5, 5] = 1.0 / Gyz
    S[0, 1] = S[1, 0] = -nuxy / Exx
    S[0, 2] = S[2, 0] = -nuxz / Exx
    S[1, 2] = S[2, 1] = -nuyz / Eyy
    C = np.linalg.inv(S)
    upper = [C[i, j] for i in range(6) for j in range(i, 6)]
    return C, upper


def make_fixture(foam_type, name, Exx, Eyy, Ezz, Gxy, Gxz, Gyz,
                 nuxy, nuxz, nuyz, core_E, core_nu, core_rho,
                 resin_E, resin_nu, resin_rho, dx, dy, thickness,
                 xgr, ygr, curvature, backend, md5):
    """Create a synthetic cprop() output JSON fixture."""
    C, upper = engineering_to_stiffness(Exx, Eyy, Ezz, Gxy, Gxz, Gyz, nuxy, nuxz, nuyz)
    resin_vf = 0.35
    eff_vf = 0.38
    halo_vf = 0.03
    area_increase = 1.25 if not ygr else 1.35
    rho_infused = core_rho * (1.0 - eff_vf) + resin_rho * eff_vf

    upper_rounded = [round(float(u), 6) for u in upper]
    upper_dict = {}
    k = 0
    for i in range(6):
        for j in range(i, 6):
            upper_dict[f"C_{i}{j}"] = upper_rounded[k]
            k += 1

    fixture = {
        "_meta": {
            "foam_type": foam_type,
            "backend": backend,
            "provenance": "synthetic fixture for Phase 1 dataset builder",
            "date": "2026-07-07",
        },
        "hash": md5,
        "Exx": Exx,
        "Eyy": Eyy,
        "Ezz": Ezz,
        "Gxy": Gxy,
        "Gxz": Gxz,
        "Gyz": Gyz,
        "nuxy": nuxy,
        "nuxz": nuxz,
        "nuyz": nuyz,
        "C_stiffness_flat": upper_rounded,
        "C_stiffness_upper": upper_dict,
        "dx": dx,
        "dy": dy,
        "thickness": thickness,
        "xgr": xgr,
        "ygr": ygr,
        "core": {"E": core_E, "nu": core_nu, "rho": core_rho},
        "resin": {"E": resin_E, "nu": resin_nu, "rho": resin_rho},
        "curvature": curvature,
        "backend": backend,
        "resin_vf": resin_vf,
        "effective_resin_vf": eff_vf,
        "halo_vf": halo_vf,
        "area_increase": area_increase,
        "rho_infused": rho_infused,
    }
    return fixture


if __name__ == "__main__":
    fixtures = [
        # Fixture 1: PVC high foam (Divinycell H250 equivalent)
        make_fixture(
            foam_type="pvc_foam_high", name="pvc_high_plain",
            Exx=320e6, Eyy=320e6, Ezz=22e6,
            Gxy=160e6, Gxz=12e6, Gyz=12e6,
            nuxy=0.30, nuxz=0.05, nuyz=0.05,
            core_E=130e6, core_nu=0.30, core_rho=100,
            resin_E=3.0e9, resin_nu=0.35, resin_rho=1100,
            dx=50, dy=50, thickness=30,
            xgr=[[10, 10, 8, 3]], ygr=[],
            curvature={},
            backend="mfem",
            md5="a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4",
        ),
        # Fixture 2: PET foam (Divinagard equivalent)
        make_fixture(
            foam_type="pet_foam", name="pet_low_curved",
            Exx=90e6, Eyy=90e6, Ezz=7e6,
            Gxy=45e6, Gxz=5e6, Gyz=5e6,
            nuxy=0.35, nuxz=0.06, nuyz=0.06,
            core_E=60e6, core_nu=0.30, core_rho=50,
            resin_E=3.0e9, resin_nu=0.35, resin_rho=1100,
            dx=50, dy=50, thickness=25,
            xgr=[[8, 8, 6, 2.5]], ygr=[],
            curvature={"kx": -0.005},
            backend="mfem",
            md5="b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5",
        ),
        # Fixture 3: Balsa foam
        make_fixture(
            foam_type="balsa_foam", name="balsa_crossed",
            Exx=2200e6, Eyy=2200e6, Ezz=450e6,
            Gxy=880e6, Gxz=220e6, Gyz=220e6,
            nuxy=0.30, nuxz=0.04, nuyz=0.04,
            core_E=1800e6, core_nu=0.30, core_rho=180,
            resin_E=3.0e9, resin_nu=0.35, resin_rho=1100,
            dx=50, dy=50, thickness=40,
            xgr=[[12, 12, 10, 4]], ygr=[[12, 12, 10, 4]],
            curvature={},
            backend="mfem",
            md5="c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6",
        ),
    ]

    for i, fixture in enumerate(fixtures, 1):
        fname = os.path.join(WORKSPACE, f"synthetic_fixture_{i}.json")
        with open(fname, "w") as fh:
            json.dump(fixture, fh, indent=2)
        print(
            f"Wrote {fname}: "
            f"foam_type={fixture['_meta']['foam_type']}, "
            f"Exx={fixture['Exx'] / 1e6:.1f} MPa"
        )

    print(f"\nDone - {len(fixtures)} synthetic fixtures created in {WORKSPACE}")