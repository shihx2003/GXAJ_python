from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class SoilTypeParam:
    category: int
    shortening: str
    swc: float              # 3: SWC (饱和含水量)
    fc: float               # 4: FC (田间持水量)
    wp: float               # 5: WP (凋萎)
    rwc: float              # 6: RWC (残余含水量) [新增]
    shc_cm_h: float         # 7: SHC (饱和导水率 cm/h) [新增]
    lambda_: float          # 8: Lambda (孔径分布指数) [新增]


# Soiltype.tbl: Category, Shortening, SWC, FC, WP, RWC, SHC(cm/h), lambda
SOILTYPE_TABLE: dict[int, SoilTypeParam] = {
    0: SoilTypeParam(0, "Ocean", 0.0010, 0.00, 0.00, 0.00000,  0.000, 0.000),
    1: SoilTypeParam(1, "S",     0.3825, 0.15, 0.04, 0.03475, 42.230, 0.836),
    2: SoilTypeParam(2, "LS",    0.4100, 0.19, 0.05, 0.04600, 31.185, 0.691),
    3: SoilTypeParam(3, "SL",    0.4225, 0.27, 0.09, 0.06725,  7.535, 0.547),
    4: SoilTypeParam(4, "SiL",   0.4725, 0.35, 0.15, 0.03650,  1.635, 0.321),
    5: SoilTypeParam(5, "Si",    0.4800, 0.34, 0.11, 0.04000,  2.000, 0.300),
    6: SoilTypeParam(6, "L",     0.4400, 0.30, 0.14, 0.06000,  1.910, 0.368),
    7: SoilTypeParam(7, "SCL",   0.4250, 0.29, 0.16, 0.11225,  3.285, 0.402),
    8: SoilTypeParam(8, "SiCL",  0.4650, 0.41, 0.24, 0.09325,  0.380, 0.237),
    9: SoilTypeParam(9, "CL",    0.4525, 0.36, 0.21, 0.11550,  0.555, 0.308),
    10: SoilTypeParam(10, "SC",  0.4200, 0.33, 0.21, 0.10900,  0.450, 0.223),
    11: SoilTypeParam(11, "SiC", 0.4525, 0.43, 0.28, 0.10850,  0.230, 0.208),
    12: SoilTypeParam(12, "C",   0.4525, 0.40, 0.28, 0.14050,  0.260, 0.219),
}


def get_soil_type_param(category: int) -> SoilTypeParam:
    if category not in SOILTYPE_TABLE:
        raise KeyError(f"Unknown soil type category: {category}")
    return SOILTYPE_TABLE[category]


def soil_lookup(name: str) -> np.ndarray:
    return np.array([getattr(SOILTYPE_TABLE[i], name) for i in sorted(SOILTYPE_TABLE)], dtype=float)

SOIL_SWC = soil_lookup("swc")
SOIL_FC = soil_lookup("fc")
SOIL_WP = soil_lookup("wp")


@dataclass(frozen=True)
class LandcoverParam:
    category: int
    shortening: str
    lai_monthly: tuple[float, ...]
    max_lai: float
    canopy_top_height: float
    manning_n: float
    rooting_depth: float
    depletion_fraction: float


# Landcover.tbl: Category, Shortening, 12 monthly LAI values, MaxLAI,
# Canopy-Top Height, Manning's N, Rooting Depth, Depletion Fraction
#                                                                               12 monthly LAI values,                                                          MaxLAI, Canopy-Top Height, Manning's N, Rooting Depth, Depletion Fraction
LANDCOVER_TABLE: dict[int, LandcoverParam] = {
    0: LandcoverParam(0, "W",   (0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,      0.0,       0.0,       0.0      ), 0.0,      0.0,       0.03,  0.0,      0.0),
    1: LandcoverParam(1, "ENF", (8.76,      9.16,      9.827,     10.093,    10.36,     10.76,     10.493,    10.227,    10.093,   9.827,     9.16,      8.76     ), 10.76,    17.0,      0.1,   1.0,      0.4),
    2: LandcoverParam(2, "EBF", (5.117,     5.117,     5.117,     5.117,     5.117,     5.117,     5.117,     5.117,     5.117,    5.117,     5.117,     5.117    ), 6.0,      35.0,      0.1,   1.25,     0.4),
    3: LandcoverParam(3, "DNF", (8.76,      9.16,      9.827,     10.093,    10.36,     10.76,     10.493,    10.227,    10.093,   9.827,     9.16,      8.76     ), 10.76,    15.5,      0.1,   1.0,      0.4),
    4: LandcoverParam(4, "DBF", (0.52,      0.52,      0.867,     2.107,     4.507,     6.773,     7.173,     6.507,     5.04,     2.173,     0.867,     0.52     ), 7.173,    20.0,      0.1,   1.25,     0.4),
    5: LandcoverParam(5, "MF",  (4.64,      4.84,      5.347,     6.1,       7.4335,    8.7665,    8.833,     8.367,     7.5665,   6.0,       5.0135,    4.64     ), 8.833,    19.25,     0.1,   1.125,    0.5),
    6: LandcoverParam(6, "Wo",  (5.276088,  5.528588,  6.006132,  6.442597,  7.244881,  8.363948,  8.540044,  8.126544,  7.253301, 6.329191,  5.625809,  5.300508 ), 8.54004,  14.3379,   0.1,   0.997475, 0.55),
    7: LandcoverParam(7, "WG",  (2.333182,  2.482112,  2.72661,   3.033015,  3.884949,  5.521223,  6.239513,  5.773302,  4.15567,  3.127464,  2.618012,  2.403912 ), 6.23951,  7.0426,    0.3,   0.872075, 0.3),
    8: LandcoverParam(8, "CS",  (0.580555,  0.6290065, 0.628558,  0.628546,  0.919255,  1.768545,  2.550697,  2.553597,  1.728642, 0.9703975, 0.726358,  0.6290065), 5.0672,   0.60027,   0.3,   0.650795, 0.3),
    9: LandcoverParam(9, "OS",  (0.3999679, 0.4043968, 0.3138257, 0.2232945, 0.2498679, 0.3300675, 0.4323964, 0.7999234, 1.166883, 0.7977234, 0.5038257, 0.4043968), 6.0023,   0.51346,   0.2,   0.577705, 0.25),
    10: LandcoverParam(10, "G",  (0.782,    0.893,     1.004,     1.116,     1.782,     3.671,     4.782,     4.227,     2.004,    1.227,     1.004,     0.893    ), 4.782,    0.5666667, 0.17,  0.75,     0.3),
    11: LandcoverParam(11, "C",  (0.782,    0.893,     1.004,     1.116,     1.782,     3.671,     4.782,     4.227,     2.004,    1.227,     1.004,     0.893    ), 5.976525, 0.55,      0.035, 0.75,     0.25),
    12: LandcoverParam(12, "BG", (0.001,    0.001,     0.001,     0.001,     0.001,     0.001,     0.001,     0.001,     0.001,    0.001,     0.001,     0.001    ), 0.7439,   0.2,       0.01,  0.55,     0.001),
    13: LandcoverParam(13, "UB", (1.286714, 1.3946,    1.550698,  1.772726,  2.519023,  4.136768,  5.021229,  4.57958,   2.848436, 1.885623,  1.517874,  1.36568  ), 5.02123,  6.017258,  0.015, 0.79722,  0.25),
}

def get_landcover_param(category: int) -> LandcoverParam:
    if category not in LANDCOVER_TABLE:
        raise KeyError(f"Unknown landcover category: {category}")
    return LANDCOVER_TABLE[category]


MON_LAI = np.array([[LANDCOVER_TABLE[i].lai_monthly[j] for j in range(12)] for i in sorted(LANDCOVER_TABLE)], dtype=float)
MAX_LAI = np.array([LANDCOVER_TABLE[i].max_lai for i in sorted(LANDCOVER_TABLE)], dtype=float)
CTOPH = np.array([LANDCOVER_TABLE[i].canopy_top_height for i in sorted(LANDCOVER_TABLE)], dtype=float)


if __name__ == "__main__":
    print(MON_LAI[6, 3])
    print(MAX_LAI)
    print(CTOPH)