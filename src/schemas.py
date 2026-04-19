from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

@dataclass
class SimulationConfig:
    start_date: date
    end_date: date
    FreeWaterCoeff: float   # 自由水系数 (Free Water Coefficient) - 需率定
    TensionWaterCoeff: float # 毛细水系数 (Tension Water Coefficient) - 需率定
    ini_sm: float


@dataclass
class CalSort:
    Length: int
    CalOrder : np.ndarray
    SortRow : np.ndarray
    SortCol : np.ndarray

@dataclass
class GeoStatic:
    Basin : np.ndarray                          # Layer 0: Basin Mask (流域掩膜, 0/1)

    Longitude : np.ndarray                            # Layer 0: Longitude (经度, °)
    Latitude : np.ndarray                            # Layer 0: Latitude (纬度, °)

    DEM : np.ndarray                            # Layer 1: DEM (高程, m)
    RiverChannel : np.ndarray                   # Layer 2: River Channel (河道掩膜, 0/1)
    TensionWaterCapacity : np.ndarray           # Layer 3: Tension Water Capacity (WM, mm) - 需率定
    FreedomWaterCapacity : np.ndarray           # Layer 4: Free Water Capacity (WUM, mm) - 需率定
    Top30cmSoilType : np.ndarray                # Layer 5: Top Soil Type ID (0-30cm) -> 对应 Soil.tbl
    Deep30to100cmSoilType : np.ndarray          # Layer 6: Deep Soil Type ID (30-100cm) -> 对应 Soil.tbl
    HumusSoilDepth : np.ndarray                 # Layer 7: Humus Soil Depth (腐殖土厚度, m?) - 需率定
    VegetationType : np.ndarray                 # Layer 8: Vegetation Type ID -> 对应 Landcover.tbl
                                                # Layer 9: 
    RunoffDistributionRatio : np.ndarray        # Layer 10: Runoff Distribution Ratio (径流分配比)
    GridFlowDirection : np.ndarray              # Layer 11: Flow Direction (流向 D8)       
    VadoseZoneDepth : np.ndarray                # Layer 12: Vadose Zone Depth (包气带厚度, m)
    FlowAccumulationArea : np.ndarray           # Layer 13: Flow Accumulation (累积汇流面积)


@dataclass
class LumParams:
    OC : float      # Col 2: OC (Outflow Coefficient?)            ! OC = LumPara(1)
    ROC: int        # Col 3: ROC (Runoff Coefficient?)            ! ROC = LumPara(2)
    K   : float     # Col 4: K (Evaporation Coefficient?)         ! KEpC = LumPara(3)
    C   : float     # Col 5: C (Deeper Layer Coefficient?)        ! DeeperC = LumPara(4)
    LUM: float      # Col 6: LUM (Upper Layer Coefficient?)       ! AlUpper = LumPara(5)
    LLM: float      # Col 7: LLM (Lower Layer Coefficient?)       ! AlLower = LumPara(6)
    CG   : float    # Col 8: CG (Groundwater Recession Constant?) ! CCg = LumPara(7)
    CI  : float     # Col 9: CI (Interflow Recession Constant?)   ! CCi = LumPara(8)
    CS : float      # Col 10: CS (Surface Recession Constant?)    ！CCs = LumPara(9)
    CCS: float
    LT: int         # Col 11: LT (Lag Time?)                      ! LagTime1 = LumPara(10)
    CHM: int        # Col 12: CHM                                 ! CCS1 = LumPara(11)
    LTH: int        # Col 13: LTH                                 ! LagTime = LumPara(12)
    Kech: float     # Col 14: Kech (Muskingum K Channel?)         ! MKch = LumPara(13)
    Kes : float     # Col 15: Kes (Muskingum K Surface?)          ! MKs = LumPara(14)
    Kei: float      # Col 16: Kei (Muskingum K Interflow?)        ! MKi = LumPara(15)
    Keg: float      # Col 17: Keg (Muskingum K Groundwater?)      ! MKg = LumPara(16)
    Xech: float     # Col 18: Xech (Muskingum X Channel?)         ! MXch = LumPara(17)
    Xes: float      # Col 19: Xes (Muskingum X Surface?)          ! MXs = LumPara(18)
    Xei: float      # Col 20: Xei (Muskingum X Interflow?)        ! MXi = LumPara(19)
    Xeg : float     # Col 21: Xeg (Muskingum X Groundwater?)      ! MXg = LumPara(20)
    Ki_tmp: float   # ! Col 22: Ki_tmp                            ! Ki_tmp=LumPara(21)

@dataclass
class EstimatedState:
    GridWUM: np.ndarray
    GridWLM: np.ndarray
    GridWDM: np.ndarray

    GridKi: np.ndarray
    GridKg: np.ndarray
    SumKgKi: np.ndarray

    GridFLC: np.ndarray

@dataclass
class CanopyState:
    Pcum: np.ndarray
    Icum_prev: np.ndarray
    Ica : np.ndarray
    Wca : np.ndarray
    Pnet : np.ndarray

@dataclass
class EvaporationState:
    Ep : np.ndarray
    Ecan : np.ndarray
    Eu : np.ndarray
    El : np.ndarray
    Ed : np.ndarray

@dataclass
class SoilState:
    WS : np.ndarray
    WU : np.ndarray
    WL : np.ndarray
    WD : np.ndarray

@dataclass
class RunoffState:
    Pe : np.ndarray
    R : np.ndarray
    Rs : np.ndarray
    Ri : np.ndarray
    Rg : np.ndarray
@dataclass
class RoutingState:
    WQs : np.ndarray
    WQi : np.ndarray
    WQg : np.ndarray
    WQch : np.ndarray

    Qs_in : np.ndarray
    Qi_in : np.ndarray
    Qg_in : np.ndarray
    Qch_in : np.ndarray

    Qs_out : np.ndarray
    Qi_out : np.ndarray
    Qg_out : np.ndarray
    Qch_out : np.ndarray

    QLagTime: list

@dataclass
class ModelState:
    CanopyState: CanopyState
    EvaporationState: EvaporationState
    SoilState: SoilState
    RunoffState: RunoffState
    RoutingState: RoutingState
