# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Create  :   2026-04-02 14:54:54
@Author  :   shihx2003
@Version :   1.0
@Contact :   shihx2003@outlook.com
'''

# here put the import lib
from datetime import timedelta
from typing import Optional
import numpy as np
import pandas as pd
import xarray as xr
from numba import njit

from core.schemas import GeoStatic, LumParams, SimulationConfig, CalSort, ModelState, EvaporationState, SoilState, CanopyState, RunoffState, RoutingState
from core.TBL import SOIL_SWC, SOIL_FC, SOIL_WP, MON_LAI, MAX_LAI, CTOPH

from utils.draw import plot_2d

class GridXAJDailyModel:
    
    PI = 3.141592654
    EARTH_RADIUS_KM = 6370.997

    DIV = 5.0
    DT = 24.0
    ETKCB_MIN = (0.15 + 0.2) / 2.0

    def __init__(self, config: SimulationConfig, geostatic: GeoStatic, lumparams: LumParams, calorder: CalSort) -> None:
        self.config = config
        self.geostatic = geostatic
        self.lumparams = lumparams
        self.calorder = calorder
        self.Nx = geostatic.Basin.shape[1]
        self.Ny = geostatic.Basin.shape[0]

    def _initialize_from_Zero(self):
        self.geostatic.FreedomWaterCapacity = self.geostatic.FreedomWaterCapacity * self.config.FreeWaterCoeff
        self.geostatic.TensionWaterCapacity = self.geostatic.TensionWaterCapacity * self.config.TensionWaterCoeff
        self.geostatic.HumusSoilDepth       = self.geostatic.HumusSoilDepth * self.config.TensionWaterCoeff

        self.GridWUM, self.GridWLM, self.GridWDM = self._Estimate_WM()
        self.GridKi, self.GridKg, self.SumKgKi   = self._Estimate_KiKg()
        self.GridFLC                             = self._Estimate_FLC()
        self.GridArea                            = self._GridArea()
        self.CT = self.GridArea / (self.DT * 3.6)

        self.state = ModelState(
            CanopyState = CanopyState(
                Pcum=np.zeros((self.Ny, self.Nx), dtype=float),
                Icum_prev=np.zeros((self.Ny, self.Nx), dtype=float),
                Ica=np.zeros((self.Ny, self.Nx), dtype=float),
                Wca=np.zeros((self.Ny, self.Nx), dtype=float),
                Pnet=np.zeros((self.Ny, self.Nx), dtype=float)
            ),
            EvaporationState = EvaporationState(
                Ep=np.zeros((self.Ny, self.Nx), dtype=float),
                Ecan=np.zeros((self.Ny, self.Nx), dtype=float),
                Eu=np.zeros((self.Ny, self.Nx), dtype=float),
                El=np.zeros((self.Ny, self.Nx), dtype=float),
                Ed=np.zeros((self.Ny, self.Nx), dtype=float)
            ),
            SoilState = SoilState(
                WS=self.geostatic.FreedomWaterCapacity.copy() * self.config.ini_sm / 10,
                WU=self.GridWUM.copy() * self.config.ini_sm / 10,
                WL=self.GridWLM.copy() * self.config.ini_sm / 10,
                WD=self.GridWDM.copy() * self.config.ini_sm / 10
            ),
            RunoffState=RunoffState(
                Pe=np.zeros((self.Ny, self.Nx), dtype=float),
                R =np.zeros((self.Ny, self.Nx), dtype=float),
                Rs=np.zeros((self.Ny, self.Nx), dtype=float),
                Ri=np.zeros((self.Ny, self.Nx), dtype=float),
                Rg=np.zeros((self.Ny, self.Nx), dtype=float)
            ), 
            RoutingState=RoutingState(
                WQs=np.zeros((self.Ny, self.Nx), dtype=float),
                WQi=np.zeros((self.Ny, self.Nx), dtype=float),
                WQg=np.zeros((self.Ny, self.Nx), dtype=float),
                WQch=np.zeros((self.Ny, self.Nx), dtype=float),

                Qs_in=np.zeros((self.Ny, self.Nx), dtype=float),
                Qi_in=np.zeros((self.Ny, self.Nx), dtype=float),
                Qg_in=np.zeros((self.Ny, self.Nx), dtype=float),
                Qch_in=np.zeros((self.Ny, self.Nx), dtype=float),

                Qs_out=np.zeros((self.Ny, self.Nx), dtype=float),
                Qi_out=np.zeros((self.Ny, self.Nx), dtype=float),
                Qg_out=np.zeros((self.Ny, self.Nx), dtype=float),
                Qch_out=np.zeros((self.Ny, self.Nx), dtype=float),
                QLagTime = list([0.0] * (self.lumparams.LT + 1))
            )
        )

    def _initialize_from_Restart(self, Restart: xr.Dataset):
        self.geostatic.FreedomWaterCapacity = Restart["FreedomWaterCapacity"].values
        self.geostatic.TensionWaterCapacity = Restart["TensionWaterCapacity"].values
        self.geostatic.HumusSoilDepth       = Restart["HumusSoilDepth"].values

        self.GridWUM, self.GridWLM, self.GridWDM = Restart["GridWUM"].values, Restart["GridWLM"].values, Restart["GridWDM"].values
        self.GridKi, self.GridKg, self.SumKgKi   = Restart["GridKi"].values, Restart["GridKg"].values, Restart["GridKi"].values + Restart["GridKg"].values
        self.GridFLC                             = Restart["GridFLC"].values
        self.GridArea                            = self._GridArea()
        self.CT = self.GridArea / (self.DT * 3.6)

        self.state = ModelState(
            CanopyState = CanopyState(
                Pcum=np.zeros((self.Ny, self.Nx), dtype=float),
                Icum_prev=np.zeros((self.Ny, self.Nx), dtype=float),
                Ica=np.zeros((self.Ny, self.Nx), dtype=float),
                Wca=np.zeros((self.Ny, self.Nx), dtype=float),
                Pnet=np.zeros((self.Ny, self.Nx), dtype=float)
            ),
            EvaporationState = EvaporationState(
                Ep=np.zeros((self.Ny, self.Nx), dtype=float),
                Ecan=np.zeros((self.Ny, self.Nx), dtype=float),
                Eu=np.zeros((self.Ny, self.Nx), dtype=float),
                El=np.zeros((self.Ny, self.Nx), dtype=float),
                Ed=np.zeros((self.Ny, self.Nx), dtype=float)
            ),
            SoilState = SoilState(
                WS=Restart["WS"].values,
                WU=Restart["WU"].values,
                WL=Restart["WL"].values,
                WD=Restart["WD"].values
            ),
            RunoffState=RunoffState(
                Pe=np.zeros((self.Ny, self.Nx), dtype=float),
                R =np.zeros((self.Ny, self.Nx), dtype=float),
                Rs=np.zeros((self.Ny, self.Nx), dtype=float),
                Ri=np.zeros((self.Ny, self.Nx), dtype=float),
                Rg=np.zeros((self.Ny, self.Nx), dtype=float)
            ), 
            RoutingState=RoutingState(
                WQs=Restart["WQs"].values,
                WQi=Restart["WQi"].values,
                WQg=Restart["WQg"].values,
                WQch=Restart["WQch"].values,

                Qs_in=np.zeros((self.Ny, self.Nx), dtype=float),
                Qi_in=np.zeros((self.Ny, self.Nx), dtype=float),
                Qg_in=np.zeros((self.Ny, self.Nx), dtype=float),
                Qch_in=np.zeros((self.Ny, self.Nx), dtype=float),

                Qs_out=np.zeros((self.Ny, self.Nx), dtype=float),
                Qi_out=np.zeros((self.Ny, self.Nx), dtype=float),
                Qg_out=np.zeros((self.Ny, self.Nx), dtype=float),
                Qch_out=np.zeros((self.Ny, self.Nx), dtype=float),
                QLagTime = list([0.0] * (self.lumparams.LT + 1))
            )
        )

    def _GridArea(self):
        lat_rad = np.radians(self.geostatic.Latitude)
        dlat = np.radians(self.geostatic.Latitude[1, 0] - self.geostatic.Latitude[0, 0])
        dlon = np.radians(self.geostatic.Longitude[0, 1] - self.geostatic.Longitude[0, 0])
        GridArea = np.mean((self.EARTH_RADIUS_KM ** 2) * np.abs(np.sin(lat_rad + dlat / 2) - np.sin(lat_rad - dlat / 2)) * dlon)
        return GridArea

    def _Estimate_FLC(self):
        GridFLC = np.zeros((12, self.Ny, self.Nx), dtype=float)
        GridVegType = self.geostatic.VegetationType.astype(int)
        for i in range(self.Ny):
            for j in range(self.Nx):
                for k in range(12):                # k 为月份索引，0-11，对应1-12月
                    ETKcb = 1.07 * (1 - np.exp(-0.84 * MON_LAI[GridVegType[i, j], k]))
                    ETKcbmax = 1.07 * (1 - np.exp(-0.84 * MAX_LAI[GridVegType[i, j]]))

                    if ETKcb - self.ETKCB_MIN < 0.0:
                        Flc = 0.0
                    else:
                        Flc = ((ETKcb - self.ETKCB_MIN) / (ETKcbmax + 0.05 - self.ETKCB_MIN)) ** (1. + 0.5 * CTOPH[GridVegType[i, j]])
                    GridFLC[k, i, j] = Flc
        return GridFLC

    def _Estimate_WM(self):
        GridWUM = np.zeros((self.Ny, self.Nx), dtype=float)
        GridWLM = np.zeros((self.Ny, self.Nx), dtype=float)
        GridWDM = np.zeros((self.Ny, self.Nx), dtype=float)

        GridWM = self.geostatic.TensionWaterCapacity
        GridSM = self.geostatic.FreedomWaterCapacity
        ThickoVZ = self.geostatic.VadoseZoneDepth

        AlUpper = self.lumparams.LUM
        AlLower = self.lumparams.LLM
        AlDeeper = 1.0 - AlUpper - AlLower

        for i in range(self.Ny):
            for j in range(self.Nx):
                ZUpper = AlUpper * ThickoVZ[i, j]
                ZLower = AlLower * ThickoVZ[i, j]
                ZDeeper = AlDeeper * ThickoVZ[i, j]

                SType030_val = int(self.geostatic.Top30cmSoilType[i, j])
                SType30100_val = int(self.geostatic.Deep30to100cmSoilType[i, j])
                
                if ZUpper > 300.0:
                    WUM_val = (SOIL_FC[SType030_val] - SOIL_WP[SType030_val]) * 300.0 + \
                            (SOIL_FC[SType30100_val] - SOIL_WP[SType30100_val]) * (ZUpper - 300.0)
                    WLM_val = (SOIL_FC[SType30100_val] - SOIL_WP[SType30100_val]) * (ZLower - 300.0)
                else:
                    WUM_val = (SOIL_FC[SType030_val] - SOIL_WP[SType030_val]) * ZUpper
                    if ZUpper + ZLower > 300.0:
                        WLM_val = (SOIL_FC[SType030_val] - SOIL_WP[SType030_val]) * (300.0 - ZUpper) + \
                                (SOIL_FC[SType30100_val] - SOIL_WP[SType30100_val]) * (ZLower - 300.0 + ZUpper)
                    else:
                        WLM_val = (SOIL_FC[SType030_val] - SOIL_WP[SType030_val]) * ZLower
                WDM_val = GridWM[i, j] - WUM_val - WLM_val
                GridWUM[i, j] = WUM_val
                GridWLM[i, j] = WLM_val
                GridWDM[i, j] = WDM_val

        return GridWUM, GridWLM, GridWDM

    def _Estimate_KiKg(self):
        HumusSoilDepth = self.geostatic.HumusSoilDepth
        Top30cmSoilType = self.geostatic.Top30cmSoilType.astype(int)
        Deep30to100cmSoilType = self.geostatic.Deep30to100cmSoilType.astype(int)

        # Masks for shallow and deep soils
        shallow_mask = HumusSoilDepth <= 300.0
        deep_mask = ~shallow_mask

        # Prepare arrays
        ThitaS = np.zeros((self.Ny, self.Nx), dtype=float)
        ThitaF = np.zeros((self.Ny, self.Nx), dtype=float)
        ThitaW = np.zeros((self.Ny, self.Nx), dtype=float)

        # Shallow
        ThitaS[shallow_mask] = SOIL_SWC[Top30cmSoilType[shallow_mask]]
        ThitaF[shallow_mask] = SOIL_FC[Top30cmSoilType[shallow_mask]]
        ThitaW[shallow_mask] = SOIL_WP[Top30cmSoilType[shallow_mask]]

        # Deep
        ratio = np.zeros((self.Ny, self.Nx), dtype=float)
        ratio[deep_mask] = 300.0 / HumusSoilDepth[deep_mask]
        ThitaS[deep_mask] = SOIL_SWC[Top30cmSoilType[deep_mask]] * ratio[deep_mask] + SOIL_SWC[Deep30to100cmSoilType[deep_mask]] * (1 - ratio[deep_mask])
        ThitaF[deep_mask] = SOIL_FC[Top30cmSoilType[deep_mask]] * ratio[deep_mask] + SOIL_FC[Deep30to100cmSoilType[deep_mask]] * (1 - ratio[deep_mask])
        ThitaW[deep_mask] = SOIL_WP[Top30cmSoilType[deep_mask]] * ratio[deep_mask] + SOIL_WP[Deep30to100cmSoilType[deep_mask]] * (1 - ratio[deep_mask])

        # Calculate
        OC = self.lumparams.OC
        ROC = self.lumparams.ROC
        ThitaF_div_ThitaS = ThitaF / ThitaS
        power = ThitaF_div_ThitaS ** OC
        DRMKi = power / (1 + ROC / (1 + 2 * (1 - ThitaW)))
        DRMKg = power - DRMKi
        SumKgKi = np.sum(power)

        return DRMKi, DRMKg, SumKgKi

        # 原来的计算方法，暂时舍弃，上面为使用np加速的方法
        # HumusSoilDepth = self.geostatic.HumusSoilDepth
        # ThitaS = np.zeros_like(HumusSoilDepth, dtype=float)
        # ThitaF = np.zeros_like(HumusSoilDepth, dtype=float)
        # ThitaW = np.zeros_like(HumusSoilDepth, dtype=float)
        # print(ThitaS.shape)
        # for i in range(self.Ny):q
        #     for j in range(self.Nx):
        #         if self.geostatic.HumusSoilDepth[i,j] <= 300.0:
        #             ThitaS[i,j] = SOIL_SWC[self.geostatic.Top30cmSoilType[i,j].astype(int)]
        #             ThitaF[i,j] = SOIL_FC[self.geostatic.Top30cmSoilType[i,j].astype(int)]
        #             ThitaW[i,j] = SOIL_WP[self.geostatic.Top30cmSoilType[i,j].astype(int)]
        #         else:
        #             ThitaS[i,j] = SOIL_SWC[self.geostatic.Top30cmSoilType[i,j].astype(int)] * (300.0 / self.geostatic.HumusSoilDepth[i,j]) \
        #                 + SOIL_SWC[self.geostatic.Deep30to100cmSoilType[i,j].astype(int)] * (1 - 300.0 / self.geostatic.HumusSoilDepth[i,j])
        #             ThitaF[i,j] = SOIL_FC[self.geostatic.Top30cmSoilType[i,j].astype(int)] * (300.0 / self.geostatic.HumusSoilDepth[i,j]) \
        #                 + SOIL_FC[self.geostatic.Deep30to100cmSoilType[i,j].astype(int)] * (1 - 300.0 / self.geostatic.HumusSoilDepth[i,j])
        #             ThitaW[i,j] = SOIL_WP[self.geostatic.Top30cmSoilType[i,j].astype(int)] * (300.0 / self.geostatic.HumusSoilDepth[i,j]) \
        #                 + SOIL_WP[self.geostatic.Deep30to100cmSoilType[i,j].astype(int)] * (1 - 300.0 / self.geostatic.HumusSoilDepth[i,j])
        # DRMKi = ((ThitaF / ThitaS) ** self.lumparams.OC) / (1 + self.lumparams.ROC / (1 + 2 * (1 - ThitaW)))
        # DRMKg = (ThitaF / ThitaS) ** self.lumparams.OC - DRMKi
        # SumKgKi = np.sum((ThitaF / ThitaS) ** self.lumparams.OC)

        # return DRMKi, DRMKg, SumKgKi

    def _save_Restart(self, file_path: str):
        Restart_vars = {
            # 估计的中间变量 (重要，否则重启动后计算不一致)
            "GridWUM": (["y", "x"], self.GridWUM),
            "GridWLM": (["y", "x"], self.GridWLM),
            "GridWDM": (["y", "x"], self.GridWDM),
            "GridKi": (["y", "x"], self.GridKi),
            "GridKg": (["y", "x"], self.GridKg),

            "GridFLC" : (["mon", "y", "x"], self.GridFLC),

            "FreedomWaterCapacity" : (["y", "x"], self.geostatic.FreedomWaterCapacity),
            "TensionWaterCapacity" : (["y", "x"], self.geostatic.TensionWaterCapacity),
            "HumusSoilDepth" : (["y", "x"], self.geostatic.HumusSoilDepth),

            # Soil State
            "WS": (["y", "x"], self.state.SoilState.WS),
            "WU": (["y", "x"], self.state.SoilState.WU),
            "WL": (["y", "x"], self.state.SoilState.WL),
            "WD": (["y", "x"], self.state.SoilState.WD),

            # Routing State
            "WQs": (["y", "x"], self.state.RoutingState.WQs),
            "WQi": (["y", "x"], self.state.RoutingState.WQi),
            "WQg": (["y", "x"], self.state.RoutingState.WQg),
            "WQch": (["y", "x"], self.state.RoutingState.WQch),

            # 地理静态变量
            "lon": (["y", "x"], self.geostatic.Longitude),
            "lat": (["y", "x"], self.geostatic.Latitude)
        }

        ds = xr.Dataset(
                    data_vars=Restart_vars,
                    coords={
                        "mon": np.arange(12),
                        "x": np.arange(self.Nx),
                        "y": np.arange(self.Ny),
                    }
        )

        ds.to_netcdf(file_path)


    def Canopy_Interception(self, time, precip):

        def Canopy_Inter_cell(P, LAI, FLC, Wca, Pcum, Icum_prev, Ica):
            if Wca <= 0.0:
                Pcum = 0.0
                Icum_prev = 0.0
            Pcum = Pcum + P
            Cvd   = 0.046 * LAI
            Scmax = max(0.935 + 0.498 * LAI - 0.00575 * LAI ** 2, 0.0)
            Icum_new = FLC * Scmax * (1.0 - np.exp(-Cvd * Pcum / Scmax))
            Ica = max(0.0, Icum_new - Icum_prev)
            Icum_prev = Icum_new

            return Pcum, Icum_prev, Ica

        mon =  pd.to_datetime(time).month

        for k in range(self.calorder.Length):
            i = self.calorder.SortRow[k]
            j = self.calorder.SortCol[k]

            P = precip[i, j]
            LAI = MON_LAI[self.geostatic.VegetationType[i, j].astype(int), mon - 1]
            FLC = self.GridFLC[mon - 1, i, j]
            Wca = self.state.CanopyState.Wca[i, j]
            Pcum = self.state.CanopyState.Pcum[i, j]
            Icum_prev = self.state.CanopyState.Icum_prev[i, j]
            Ica = self.state.CanopyState.Ica[i, j]

            Pcum, Icum_prev, Ica = Canopy_Inter_cell(P, LAI, FLC, Wca, Pcum, Icum_prev, Ica)

            self.state.CanopyState.Pcum[i, j] = Pcum
            self.state.CanopyState.Icum_prev[i, j] = Icum_prev
            self.state.CanopyState.Ica[i, j] = Ica
    
    def Canopy_Evaporation(self, precip, evap):
        def Canopy_Evap_cell(P, Ep, Ica, Wca):
            Wca = Wca + Ica
            Ecan = min(Ep, Wca)
            Pnet = P - Ecan
            Wca = Wca - Ecan

            return Pnet, Ecan, Wca
        
        for k in range(self.calorder.Length):
            i = self.calorder.SortRow[k]
            j = self.calorder.SortCol[k]

            P = precip[i, j]
            Ep = evap[i, j]                         # 在实际计算中这个表示通量，原始值为evap，这个变量含义应为剩余蒸散发量

            Ica = self.state.CanopyState.Ica[i, j]
            Wca = self.state.CanopyState.Wca[i, j]

            Pnet, Ecan, Wca = Canopy_Evap_cell(P, Ep, Ica, Wca)

            self.state.CanopyState.Pnet[i, j] = Pnet
            self.state.CanopyState.Wca[i, j] = Wca
            self.state.EvaporationState.Ep[i, j] = Ep - Ecan
            self.state.EvaporationState.Ecan[i, j] = Ecan
            
    def SoilLayer_Evaporation(self):
        def ThreeLayer_Evap_cell(Pnet, Ep, C, WU, WL, WD, WLM):
            EU = 0.0
            EL = 0.0
            ED = 0.0
            # 情况 1：无有效降水或降水不足 —— 蒸散受土壤水限制
            if Pnet <= 0.0:
                # ---------- 上层优先 ----------
                EU = min(Ep, WU)
                WU = WU - EU
                RemE = Ep - EU
                if RemE > 0.0:
                # ---------- 下层比例 ----------
                    if WLM > 0.0:
                        EL = RemE * (WL / WLM)
                    else:
                        EL = 0.0
                # ---------- 深层下限约束 ----------
                    if EL < C * RemE:
                        EL = C * RemE
                    if EL > WL:
                        EL = WL
                    WL = WL - EL
                    ED = RemE - EL
                    if ED > WD:
                        ED = WD
                    WD = WD - ED
                ET = EU + EL + ED
                Pe = 0.0
            # 情况 2：有有效降水 —— 蒸散不受限制
            else:
                ET = Ep
                Pe = Pnet - ET
                if Pe < 0.0:
                    ET = Pnet
                    Pe = 0.0
                EU = 0.0
                EL = 0.0
                ED = 0.0

            return Pe, EU, EL, ED, WU, WL, WD
        
        for k in range(self.calorder.Length):
            i = self.calorder.SortRow[k]
            j = self.calorder.SortCol[k]

            Pnet = self.state.CanopyState.Pnet[i, j]
            Ep = self.state.EvaporationState.Ep[i, j]
            C = self.lumparams.C
            WU = self.state.SoilState.WU[i, j]
            WL = self.state.SoilState.WL[i, j]
            WD = self.state.SoilState.WD[i, j]
            WLM = WU + WL

            Pe, EU, EL, ED, WU, WL, WD = ThreeLayer_Evap_cell(Pnet, Ep, C, WU, WL, WD, WLM)
            self.state.RunoffState.Pe[i, j]      = Pe

            self.state.EvaporationState.Eu[i, j] = EU
            self.state.EvaporationState.El[i, j] = EL
            self.state.EvaporationState.Ed[i, j] = ED
            self.state.SoilState.WU[i, j]        = WU
            self.state.SoilState.WL[i, j]        = WL
            self.state.SoilState.WD[i, j]       = WD
        
    def Runoff_ThreeSourceDivision(self):
        def Runoff_ThreeSourceDivision_cell(Pe, WS, WU, WL, WD, WSM, WUM, WLM, WDM, WMM, Ki, Kg):

            Rs = 0.0
            Ri = 0.0
            Rg = 0.0

            if (Pe <= 0.0):
                Rs = 0.0
                Ri = WS * Ki
                Rg = WS * Kg
                WS = WS * (1.0 - Ki - Kg)
                return Rs, Ri, Rg, WS, WU, WL, WD
            
            # 将降水划分成5mm一块
            if (Pe % 5.0 == 0.0):
                nd = int(Pe / 5.0)
                PPe = np.array([5.0] * nd)
            else:
                nd = int(Pe / 5.0) + 1
                PPe = np.array([5.0] * (nd - 1) + [Pe - 5.0 * (nd - 1)])

            KKiKg = (1.0 - (1.0 - (Kg + Ki))**(1.0 / nd)) / (Kg + Ki)
            KKi = Ki * KKiKg
            KKg = Kg * KKiKg

            for n in range(nd):
                PPe_n = PPe[n]
                # 情况 1：张力水库未满，产流受限制
                if PPe_n + WU + WL + WD < WMM:
                    if PPe_n + WU < WUM:
                    # ---------- 上层能容下 ----------
                        WU = WU + PPe_n
                    elif WL - WUM + WU + PPe_n < WLM:
                    # ---------- 上层容不下， 下层容下 ----------
                        WL = WL - WUM + WU + PPe_n
                        WU = WUM
                    else:
                    # ---------- 上层容不下，下层也容不下，深层容下 ----------
                        WD = WD + PPe_n - (WUM - WU) - (WLM - WL)
                        WU = WUM
                        WL = WLM
                    # 自由水库计算，未有入流
                    WS = WS
                    Rs = Rs
                    Ri = Ri + WS * KKi
                    Rg = Rg + WS * KKg
                    WS = WS * (1.0 - KKi - KKg)
                # 情况 2：张力水库满，产流不受限制
                else:
                    R = PPe_n + WU + WL + WD - WMM
                    WU = WUM
                    WL = WLM
                    WD = WDM

                    # 自由水库计算，入流为R
                    if R + WS <= WSM:
                    # ---------- 自由水库未满，无地表径流 ----------
                        WS = WS + R
                        Rs = Rs
                        Ri = Ri + WS * KKi
                        Rg = Rg + WS * KKg
                        WS = WS * (1.0 - KKi - KKg)
                    else:
                    # ---------- 自由水库满，有地表径流 ----------
                        Rs = Rs + R + WS - WSM
                        WS = WSM
                        Ri = Ri + WSM * KKi
                        Rg = Rg + WSM * KKg
                        WS = WSM * (1.0 - KKi - KKg)
            return Rs, Ri, Rg, WS, WU, WL, WD
        
        for k in range(self.calorder.Length):
            i = self.calorder.SortRow[k]
            j = self.calorder.SortCol[k]

            Pe = self.state.RunoffState.Pe[i, j]

            Ki = self.GridKi[i, j]
            Kg = self.GridKg[i, j]

            WS = self.state.SoilState.WS[i, j]
            WU = self.state.SoilState.WU[i, j]
            WL = self.state.SoilState.WL[i, j]
            WD = self.state.SoilState.WD[i, j]

            WSM = self.geostatic.FreedomWaterCapacity[i, j]
            WUM = self.GridWUM[i, j]
            WLM = self.GridWLM[i, j]
            WDM = self.GridWDM[i, j]
            WMM = WUM + WLM + WDM

            Rs, Ri, Rg, WS, WU, WL, WD = Runoff_ThreeSourceDivision_cell(Pe, WS, WU, WL, WD, WSM, WUM, WLM, WDM, WMM, Ki, Kg)

            self.state.RunoffState.Rs[i, j] = Rs
            self.state.RunoffState.Ri[i, j] = Ri
            self.state.RunoffState.Rg[i, j] = Rg

            self.state.SoilState.WS[i, j] = WS
            self.state.SoilState.WU[i, j] = WU
            self.state.SoilState.WL[i, j] = WL
            self.state.SoilState.WD[i, j] = WD

        self.state.RunoffState.R = self.state.RunoffState.Rs + self.state.RunoffState.Ri + self.state.RunoffState.Rg

    def DailyRouting(self):
        # 线性水库法，对出口点进行滞后演算法

        # ----------------仅用于记录状态，无计算应用------------------------
        self.state.RoutingState.Qs_out = np.zeros((self.Ny, self.Nx), dtype=float)
        self.state.RoutingState.Qi_out = np.zeros((self.Ny, self.Nx), dtype=float)
        self.state.RoutingState.Qg_out = np.zeros((self.Ny, self.Nx), dtype=float)
        # ----------------上游网格累计向下游网格传递水量------------------------
        self.state.RoutingState.Qs_in = np.zeros((self.Ny, self.Nx), dtype=float)
        self.state.RoutingState.Qi_in = np.zeros((self.Ny, self.Nx), dtype=float)
        self.state.RoutingState.Qg_in = np.zeros((self.Ny, self.Nx), dtype=float)
        self.state.RoutingState.Qch_in = np.zeros((self.Ny, self.Nx), dtype=float)
        # ----------------仅用于记录状态，无计算应用------------------------

        for k in range(self.calorder.Length):
            i = self.calorder.SortRow[k]
            j = self.calorder.SortCol[k]
            
            QRs = self.state.RunoffState.Rs[i, j] * self.CT
            QRi = self.state.RunoffState.Ri[i, j] * self.CT
            QRg = self.state.RunoffState.Rg[i, j] * self.CT

            Qs_out = self.state.RoutingState.WQs[i, j] * self.lumparams.CS  \
                            + QRs * (1 - self.lumparams.CS)  * (1 - self.geostatic.RunoffDistributionRatio[i, j]) # 当前格点非河道格点，不需要直接
            Qch_out = QRs * (1 - self.lumparams.CS)  * (1 - self.geostatic.RunoffDistributionRatio[i, j])             # Qch降水直接加入到河道出口点
            Qi_out = self.state.RoutingState.WQi[i, j] * self.lumparams.CI \
                            + QRi * (1 - self.lumparams.CI)
            Qg_out = self.state.RoutingState.WQg[i, j] * self.lumparams.CG \
                            + QRg * (1 - self.lumparams.CG)

            next_i, next_j = self._getNextGridXY(i, j)

            if next_i is not None and next_j is not None:           # 非流域出口点
                # ----------------更新当前格点状态------------------------
                self.state.RoutingState.WQs[i, j] = self.state.RoutingState.WQs[i, j] + QRs - Qs_out 
                self.state.RoutingState.WQi[i, j] = self.state.RoutingState.WQi[i, j] + QRi - Qi_out
                self.state.RoutingState.WQg[i, j] = self.state.RoutingState.WQg[i, j] + QRg - Qg_out
                # ----------------上游出流更新至下一格点状态---------------
                self.state.RoutingState.WQs[next_i, next_j] += Qs_out
                self.state.RoutingState.WQi[next_i, next_j] += Qi_out
                self.state.RoutingState.WQg[next_i, next_j] += Qg_out
                self.state.RoutingState.WQch[next_i, next_j] += Qch_out

            else:
                # ----------------更新当前格点状态------------------------
                self.state.RoutingState.WQs[i, j] = self.state.RoutingState.WQs[i, j] + QRs - Qs_out 
                self.state.RoutingState.WQi[i, j] = self.state.RoutingState.WQi[i, j] + QRi - Qi_out
                self.state.RoutingState.WQg[i, j] = self.state.RoutingState.WQg[i, j] + QRg - Qg_out
                # ----------------流域出口点，滞后演算法------------------------
                
                QLagTime_before = self.state.RoutingState.QLagTime[-1]  # 获取滞后时间队列中的最后一个值，即当前时刻的滞后水量

                QLagTime_current = QLagTime_before * self.lumparams.CCS + (Qs_out + Qi_out + Qg_out + Qch_out) * (1.0 - self.lumparams.CCS)
                self.state.RoutingState.QLagTime.pop(0)
                self.state.RoutingState.QLagTime.append(QLagTime_current)

                outflow = self.state.RoutingState.QLagTime[0]  # 滞后时间队列中的第一个值即为当前时刻的出流量

                return outflow

    def Routing(self):
        SimQ = self.DailyRouting()
        return SimQ

    def GXAJ_drv(self, time, precip, evap, iStep):

        print(f"Running time step {iStep + 1}/{len(self.timeseries)}: {self.timeseries[iStep].strftime('%Y-%m-%d')}")

        evap = evap * self.lumparams.K  # 蒸散发折算系数
        self.Canopy_Interception(time, precip)
        self.Canopy_Evaporation(precip, evap)

        self.SoilLayer_Evaporation()
        self.Runoff_ThreeSourceDivision()
        simQ = self.Routing()

        if self.config.save_restart:
            self._save_Restart(f"{self.config.outdir}/RESTART_{self.timeseries[iStep].strftime('%Y-%m-%d')}.nc")
        
        return simQ

    def run(self, precip_series, evap_series, Restart: Optional[xr.Dataset] = None):

                # initialize the model state
        if Restart is None:
            self._initialize_from_Zero()
        else:
            self._initialize_from_Restart(Restart)

        self.timeseries = pd.date_range(self.config.start_date, self.config.end_date, freq='D')
        SimQseries = np.full(len(self.timeseries), np.nan, dtype=float)

        for iStep in range(len(self.timeseries)):
            simQ = self.GXAJ_drv(self.timeseries[iStep], precip_series[iStep], evap_series[iStep], iStep)
            SimQseries[iStep] = simQ
            
        SimQresult = pd.DataFrame({"time": self.timeseries,"SimQ": SimQseries})
        return SimQresult
    

    
    def _getNextGridXY(self, i, j):
        # D8流向编码
        flow_dir = self.geostatic.GridFlowDirection[i, j]
        if flow_dir == 0:
            return i - 1, j + 1
        elif flow_dir == 1:
            return i, j + 1
        elif flow_dir == 2:
            return i + 1, j + 1
        elif flow_dir == 3:
            return i + 1, j
        elif flow_dir == 4:
            return i + 1, j - 1
        elif flow_dir == 5:
            return i, j - 1
        elif flow_dir == 6:
            return i - 1, j - 1
        elif flow_dir == 7:
            return i - 1, j
        else :  # flow_dir == 8 , 出口点
            return None, None





