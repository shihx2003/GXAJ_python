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
from numba import njit

from src.schemas import GeoStatic, LumParams
from src.TBL import SOIL_SWC, SOIL_FC, SOIL_WP, MON_LAI, MAX_LAI, CTOPH

from utils.draw import plot_2d

class DailySEM3LayersModel:
    
    PI = 3.141592654
    EARTH_RADIUS_KM = 6370.997

    DIV = 5.0
    DT = 24.0
    ETKCB_MIN = (0.15 + 0.2) / 2.0

    def __init__(self, geostatic: GeoStatic, lumparams: LumParams) -> None:
        self.geostatic = geostatic
        self.lumparams = lumparams
        self.Nx = geostatic.Basin.shape[1]
        self.Ny = geostatic.Basin.shape[0]

        self.GridWUM, self.GridWLM, self.GridWDM = self._Estimate_WM()
        self.GridKi, self.GridKg, self.SumKgKi   = self._Estimate_KiKg()
        self.GridFLC                             = self._Estimate_FLC()


    def _GridArea(self):
        GridArea = np.ones((self.Ny, self.Nx), dtype=float)
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
    

    def Read_Forcing(self):
        pass
    def Canopy_Interception():
        pass
    def Canopy_Evaporation():
        pass
    def SoilLayer_Evaporation():
        pass
    def Runoff_ThreeSourceDivision():
        pass
    def Routing():
        pass
        
    def GXAJ_drv(self,):
        self.Read_Forcing()

    
    def run(self, state):
        for i in range(self.istep):
            self.GXAJ_drv()




