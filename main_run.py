# -*- encoding: utf-8 -*-
'''
@File    :   main_run.py
@Create  :   2026-04-13 18:55:18
@Author  :   shihx2003
@Version :   1.0
@Contact :   shihx2003@outlook.com
'''

# here put the import lib
import xarray as xr
import pandas as pd
import numpy as np

from src.model import DailySEM3LayersModel
from src.schemas import SimulationConfig, GeoStatic, LumParams, CalSort

from utils.draw import plot_2d

geostatic_ds = xr.open_dataset("./test/geo_static.nc")
lumparams = pd.read_csv("./test/Lumpara_Sub.tbl", skipinitialspace=True)
calorder = pd.read_csv("./test/CalSort_01.txt", sep="\s+")

config = SimulationConfig(
    start_date = pd.to_datetime("2011-01-01"),
    end_date   = pd.to_datetime("2011-12-31"),

    FreeWaterCoeff = 1.5,
    TensionWaterCoeff = 1.5,
    ini_sm        = 1
)

calorder = CalSort(
    Length = len(calorder),
    CalOrder = calorder["No.CalOrder"].values,
    SortRow  = calorder["SortRow"].values - 1,      # 原来的计算顺序索引是从1开始的，py 1转换为从0开始的索引
    SortCol  = calorder["SortCol"].values - 1       # 原来的计算顺序索引是从1开始的，py 1转换为从0开始的索引
)

geostatic = GeoStatic(
    Basin                   = geostatic_ds["Basin"].values,

    Longitude               = geostatic_ds["lon"].values,
    Latitude                = geostatic_ds["lat"].values,

    DEM                     = geostatic_ds["DEM"].values,
    RiverChannel            = geostatic_ds["RiverChannel"].values,
    TensionWaterCapacity    = geostatic_ds["TensionWaterCapacity"].values,
    FreedomWaterCapacity    = geostatic_ds["FreedomWaterCapacity"].values,
    Top30cmSoilType         = geostatic_ds["SoilType_0_30"].values,
    Deep30to100cmSoilType   = geostatic_ds["SoilType_30_100"].values,
    HumusSoilDepth          = geostatic_ds["HumusSoilDepth"].values,
    VegetationType          = np.nan_to_num(geostatic_ds["VegetationType"].values, nan=10).astype(int),
    RunoffDistributionRatio = geostatic_ds["RunoffDistributionRatio"].values,
    GridFlowDirection       = geostatic_ds["GridFlowDirection"].values,
    VadoseZoneDepth         = geostatic_ds["VadoseZoneDepth"].values,
    FlowAccumulationArea    = geostatic_ds["FlowAccumulationArea"].values,
)
plot_2d(geostatic.TensionWaterCapacity, title="Tension Water Capacity", save_path="./test/output/TensionWaterCapacity.png")
lumparams = LumParams(
    OC  = lumparams["OC"].values,
    ROC = lumparams["ROC"].values,
    K   = lumparams["K"].values,
    C   = lumparams["C"].values,
    LUM = lumparams["LUM"].values,
    LLM = lumparams["LLM"].values,
    CG  = lumparams["CG"].values,
    CI  = lumparams["CI"].values,
    CS  = lumparams["CS"].values,
    LT  = lumparams["LT"].values,
    CHM = lumparams["CHM"].values,
    LTH = lumparams["LTH"].values,
    Kech = lumparams["Kech"].values,
    Kes = lumparams["Kes"].values,
    Kei = lumparams["Kei"].values,
    Keg = lumparams["Keg"].values,
    Xech = lumparams["Xech"].values,
    Xes = lumparams["Xes"].values,
    Xei = lumparams["Xei"].values,
    Xeg = lumparams["Xeg"].values,
    Ki_tmp = lumparams["Ki_tmp"].values
)



model = DailySEM3LayersModel(config, geostatic, lumparams, calorder)

forcing = xr.open_dataset("./test/GXAJ_Forcing_2011.nc")
precip = forcing["precip"].values
evap = forcing["evap"].values
model.run(precip, evap)
