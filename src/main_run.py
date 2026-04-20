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

from core.DailyModel import GridXAJDailyModel
from core.schemas import SimulationConfig, GeoStatic, LumParams, CalSort

from utils.draw import plot_2d, plot_line
from utils.objfun import FastObjFun

geostatic_ds = xr.open_dataset("../test/geo_static.nc")
lumparams = pd.read_csv("../test/Lumpara_Sub.tbl", skipinitialspace=True)
calorder = pd.read_csv("../test/CalSort_01.txt", sep="\s+")

config = SimulationConfig(
    start_date = pd.to_datetime("2011-01-01"),
    end_date   = pd.to_datetime("2011-12-31"),

    outdir = "../test/output",

    FreeWaterCoeff = 0.1,
    TensionWaterCoeff = 0.5,
    ini_sm        = 10,

    save_restart = 1
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
lumparams = LumParams(
    OC  = 1.8,
    ROC = 1.0,
    K   = 0.8,
    C   = 0.17,
    LUM = 0.1,          # WM，上层占的比例
    LLM = 0.5,          # WM，下层占的比例
    CG  = 0.8,
    CI  = 0.8,
    CS  = 0.995,
    CCS = 0.01,
    LT  = 0,
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
Restart = xr.open_dataset(r"E:\Learning\GXAJ_fromVB\test\Restart\RESTART_2011-08-09.nc")
model = GridXAJDailyModel(config, geostatic, lumparams, calorder)

forcing = xr.open_dataset("../test/GXAJ_Forcing_2011.nc")
precip = forcing["precip"].values
evap = forcing["evap"].values

simQ = model.run(precip, evap, Restart=Restart)

Qobs = pd.read_csv("../test/Qobs/Tunxi_2011.csv")
simQ["Qobs"] = Qobs["Q"].values
simQ["P10"]  = np.average(precip, axis=(1, 2)) * 10
plot_line(simQ, title="Simulated Streamflow", save_path="../test/output/SimulatedQ.png")
metrics_df = FastObjFun(simQ, simQ)
print(metrics_df)