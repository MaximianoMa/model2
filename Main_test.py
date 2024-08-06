import pandas as pd
import numpy as np
from Year_simulation import Year_simulation

# Clearing workspace is not necessary in Python as it is in MATLAB

# Input Data
D = 365
T = 24
T_week = 168

# Load
days_of_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

# 初始化一个空列表来存储扩展的数据
expanded_data = []
load_0 = pd.read_excel('若羌~四川直流-p_rate-ALS.xlsx', header=None).to_numpy()
# 对每个月份的数据进行扩展
for i in range(12):
    for day in range(days_of_month[i]):
        expanded_data.append(load_0[:, i])

# 将扩展后的数据转换成 NumPy 数组，并重塑为 24 行 365 列
load_pu = np.array(expanded_data).T

# 确保最终的形状是 (24, 365)
load_pu = load_pu.reshape((24, 365))


Load_max = 6000  # MW

# WT
N_wt = 1
WT_cap = 2099
wt_pu = pd.read_excel("若羌风电-imax_rate-ALS.xlsx", header=None).to_numpy()
wt_pu = wt_pu.reshape(365, 24).T
T_wt = wt_pu.sum(axis=0)

# PV
N_pv = 1
PV_cap = 9530
pv_pu = pd.read_excel("若羌光伏-imax_rate-ALS.xlsx", header=None).to_numpy()
pv_pu = pv_pu.reshape(365, 24).T
T_pv = pv_pu.sum(axis=0)

# Gas
N_Gas = 1
Gas_cap = 3300  # MW
cost_kWh_Gas = 0.204  # ￥/MWh
gas_max_0 = 1
gas_min_0 = 0
Gas_max = gas_max_0 * Gas_cap
Gas_min = gas_min_0 * Gas_cap

# ESS
N_ESS0 = 1
N_ESS_week0 = 1
N_ESS = N_ESS_week0 + N_ESS0
N_ESS_week = N_ESS_week0 + N_ESS0

P_ESS0 = 600  # MW
Cap_ESS0 = 1200  # MWh
P_ESS_week0 = 100
Cap_ESS_week0 = 800
P_ESS = np.array([P_ESS_week0, P_ESS0])
Cap_ESS = np.array([Cap_ESS_week0, Cap_ESS0])

P_ESS_week = np.array([P_ESS_week0, P_ESS0])
Cap_ESS_week = np.array([Cap_ESS_week0, Cap_ESS0])

Ita = 0.9
Ita_week = 0.9

t_peak = 17  # Peak demand hour in the day

# load_adj
load_adj = np.array([0, 0])

# Objective coefficients
Op = np.array([5, 0.3, 0.15, 0.204, 0.5, 0.1])

# Call the Year_simulation function
# This function needs to be translated and defined in Python


results = Year_simulation(D, T, T_week, t_peak, Load_max, load_pu, N_wt, WT_cap, wt_pu, T_wt, N_pv, PV_cap, pv_pu, T_pv,
                          N_Gas, Gas_max, Gas_min, N_ESS, N_ESS_week, Cap_ESS, P_ESS, P_ESS_week,
                          Cap_ESS_week, Ita, Ita_week, load_adj, Op)


# Unpack results
# (Rate_abn_all, Rate_QRES, Rate_Qgrid, Pwt_year, Ppv_year, Pgas_year, Pabn_year, Pgrid_year,
#  pun_abn_year, Pbd_year, Pbc_year, soc_year, delta_pgrid) = results