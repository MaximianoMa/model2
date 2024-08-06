import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


def Year_simulation(D, T, T_week, t_peak, Load_max, load_pu, N_wt, WT_cap, wt_pu, T_wt, N_pv, PV_cap, pv_pu, T_pv,
                    N_Gas, Gas_max, Gas_min, N_ESS, N_ESS_week, Cap_ESS, P_ESS, P_ESS_week,
                    Cap_ESS_week, Ita, Ita_week, load_adj0, Op):
    Pur_grid = []
    P_abn = []
    Q_RES_all = 0
    Q_gas_all = 0
    Q_Pgrid_all = 0
    Q_Pbd_all = 0
    Q_Pbc_all = 0
    soc_week_Tra = np.zeros(D)

    Pwt_year = []
    Ppv_year = []
    Pgas_year = []
    Pabn_year = []
    Pgrid_year = []
    pun_abn_year = []
    Pbd_year = []
    Pbc_year = []
    soc_year = []
    delta_pgrid = []

    for m in range(D):
        # Load_week = Load_max * load_pu[:, m:m + 7].T
        # wt0_week = wt_pu[:, m:m + 7].T
        # pv0_week = pv_pu[:, m:m + 7].T
        #
        # Load_week = Load_week.reshape(168, 1)
        # wt0_week = wt0_week.reshape(168, 1)
        # pv0_week = pv0_week.reshape(168, 1)
        #
        # Q_Pgrid_week = 0
        # Q_gas_week = 0
        # Q_WT_week = 0
        # Q_PV_week = 0
        # Q_RES_week = 0
        # Q_Pbd_week = 0
        # Q_Pbc_week = 0
        #
        # P_all_week = [0] * T_week
        # Pabn0_week = [0] * T_week
        #
        # Pbc_week = cp.Variable((N_ESS_week, T_week))  # 电池充电策略
        # Pbd_week = cp.Variable((N_ESS_week, T_week))  # 电池放电策略
        # soc_week = cp.Variable((N_ESS_week, T_week + 1))  # 电池状态量
        # Temp_cha_week = cp.Variable((N_ESS_week, T_week), boolean=True)  # 电池充电时间
        # Temp_dis_week = cp.Variable((N_ESS_week, T_week), boolean=True)  # 电池放电时间
        #
        # Pgas_week = cp.Variable((N_Gas, T_week))  # 天然气的使用量
        # Pwt_week = cp.Variable((N_wt, T_week))  # 风能的使用量
        # Ppv_week = cp.Variable((N_pv, T_week))  # 太阳能的使用量
        # Pabn_week = cp.Variable((1, T_week))  # 抽水蓄能的使用量
        # Pgrid_week = cp.Variable((1, T_week))  # 电网的使用量
        #
        # Load_week.reshape(1, 168)
        # load_adj_week = np.array([-load_adj0[0] * Load_week, load_adj0[0] * Load_week])
        #
        # Constraints = []
        #
        # # 电池充电和放电约束
        # for i in range(N_ESS_week):
        #     Constraints.append(soc_week[i, 0] == 0.1 * Cap_ESS_week[i])  # SOC初始值约束
        #
        # for k in range(T_week):
        #     # 能量平衡约束
        #     Constraints.append(0 <= Pgrid_week[0, k])
        #     P_all_week[k] = Pgrid_week[0, k]
        #
        #     # 风能约束
        #     for i in range(N_wt):
        #         Constraints.append(Pwt_week[i, k] <= WT_cap * wt0_week[k])
        #         Constraints.append(0 <= Pwt_week[i, k])
        #         P_all_week[k] += Pwt_week[i, k]
        #         Pabn0_week[k] = WT_cap * wt0_week[k] - Pwt_week[i, k]
        #
        #     # 太阳能约束
        #     for i in range(N_pv):
        #         Constraints.append(Ppv_week[i, k] <= PV_cap * pv0_week[k])
        #         Constraints.append(0 <= Ppv_week[i, k])
        #         P_all_week[k] += Ppv_week[i, k]
        #         Pabn0_week[k] += PV_cap * pv0_week[k] - Ppv_week[i, k]
        #
        #     # 天然气约束
        #     for i in range(N_Gas):
        #         Constraints.append(Gas_min <= Pgas_week[i, k])
        #         Constraints.append(Pgas_week[i, k] <= Gas_max)
        #         P_all_week[k] += Pgas_week[i, k]
        #
        #     # 电池约束
        #     for i in range(N_ESS_week):
        #         P_all_week[k] += Pbd_week[i, k] - Pbc_week[i, k]
        #         Constraints.append(Pbc_week[i, k] <= Temp_cha_week[i, k] * P_ESS_week[i])
        #         Constraints.append(0 <= Pbc_week[i, k])
        #         Constraints.append(0 <= Pbd_week[i, k])
        #         Constraints.append(Pbd_week[i, k] <= Temp_dis_week[i, k] * P_ESS_week[i])
        #         Constraints.append(
        #             soc_week[i, k + 1] == soc_week[i, k] + Ita_week * Pbc_week[i, k] - Pbd_week[i, k] / Ita_week)
        #         Constraints.append(0 <= soc_week[i, k])
        #         Constraints.append(soc_week[i, k] <= 1 * Cap_ESS_week[i])
        #         Constraints.append(0 <= soc_week[i, k + 1])
        #         Constraints.append(soc_week[i, k + 1] <= 1 * Cap_ESS_week[i])
        #         Constraints.append(
        #             Pabn_week[0, k] == Pabn0_week[k] + (1 - Ita_week) * Pbc_week[i, k] + (1 - Ita_week) * Pbd_week[
        #                 i, k] / Ita_week)
        #
        #     # 电池充电和放电时间约束
        #     for i in range(N_ESS_week):
        #         for j in range(N_ESS_week):
        #             Constraints.append(0 <= Temp_cha_week[i, k] + Temp_dis_week[j, k])
        #             Constraints.append(Temp_cha_week[i, k] + Temp_dis_week[j, k] <= 1)
        #
        #     # SOC约束
        #     if k % T == 0:
        #         Constraints.append(soc_week[1, k] == soc_week[1, 0])
        #
        #     # 能量平衡约束
        #     Constraints.append(P_all_week[k] >= Load_week[k] + load_adj_week[0, k])
        #     Constraints.append(P_all_week[k] <= Load_week[k] + load_adj_week[1, k])
        #
        #     for i in range(N_ESS_week):
        #         Constraints.append(soc_week[i, k] == soc_week[i, 0])
        #
        # Cgrid_week = cp.sum(Pgrid_week)
        #
        # pun_abn_week = cp.sum(Pabn_week)
        #
        # C_adj = cp.sum(cp.abs(load_adj_week[0, :]))
        #
        # flu = cp.max(Pgrid_week) - cp.min(Pgrid_week)
        #
        # C_week = (Op[0] * Cgrid_week + Op[1] * pun_abn_week + Op[3] * (sum(sum(Pbd_week)) + sum(sum(Pbc_week))) + Op[
        #     2] * sum(sum(Pgas_week)) + Op[5] * C_adj + 0 * flu)
        #
        # # 设置优化器
        # prob = cp.Problem(cp.Minimize(C_week), constraints=Constraints)
        #
        # prob.solve(solver=cp.CPLEX)
        #
        # # 获取决策变量的值
        # Pwt_week_value = Ppv_week.value
        # Ppv_week_value = Ppv_week.value
        # Pgas_week_value = Pgas_week.value
        # Pabn_week_value = Pabn_week.value
        # Pgrid_week_value = Pgrid_week.value
        # pun_abn_week_value = pun_abn_week.value
        # Pbd_week_value = Pbd_week.value
        # Pbc_week_value = Pbc_week.value
        # soc_week_value = soc_week.value
        #
        # # 计算总能量
        # Q_Pgrid_week = np.sum(Pgrid_week_value)
        # Q_gas_week = np.sum(Pgas_week_value)
        # Q_WT_week = np.sum(Pwt_week_value)
        # Q_PV_week = np.sum(Ppv_week_value)
        # Q_RES_week = Q_WT_week + Q_PV_week
        # Q_Pbd_week = np.sum(Pbd_week_value)
        # Q_Pbc_week = np.sum(Pbc_week_value)
        #
        # soc_week_Tra[m] = soc_week_value[0, 24]
        #
        # PP = np.vstack([Pwt_week_value, Ppv_week_value, Pbd_week_value[0, :], Pbd_week_value[1, :], Pgrid_week_value,
        #                 Pgas_week_value])
        # PPC = np.vstack([Pabn_week_value, Pbc_week_value[0, :], Pbc_week_value[1, :]])
        #
        # # 绘图
        # fig, ax = plt.subplots()
        # x = np.arange(T_week)
        #
        # # 画正条形图
        # for i in range(PP.shape[0]):
        #     ax.bar(x, PP[i, :], label=f'PP_{i}', bottom=np.sum(PP[:i, :], axis=0))
        #
        # # 画负条形图
        # for i in range(PPC.shape[0]):
        #     ax.bar(x, PPC[i, :], label=f'PPC_{i}', bottom=np.sum(PPC[:i, :], axis=0))
        #
        # # 绘制负载曲线
        # ax.plot(np.arange(T_week), Load_week, 'r', linewidth=2)
        #
        # Ca = np.array([[77, 190, 238],
        #                [46, 139, 87],
        #                [255, 165, 0],
        #                [220, 20, 60],
        #                [210, 105, 30],
        #                [126, 47, 142]]) / 255.0
        #
        # Cb = np.array([[0, 114, 189],
        #                [255, 165, 0],
        #                [220, 20, 60]]) / 255.0
        #
        # for i, bar in enumerate(ax.patches[:len(Ca)]):
        #     bar.set_facecolor(Ca[i % len(Ca)])
        #
        # for i, bar in enumerate(ax.patches[len(Ca):len(Ca) + len(Cb)]):
        #     bar.set_facecolor(Cb[i % len(Cb)])
        # # 设置标题和标签
        # ax.set_title('能源分配图')
        # ax.set_xlabel('小时')
        # ax.set_ylabel('功率 (MW)')
        #
        # # 设置图例
        # ax.legend(['负载', 'Pwt', 'Ppv', 'Pbd_week1', 'Pbd_week2', 'Pgrid', 'Pgas', 'Pabn', 'Pbc_week1', 'Pbc_week2'],
        #           loc='upper right')
        #
        # # 设置网格、颜色和刻度
        # ax.set_facecolor('white')
        # ax.grid(False)
        # ax.tick_params(axis='x', colors=[0.1, 0.1, 0.1], direction='out', length=6, width=1)
        # ax.tick_params(axis='y', colors=[0.1, 0.1, 0.1], direction='out', length=6, width=1)
        # ax.set_xticks(np.arange(0, T_week + 1, 2))
        # ax.set_xlim([0, T_week + 1])
        # ax.set_yticks(np.arange(-4000, 9000, 1000))
        # ax.set_ylim([-4000, 8000])
        #
        # # 调整图形位置和大小
        # fig.set_size_inches(15, 8)
        # fig.tight_layout()
        #
        # plt.show()

        print("--------------------------Entering Day simulator---------------------------------")

        # Day simulator
        Load = Load_max * load_pu[:, m]
        wt0 = wt_pu[:, m]
        pv0 = pv_pu[:, m]
        load_adj = np.array([-load_adj0[0] * Load, load_adj0[0] * Load])


        P_all = [0] * T
        Pabn0 = [0] * T

        # Pbc = np.zeros((N_ESS, T))
        # Pbd = np.zeros((N_ESS, T))
        # soc = np.zeros((N_ESS, T + 1))
        # Temp_cha = np.ones((N_ESS, T))
        # Temp_dis = np.ones((N_ESS, T))

        # Pgas = np.zeros((N_Gas, T))
        # Pwt = np.zeros((N_wt, T))
        # Ppv = np.zeros((N_pv, T))
        Pabn = [cp.Variable() for _ in range(T)]
        # Pgrid = np.zeros(T)

        Pbc = cp.Variable((N_ESS, T))  # battery bc
        Pbd = cp.Variable((N_ESS, T))  # battery bd
        soc = cp.Variable((N_ESS, T + 1))  # state of charge
        Temp_cha = cp.Variable((N_ESS, T), boolean=True)  # battery charge
        Temp_dis = cp.Variable((N_ESS, T), boolean=True)  # battery discharge

        Pgas = cp.Variable((N_Gas, T))  # gas
        Pwt = cp.Variable((N_wt, T))  # wind turbine
        Ppv = cp.Variable((N_pv, T))  # photovoltaic
        # Pabn = cp.Variable(T)  # abandonment
        Pgrid = cp.Variable(T)  # grid usage

        Constraints = []

        for i in range(N_ESS):
            Constraints.append(soc[i, 0] == 0.1 * Cap_ESS[i])  # SOC初始值约束

        for k in range(T):
            Constraints.append(Pgrid[k] >= 0)
            P_all[k] = Pgrid[k]

            # N_wt 只有0/1
            for i in range(N_wt):
                Constraints.append(Pwt[i, k] <= WT_cap * wt0[k])
                Constraints.append(Pwt[i, k] >= 0)
                P_all[k] += Pwt[i, k]
                Pabn0[k] = WT_cap * wt0[k] - Pwt[i, k]

            for i in range(N_pv):
                Constraints.append(Ppv[i, k] <= PV_cap * pv0[k])
                Constraints.append(0 <= Ppv[i, k])
                P_all[k] += Ppv[i, k]
                Pabn0[k] += PV_cap * pv0[k] - Ppv[i, k]

            # 天然气约束
            for i in range(N_Gas):
                Constraints.append(Gas_min <= Pgas[i, k])
                Constraints.append(Pgas[i, k] <= Gas_max)
                P_all[k] += Pgas[i, k]

            # 电池约束
            for i in range(N_ESS):
                P_all[k] += Pbd[i, k] - Pbc[i, k]
                Constraints.append(Pbc[i, k] <= Temp_cha[i, k] * P_ESS[i])
                Constraints.append(0 <= Pbc[i, k])
                Constraints.append(0 <= Pbd[i, k])
                Constraints.append(Pbd[i, k] <= Temp_dis[i, k] * P_ESS[i])
                Constraints.append(soc[i, k + 1] == soc[i, k] + Ita * Pbc[i, k] - Pbd[i, k] / Ita)
                Constraints.append(0 <= soc[i, k])
                Constraints.append(soc[i, k] <= Cap_ESS[i])
                Constraints.append(0 <= soc[i, k + 1])
                Constraints.append(soc[i, k + 1] <= Cap_ESS[i])

                #Todo
                Pabn[k] = Pabn0[k] + (1 - Ita) * Pbc[i, k] + (1 - Ita) * Pbd[i, k] / Ita


            # 电池充电和放电时间约束
            for i in range(N_ESS):
                for j in range(N_ESS):
                    Constraints.append(0 <= Temp_cha[i, k] + Temp_dis[j, k])
                    Constraints.append(Temp_cha[i, k] + Temp_dis[j, k] <= 1)

            if k == t_peak - 1:
                Constraints.append(soc[1, k] == Cap_ESS[1])

            # 能量平衡约束
            Constraints.append(P_all[k] >= Load[k] + load_adj[0, k])
            Constraints.append(P_all[k] <= Load[k] + load_adj[1, k])

        Constraints.append(soc[1, T] == soc[1, 0])
        Constraints.append(soc[0, T] == soc_week_Tra[m])

        # convert np 2 cp
        # Pgrid = cp.Variable(shape=Pgrid.shape, value=Pgrid)
        # Pabn = cp.Variable(shape=Pabn.shape, value=Pabn)
        # Pgas = cp.Variable(shape=Pgas.shape, value=Pgas)
        # Pbd = cp.Variable(shape=Pbd.shape, value=Pbd)
        # Pbc = cp.Variable(shape=Pbc.shape, value=Pbc)


        # 目标函数
        Cgrid_total = cp.sum(Pgrid)
        pun_abn = cp.sum(Pabn)
        C_adj = cp.sum(cp.abs(load_adj[0, :]))
        flu = cp.max(Pgrid) - cp.min(Pgrid)


        C_year = Op[0] * Cgrid_total + Op[1] * pun_abn + Op[2] * cp.sum(cp.sum(Pbc)) + Op[2] * cp.sum(cp.sum(Pbd)) + Op[3] * cp.sum(
            cp.sum(Pgas)) + Op[4] * C_adj + Op[5] * flu



        # 设置优化器
        prob = cp.Problem(cp.Minimize(C_year), constraints=Constraints)
        result = prob.solve(solver=cp.CPLEX, verbose=True)

        if result is None or result == float('inf') or result == float('-inf'):
            print(f"Optimization failed for day {m}")
            continue

        # 获取决策变量的值
        Pwt_value = Pwt.value
        Ppv_value = Ppv.value
        Pgas_value = Pgas.value
        Pabn_value = np.array([Pabn[k].value for k in range(T)])
        Pgrid_value = Pgrid.value
        pun_abn_value = pun_abn.value
        Pbd_value = Pbd.value
        Pbc_value = Pbc.value
        soc_value = soc.value



        # 计算总能量
        Q_Pgrid = np.sum(Pgrid_value)
        Q_gas = np.sum(Pgas_value)
        Q_WT = np.sum(Pwt_value)
        Q_PV = np.sum(Ppv_value)
        Q_RES = Q_WT + Q_PV
        Q_Pbd = np.sum(Pbd_value)
        Q_Pbc = np.sum(Pbc_value)

        soc_week_Tra[m] = soc_value[0, 24]

        # 添加到年数据中
        Pwt_year.append(Pwt_value)
        Ppv_year.append(Ppv_value)
        Pgas_year.append(Pgas_value)
        Pabn_year.append(Pabn_value)
        Pgrid_year.append(Pgrid_value)
        pun_abn_year.append(pun_abn_value)
        Pbd_year.append(Pbd_value)
        Pbc_year.append(Pbc_value)
        soc_year.append(soc_value)
        delta_pgrid.append(np.max(Pgrid_value) - np.min(Pgrid_value))

        x = np.arange(1, T + 1)
        PP = np.vstack([Pwt_value, Ppv_value, Pbd_value[0, :], Pbd_value[1, :], Pgrid_value, Pgas_value])
        PPC = np.vstack([-Pabn_value, -Pbc_value[0, :], -Pbc_value[1, :]])

        fig, ax = plt.subplots()

        # Plot positive bars
        bottom = np.zeros(T)
        for i in range(PP.shape[0]):
            ax.bar(x, PP[i, :], label=f'PP_{i}', bottom=bottom)
            bottom += PP[i, :]

        # Plot negative bars
        bottom = np.zeros(T)
        for i in range(PPC.shape[0]):
            ax.bar(x, PPC[i, :], label=f'PPC_{i}', bottom=bottom)
            bottom += PPC[i, :]

        # Plot Load line
        ax.plot(x, Load, 'r', linewidth=2, label='Load')

        # Colors for bars
        Ca = np.array([[77, 190, 238],
                       [46, 139, 87],
                       [255, 165, 0],
                       [220, 20, 60],
                       [210, 105, 30],
                       [126, 47, 142]]) / 255.0

        Cb = np.array([[0, 114, 189],
                       [255, 165, 0],
                       [220, 20, 60]]) / 255.0

        for i, bar in enumerate(ax.patches[:len(Ca)]):
            bar.set_facecolor(Ca[i % len(Ca)])

        for i, bar in enumerate(ax.patches[len(Ca):len(Ca) + len(Cb)]):
            bar.set_facecolor(Cb[i % len(Cb)])

        # Set titles and labels
        ax.set_title('能源分配图', fontsize=16, fontweight='bold')
        ax.set_xlabel('小时', fontsize=16)
        ax.set_ylabel('功率 (MW)', fontsize=16)

        # Set legend
        ax.legend(['Load', 'Pwt', 'Ppv', 'Pbd_week1', 'Pbd_week2', 'Pgrid', 'Pgas', 'Pabn', 'Pbc_week1', 'Pbc_week2'],
                  loc='upper right', fontsize=12, frameon=False)

        # Set grid, colors, and ticks
        ax.set_facecolor('white')
        ax.grid(False)
        ax.tick_params(axis='x', colors=[0.1, 0.1, 0.1], direction='out', length=6, width=1)
        ax.tick_params(axis='y', colors=[0.1, 0.1, 0.1], direction='out', length=6, width=1)
        ax.set_xticks(np.arange(0, T + 1, 2))
        ax.set_xlim([0, T + 1])
        ax.set_yticks(np.arange(-4000, 9000, 1000))
        ax.set_ylim([-4000, 8000])

        # Adjust figure size and layout
        fig.set_size_inches(15, 8)
        fig.tight_layout()

        plt.show()

    return Pwt_year, Ppv_year, Pgas_year, Pabn_year, Pgrid_year, pun_abn_year, Pbd_year, Pbc_year, soc_year, delta_pgrid
