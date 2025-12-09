import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)
from pyomo.environ import (
    ConcreteModel, Set, Param, Var, NonNegativeReals, Reals,
    Constraint, Objective, minimize, SolverFactory
)

# =========================
# 1.  read data
# =========================

df = pd.read_csv("Hourly Profiles.csv")
df.columns = [c.strip() for c in df.columns]

time_index = df.index.to_list()
T = len(time_index)

wind_profile = df["Normalised Renewable Production Profile"].values
load_profile = df["Normalised Electricity Consumption Profile"].values

P_load_peak = 100.0   # MW, load under normal conditions
P_wind_cap = 150.0    # MW, rated power capacity of wind farm
P_therm_cap = 250.0   # MW, rated power capacity of thermal power plant
P_batt_power = 100.0  # MW, rated power capacity of the battery
E_batt_cap = P_batt_power * 24.0  # MWh, battery energy capacity

P_line_cap = 100.0  # MW, line capacity

C_therm = 50.0      # £/MWh, marginal cost of thermal power plant
VOLL = 5000.0       # £/MWh, assumed cost of load shedding
C_curt = 100.0       # £/MWh, assumed cost of wind power curtailment

eta_c = 1.0         # battery charging efficiency
eta_d = 1.0         # battery discharging efficiency

SoC_init = 0.5 * E_batt_cap # assumed initial state of charge of battery

week2_start_hour = 7 * 24        # Dunkelflaute event start time
week2_end_hour = 14 * 24 - 1     # Dunkelflaute event end time
event_hours = np.zeros(T, dtype=bool)
event_hours[week2_start_hour:week2_end_hour + 1] = True

# load and wind power under normal conditions on the time interval of 1 hour
P_load_normal = P_load_peak * load_profile
P_wind_avail_normal = P_wind_cap * wind_profile

# load and wind power under Dunkelflaute event
P_load_event = P_load_normal.copy()
P_wind_avail_event = P_wind_avail_normal.copy()
P_load_event[event_hours] *= 1.4
P_wind_avail_event[event_hours] *= 0.4


# =========================
# 2. modeling
# =========================

def build_and_solve_scenario(
    P_load, P_wind_avail,
    outage_mask=None, # outage of thermal power plant, 0 is work, 1 is outage
    solver_name="highs",
    verbose=False
):

    if outage_mask is None:
        outage_mask = np.zeros(T, dtype=bool)

    m = ConcreteModel()

    # set
    m.T = Set(initialize=time_index, ordered=True)

    # para
    m.P_load = Param(m.T, initialize=lambda m, t: float(P_load[t]))
    m.P_wind_avail = Param(m.T, initialize=lambda m, t: float(P_wind_avail[t]))
    m.outage = Param(m.T, initialize=lambda m, t: int(outage_mask[t]))  # 0 or 1

    # variable
    m.P_therm = Var(m.T, within=NonNegativeReals)   # thermal power plant output
    m.P_wind_inj = Var(m.T, within=NonNegativeReals)    # the injection to grid of the wind farm
    m.P_wind_curt = Var(m.T, within=NonNegativeReals)   # the curtailment of wind farm
    m.P_ch = Var(m.T, within=NonNegativeReals)      # charging power of battery
    m.P_dis = Var(m.T, within=NonNegativeReals)     # discharging power of battery
    m.SoC = Var(m.T, within=NonNegativeReals)       # stage of charge of battery
    m.P_shed = Var(m.T, within=NonNegativeReals)    # load shedding
    m.f12 = Var(m.T, within=Reals)                  # power flow of 1-2 line
    m.f13 = Var(m.T, within=Reals)                  # power flow of 1-3 line
    m.f23 = Var(m.T, within=Reals)                  # power flow of 2-3 line

    # constraints of thermal power plant output
    def thermal_limit_rule(m, t):
        return m.P_therm[t] <= P_therm_cap * (1 - m.outage[t])
    m.thermal_limit = Constraint(m.T, rule=thermal_limit_rule)

    # constraints of wind farm injection
    def wind_balance_rule(m, t):
        return m.P_wind_inj[t] + m.P_wind_curt[t] == m.P_wind_avail[t]
    m.wind_balance = Constraint(m.T, rule=wind_balance_rule)

    # constraints of battery output and energy limit
    def batt_power_ch_rule(m, t):
        return m.P_ch[t] <= P_batt_power
    m.batt_power_ch = Constraint(m.T, rule=batt_power_ch_rule)

    def batt_power_dis_rule(m, t):
        return m.P_dis[t] <= P_batt_power
    m.batt_power_dis = Constraint(m.T, rule=batt_power_dis_rule)

    def batt_soc_cap_rule(m, t):
        return m.SoC[t] <= E_batt_cap
    m.batt_soc_cap = Constraint(m.T, rule=batt_soc_cap_rule)

    # constraints of battery operation
    time_list = list(m.T)

    def batt_soc_dyn_rule(m, idx):
        t_idx = time_list.index(idx)
        if t_idx == 0:
            return m.SoC[idx] == SoC_init
        else:
            t_prev = time_list[t_idx - 1]
            return m.SoC[idx] == m.SoC[t_prev] + eta_c * m.P_ch[t_prev] - (1.0 / eta_d) * m.P_dis[t_prev]
    m.batt_soc_dyn = Constraint(m.T, rule=batt_soc_dyn_rule)

    def batt_cycle_rule(m):
        t_last = time_list[-1]
        return m.SoC[t_last] + eta_c * m.P_ch[t_last] - (1.0 / eta_d) * m.P_dis[t_last] == SoC_init
    m.batt_cycle = Constraint(rule=batt_cycle_rule)

    # constraints of line capacity
    def line12_rule(m, t):
        return (-P_line_cap, m.f12[t], P_line_cap)
    def line13_rule(m, t):
        return (-P_line_cap, m.f13[t], P_line_cap)
    def line23_rule(m, t):
        return (-P_line_cap, m.f23[t], P_line_cap)

    m.line12 = Constraint(m.T, rule=line12_rule)
    m.line13 = Constraint(m.T, rule=line13_rule)
    m.line23 = Constraint(m.T, rule=line23_rule)

    # constraints of nodal power balance, based on the transport model
    def bus1_balance_rule(m, t):
        return m.P_therm[t] - m.f12[t] - m.f13[t] == 0.0
    m.bus1_balance = Constraint(m.T, rule=bus1_balance_rule)

    def bus2_balance_rule(m, t):
        return m.P_wind_inj[t] + m.P_dis[t] - m.P_ch[t] + m.f12[t] - m.f23[t] == 0.0
    m.bus2_balance = Constraint(m.T, rule=bus2_balance_rule)

    def bus3_balance_rule(m, t):
        return -m.P_load[t] + m.P_shed[t] + m.f13[t] + m.f23[t] == 0.0
    m.bus3_balance = Constraint(m.T, rule=bus3_balance_rule)

    # constraints of load shedding
    def shed_limit_rule(m, t):
        return m.P_shed[t] <= m.P_load[t]
    m.shed_limit = Constraint(m.T, rule=shed_limit_rule)

    # obj = cost of thermal power plant + cost of load shedding + cost of wind curtailment
    def objective_rule(m):
        return sum(
            C_therm * m.P_therm[t]
            + VOLL * m.P_shed[t]
            + C_curt * m.P_wind_curt[t]
            for t in m.T
        )
    m.obj = Objective(rule=objective_rule, sense=minimize)

    # solve
    solver = SolverFactory(solver_name)
    results = solver.solve(m, tee=verbose)

    # read results
    P_therm_sol = np.array([m.P_therm[t]() for t in m.T])
    P_wind_inj_sol = np.array([m.P_wind_inj[t]() for t in m.T])
    P_wind_curt_sol = np.array([m.P_wind_curt[t]() for t in m.T])
    P_ch_sol = np.array([m.P_ch[t]() for t in m.T])
    P_dis_sol = np.array([m.P_dis[t]() for t in m.T])
    P_shed_sol = np.array([m.P_shed[t]() for t in m.T])
    SoC_sol = np.array([m.SoC[t]() for t in m.T])

    total_cost = sum(
        C_therm * P_therm_sol
        + VOLL * P_shed_sol
        + C_curt * P_wind_curt_sol
    )
    thermal_cost = sum(C_therm * P_therm_sol)
    shed_cost = sum(VOLL * P_shed_sol)
    curt_cost = sum(C_curt * P_wind_curt_sol)
    total_shed_energy = P_shed_sol.sum()
    total_curt_energy = P_wind_curt_sol.sum()

    return {
        "model": m,
        "results": results,
        "P_therm": P_therm_sol,
        "P_wind_inj": P_wind_inj_sol,
        "P_wind_curt": P_wind_curt_sol,
        "P_ch": P_ch_sol,
        "P_dis": P_dis_sol,
        "P_shed": P_shed_sol,
        "SoC": SoC_sol,
        "total_cost": total_cost,
        "thermal_cost": thermal_cost,
        "shed_cost": shed_cost,
        "curt_cost": curt_cost,
        "total_shed_energy": total_shed_energy,
        "total_curt_energy": total_curt_energy,
    }


# =========================
# 3. solve with scenarios
# =========================

# scenario 1: normal conditions with no outage of thermal power plant
scenario1 = build_and_solve_scenario(
    P_load=P_load_normal,
    P_wind_avail=P_wind_avail_normal,
    outage_mask=None,
    solver_name="highs",
    verbose=False
)

print("=== scenario 1: normal conditions with no outage of thermal power plant ===")
print(f"Total costs: £{scenario1['total_cost']:.2f}")
print(f"Cost of Thermal power plant: £{scenario1['thermal_cost']:.2f}")
print(f"Cost of Load shedding: £{scenario1['shed_cost']:.2f}")
print(f"Cost of Wind curtailment: £{scenario1['curt_cost']:.2f}")
print(f"Total amount of Load shedding: {scenario1['total_shed_energy']:.2f} MWh")
print(f"Total amount of Wind curtailment: {scenario1['total_curt_energy']:.2f} MWh\n")

hours = np.array(time_index)
# Define event indices and event-local hour axis
event_idx = np.where(event_hours)[0]
hours_event = np.arange(len(event_idx))  # 0..len(event_idx)-1 within Dunkelflaute week

# Scenario 1: plot event week zoomed version
P_therm_s1_ev = scenario1['P_therm'][event_idx]
P_wind_inj_s1_ev = scenario1['P_wind_inj'][event_idx]
P_shed_s1_ev = scenario1['P_shed'][event_idx]
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].bar(hours_event - 0.2, P_therm_s1_ev, width=0.4, label='Thermal')
axes[0].bar(hours_event + 0.2, P_wind_inj_s1_ev, width=0.4, alpha=0.7, label='Wind injection')
axes[0].set_ylabel('Power (MW)')
axes[0].set_title('Scenario 1: Thermal and Wind Injection (Event Week)')
axes[0].legend()
axes[1].bar(hours_event, P_shed_s1_ev, width=0.6, color='red')
axes[1].set_xlabel('Hour in event week')
axes[1].set_ylabel('Load Shedding (MW)')
axes[1].set_title('Scenario 1: Load Shedding (Event Week)')
axes[1].set_xticks(np.arange(0, len(hours_event), 24))
plt.savefig(os.path.join(FIG_DIR, "scenario1_power_shed.png"), dpi=300)
plt.tight_layout()
plt.show()

fig_soc1, ax_soc1 = plt.subplots(figsize=(12, 4))
ax_soc1.plot(hours, scenario1['SoC'])
ax_soc1.set_xlabel('Hour')
ax_soc1.set_ylabel('State of Charge (MWh)')
ax_soc1.set_title('Scenario 1: Battery SoC Over Time')
plt.savefig(os.path.join(FIG_DIR, "scenario1_soc.png"), dpi=300)
plt.tight_layout()
plt.show()


# scenario 2: Dunkelflaute event with no outage of thermal power plant
scenario2 = build_and_solve_scenario(
    P_load=P_load_event,
    P_wind_avail=P_wind_avail_event,
    outage_mask=None,
    solver_name="highs",
    verbose=False
)

print("=== scenario 2: Dunkelflaute event with no outage of thermal power plant ===")
print(f"Total costs: £{scenario2['total_cost']:.2f}")
print(f"Cost of Thermal power plant: £{scenario2['thermal_cost']:.2f}")
print(f"Cost of Load shedding: £{scenario2['shed_cost']:.2f}")
print(f"Cost of Wind curtailment: £{scenario2['curt_cost']:.2f}")
print(f"Total amount of Load shedding: {scenario2['total_shed_energy']:.2f} MWh")
print(f"Total amount of Wind curtailment: {scenario2['total_curt_energy']:.2f} MWh\n")

P_therm_s2_ev = scenario2['P_therm'][event_idx]
P_wind_inj_s2_ev = scenario2['P_wind_inj'][event_idx]
P_shed_s2_ev = scenario2['P_shed'][event_idx]
P_load_ev = P_load_event[event_idx]
P_wind_avail_ev = P_wind_avail_event[event_idx]
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].bar(hours_event - 0.2, P_therm_s2_ev, width=0.4, label='Thermal')
axes[0].bar(hours_event + 0.2, P_wind_inj_s2_ev, width=0.4, alpha=0.7, label='Wind injection')
axes[0].plot(hours_event, P_load_ev, linestyle='--', label='Load (event)')
axes[0].plot(hours_event, P_wind_avail_ev, linestyle=':', label='Wind available (event)')
axes[0].set_ylabel('Power (MW)')
axes[0].set_title('Scenario 2: Thermal and Wind Injection (Event Week)')
axes[0].legend()
axes[1].bar(hours_event, P_shed_s2_ev, width=0.6, color='red')
axes[1].set_xlabel('Hour in event week')
axes[1].set_ylabel('Load Shedding (MW)')
axes[1].set_title('Scenario 2: Load Shedding (Event Week)')
axes[1].set_xticks(np.arange(0, len(hours_event), 24))
plt.savefig(os.path.join(FIG_DIR, "scenario2_power_shed_event.png"), dpi=300)
plt.tight_layout()
plt.show()

fig_soc2, ax_soc2 = plt.subplots(figsize=(12, 4))
ax_soc2.plot(hours, scenario2['SoC'])
ax_soc2.set_xlabel('Hour')
ax_soc2.set_ylabel('State of Charge (MWh)')
ax_soc2.set_title('Scenario 2: Battery SoC Over Time')
plt.savefig(os.path.join(FIG_DIR, "scenario2_soc.png"), dpi=300)
plt.tight_layout()
plt.show()

# shifting windows of 24 hours outage under Dunkelflaute event
def evaluate_outage_windows(P_load, P_wind_avail, mode="planned"):
    """
    mode = 'planned'  -> scenario 3: planned outage
    mode = 'unplanned'-> scenario 4: unplanned outage
    """
    assert mode in ("planned", "unplanned")
    best_result = None
    best_window = None

    all_windows = []
    all_shed_energies = []
    all_SoC = []

    indices = np.where(event_hours)[0]
    start_candidates = [i for i in indices if i + 24 <= indices[-1] + 1]

    for start in start_candidates:
        end = start + 24
        if not event_hours[start:end].all():
            continue

        outage_mask = np.zeros(T, dtype=bool)
        outage_mask[start:end] = True

        result = build_and_solve_scenario(
            P_load=P_load,
            P_wind_avail=P_wind_avail,
            outage_mask=outage_mask,
            solver_name="highs",
            verbose=False
        )

        shed_energy = result["total_shed_energy"]

        all_windows.append((start, end-1))
        all_shed_energies.append(shed_energy)
        all_SoC.append(result["SoC"])

        if best_result is None:
            best_result = result
            best_window = (start, end-1)
        else:
            if mode == "planned":
                if shed_energy < best_result["total_shed_energy"]:
                    best_result = result
                    best_window = (start, end-1)
            else:
                if shed_energy > best_result["total_shed_energy"]:
                    best_result = result
                    best_window = (start, end-1)

    return best_window, best_result, all_windows, all_shed_energies, all_SoC


# Scenario 3: Dunkelflaute event with 24 hours planned outage
best_window_planned, scenario3, windows3, shed3, SoC_all3 = evaluate_outage_windows(
    P_load=P_load_event,
    P_wind_avail=P_wind_avail_event,
    mode="planned"
)

print("=== Scenario 3: Dunkelflaute event with 24 hours planned outage ===")
print(f"Best optimal outage windows: Hours {best_window_planned[0]} – {best_window_planned[1]}")
print(f"Total costs: £{scenario3['total_cost']:.2f}")
print(f"Cost of Thermal power plant: £{scenario3['thermal_cost']:.2f}")
print(f"Cost of Load shedding: £{scenario3['shed_cost']:.2f}")
print(f"Cost of Wind curtailment: £{scenario3['curt_cost']:.2f}")
print(f"Total amount of Load shedding: {scenario3['total_shed_energy']:.2f} MWh")
print(f"Total amount of Wind curtailment: {scenario3['total_curt_energy']:.2f} MWh\n")

P_therm_s3_ev = scenario3['P_therm'][event_idx]
P_wind_inj_s3_ev = scenario3['P_wind_inj'][event_idx]
P_shed_s3_ev = scenario3['P_shed'][event_idx]
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].bar(hours_event - 0.2, P_therm_s3_ev, width=0.4, label='Thermal')
axes[0].bar(hours_event + 0.2, P_wind_inj_s3_ev, width=0.4, alpha=0.7, label='Wind injection')
axes[0].set_ylabel('Power (MW)')
axes[0].set_title('Scenario 3: Thermal and Wind Injection (Event Week, Planned Outage)')
axes[0].legend()
axes[1].bar(hours_event, P_shed_s3_ev, width=0.6, color='red')
axes[1].set_xlabel('Hour in event week')
axes[1].set_ylabel('Load Shedding (MW)')
axes[1].set_title('Scenario 3: Load Shedding (Event Week)')
axes[1].set_xticks(np.arange(0, len(hours_event), 24))
plt.savefig(os.path.join(FIG_DIR, "scenario3_power_shed.png"), dpi=300)
plt.tight_layout()
plt.show()

fig2, ax2 = plt.subplots(figsize=(12, 6))
x = np.arange(len(windows3))
ax2.bar(x, shed3, color='orange')
ax2.set_xticks(x)
ax2.set_xticklabels([f"{w[0]}-{w[1]}" for w in windows3], rotation=45, ha='right')
ax2.set_ylabel('Total Load Shedding (MWh)')
ax2.set_title('Scenario 3: Total Load Shedding for Different 24h Outage Windows')
plt.savefig(os.path.join(FIG_DIR, "scenario3_outage_windows_shed.png"), dpi=300)
plt.tight_layout()
plt.show()

fig_soc3_best, ax_soc3_best = plt.subplots(figsize=(12, 4))
ax_soc3_best.plot(hours, scenario3['SoC'])
ax_soc3_best.set_xlabel('Hour')
ax_soc3_best.set_ylabel('State of Charge (MWh)')
ax_soc3_best.set_title('Scenario 3: Battery SoC Over Time (Best Outage Window)')
plt.savefig(os.path.join(FIG_DIR, "scenario3_soc_best.png"), dpi=300)
plt.tight_layout()
plt.show()

# Scenario 3: SoC curves for all outage windows
fig_soc3_all, ax_soc3_all = plt.subplots(figsize=(12, 4))

# plot SOC of all windows
for soc in SoC_all3:
    ax_soc3_all.plot(hours, soc, color='gray', alpha=0.2)

# highlight the best windows
best_idx3 = windows3.index(best_window_planned)
ax_soc3_all.plot(hours, SoC_all3[best_idx3], linewidth=2, label='Best outage window')

ax_soc3_all.set_xlabel('Hour')
ax_soc3_all.set_ylabel('State of Charge (MWh)')
ax_soc3_all.set_title('Scenario 3: Battery SoC Over Time for All 24h Outage Windows')
ax_soc3_all.legend()
plt.savefig(os.path.join(FIG_DIR, "scenario3_soc_all_windows.png"), dpi=300)
plt.tight_layout()
plt.show()


# Scenarios 4: Dunkelflaute event with 24 hours unplanned outage
worst_window_unplanned, scenario4, windows4, shed4, SoC_all4 = evaluate_outage_windows(
    P_load=P_load_event,
    P_wind_avail=P_wind_avail_event,
    mode="unplanned"
)

print("=== Scenarios 4: Dunkelflaute event with 24 hours unplanned outage ===")
print(f"Worst outage windows: Hours {worst_window_unplanned[0]} – {worst_window_unplanned[1]}")
print(f"Total costs: £{scenario4['total_cost']:.2f}")
print(f"Cost of Thermal power plant: £{scenario4['thermal_cost']:.2f}")
print(f"Cost of Load shedding: £{scenario4['shed_cost']:.2f}")
print(f"Cost of Wind curtailment: £{scenario4['curt_cost']:.2f}")
print(f"Total amount of Load shedding: {scenario4['total_shed_energy']:.2f} MWh")
print(f"Total amount of Wind curtailment: {scenario4['total_curt_energy']:.2f} MWh\n")


P_therm_s4_ev = scenario4['P_therm'][event_idx]
P_wind_inj_s4_ev = scenario4['P_wind_inj'][event_idx]
P_shed_s4_ev = scenario4['P_shed'][event_idx]
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].bar(hours_event - 0.2, P_therm_s4_ev, width=0.4, label='Thermal')
axes[0].bar(hours_event + 0.2, P_wind_inj_s4_ev, width=0.4, alpha=0.7, label='Wind injection')
axes[0].set_ylabel('Power (MW)')
axes[0].set_title('Scenario 4: Thermal and Wind Injection (Event Week, Unplanned Outage)')
axes[0].legend()
axes[1].bar(hours_event, P_shed_s4_ev, width=0.6, color='red')
axes[1].set_xlabel('Hour in event week')
axes[1].set_ylabel('Load Shedding (MW)')
axes[1].set_title('Scenario 4: Load Shedding (Event Week)')
axes[1].set_xticks(np.arange(0, len(hours_event), 24))
plt.savefig(os.path.join(FIG_DIR, "scenario4_power_shed.png"), dpi=300)
plt.tight_layout()
plt.show()

fig2, ax2 = plt.subplots(figsize=(12, 6))
x = np.arange(len(windows4))
ax2.bar(x, shed4, color='orange')
ax2.set_xticks(x)
ax2.set_xticklabels([f"{w[0]}-{w[1]}" for w in windows4], rotation=45, ha='right')
ax2.set_ylabel('Total Load Shedding (MWh)')
ax2.set_title('Scenario 4: Total Load Shedding for Different 24h Outage Windows')
plt.savefig(os.path.join(FIG_DIR, "scenario4_outage_windows_shed.png"), dpi=300)
plt.tight_layout()
plt.show()

# Scenario 4: SoC curve for worst outage window
fig_soc4_best, ax_soc4_best = plt.subplots(figsize=(12, 4))
ax_soc4_best.plot(hours, scenario4['SoC'])
ax_soc4_best.set_xlabel('Hour')
ax_soc4_best.set_ylabel('State of Charge (MWh)')
ax_soc4_best.set_title('Scenario 4: Battery SoC Over Time (Worst Outage Window)')
plt.savefig(os.path.join(FIG_DIR, "scenario4_soc_best.png"), dpi=300)
plt.tight_layout()
plt.show()

# Scenario 4: SoC curves for all outage windows
fig_soc4_all, ax_soc4_all = plt.subplots(figsize=(12, 4))

for soc in SoC_all4:
    ax_soc4_all.plot(hours, soc, color='gray', alpha=0.2)

# highlight the 'worst' window
best_idx4 = windows4.index(worst_window_unplanned)
ax_soc4_all.plot(hours, SoC_all4[best_idx4], linewidth=2, label='Worst outage window')

ax_soc4_all.set_xlabel('Hour')
ax_soc4_all.set_ylabel('State of Charge (MWh)')
ax_soc4_all.set_title('Scenario 4: Battery SoC Over Time for All 24h Outage Windows')
ax_soc4_all.legend()
plt.savefig(os.path.join(FIG_DIR, "scenario4_soc_all_windows.png"), dpi=300)
plt.tight_layout()
plt.show()