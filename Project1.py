import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

np.random.seed(42)  # for reproducibility

# CMRC (Central Manufacturing & Repair Center)
CMRC = np.array([[0, 0]])

# CDRC (Customer Distribution & Repair Centers)
CDRC_coords = np.array([
    [30.59264474, 6.97469303],
    [14.60723243, 18.31809216],
    [22.80349921, 39.25879807]
])

# Customer Zone (CZ) coordinates
Customer_coords = np.array([
    [9.983689, 25.71172], [29.62073, 2.322521], [30.37724, 8.526206],
    [3.25258, 47.44428], [48.2816, 40.41987], [15.23069, 4.883606],
    [34.21165, 22.00762], [6.101912, 24.75885], [1.719426, 45.46602],
    [12.939, 33.12611]
])

# Distances
dist_CMRC_to_CDRC = cdist(CMRC, CDRC_coords)[0]
dist_CDRC_to_Customers = cdist(CDRC_coords, Customer_coords)
dist_Customers_to_CDRC = cdist(Customer_coords, CDRC_coords)
dist_CDRC_to_CMRC = cdist(CDRC_coords, CMRC)[..., 0]

# Network visualization
def plot_nsga2_supplychain_layout():
    plt.figure(figsize=(8, 8))
    plt.scatter(*CMRC[0], color='red', label='CMRC (0,0)', s=150, marker='s')
    plt.scatter(CDRC_coords[:, 0], CDRC_coords[:, 1], c='blue', label='CDRCs', s=100, marker='^')
    plt.scatter(Customer_coords[:, 0], Customer_coords[:, 1], c='green', label='Customers', s=60)
    for i, (x, y) in enumerate(CDRC_coords):
        plt.text(x + 0.5, y + 0.5, f"CDRC-{i+1}", fontsize=8)
    for i, (x, y) in enumerate(Customer_coords):
        plt.text(x + 0.5, y + 0.5, f"CZ-{i+1}", fontsize=7)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.title("NSGA2-Structured Supply Chain Layout")
    plt.legend()
    plt.show()

plot_nsga2_supplychain_layout()
#%%
# Dimensions
R = CDRC_coords.shape[0]     # Number of CDRCs
I = Customer_coords.shape[0] # Number of customers
P = 2                        # Number of products
T = 240                      # Working days

# Demand and return structure
demand = np.random.uniform(20, 30, (I, P))
return_rate = np.random.uniform(0.3, 0.6, P)
fraction_defective = 0.3
fraction_non_defective = 0.7  # Unused directly in cost
gamma=0.3

# Cost Parameters
fixed_cost = np.random.uniform(1000, 1500, R)
admin_cost = np.random.uniform(5, 10, (R, P))
holding_cost = np.random.uniform(1, 3, (R, P))
repair_cost = np.random.uniform(5, 10, P)

# Shipment Costs
shipment_CMRC_CDRC = np.tile(650 * dist_CMRC_to_CDRC[:, None], (1, P))
shipment_CDRC_CZ = np.tile(102 * dist_CDRC_to_Customers[:, :, None],(1,P))
shipment_CDRC_CMRC = np.tile(102 * dist_CDRC_to_CMRC[:, None], (1, P))

# Emissions: 5 kg/km
carbon_CMRC_to_CDRC = np.tile(5 * dist_CMRC_to_CDRC[:, None], (1, P))
carbon_CDRC_to_Customer = 5 * dist_CDRC_to_Customers[:, :, None]
carbon_Customer_to_CDRC = 5 * dist_Customers_to_CDRC
carbon_CDRC_to_CMRC = np.tile(5 * dist_CDRC_to_CMRC[:, None], (1, P))
#%%
from gurobipy import Model, GRB, quicksum

def build_nsga2_like_model(emission_cap=None):
    model = Model("NSGA2_Like_Gurobi_Model")

    # === Variables ===
    Y = model.addVars(R, vtype=GRB.BINARY, name="Y")                    # CDRC open status
    Z = model.addVars(I, P, R, vtype=GRB.BINARY, name="Z")              # assignment forward
    R_vars = model.addVars(I, P, R, vtype=GRB.BINARY, name="R")         # return assignment
    Q = model.addVars(R, P, lb=0, name="Q")                             # order quantities

    # === Objective: Minimize Z1 = Total Cost ===
    model.setObjective(
        quicksum(Y[r] * fixed_cost[r] for r in range(R)) +  # setup cost

        quicksum(
            Z[i, p, r] * T * demand[i, p] * shipment_CDRC_CZ[r, i, p]
            for i in range(I) for p in range(P) for r in range(R)
        ) +  # forward shipping

        quicksum(
            R_vars[i, p, r] * T * demand[i, p] * return_rate[p] *
            (fraction_defective * repair_cost[p] + shipment_CDRC_CMRC[r, p])
            for i in range(I) for p in range(P) for r in range(R)
        ) +  # return shipping + repair

        quicksum(
            Y[r] * (admin_cost[r, p] + shipment_CMRC_CDRC[r, p] + holding_cost[r, p] * Q[r, p] / 2)
            for r in range(R) for p in range(P)
        ), GRB.MINIMIZE
    )

    # === Constraints ===

    # (3) At least one CDRC must be constructed
    model.addConstr(quicksum(Y[r] for r in range(R)) >= 1, name="AtLeastOneCDRC")

    # (4) Each customer-product assigned to one CDRC
    for i in range(I):
        for p in range(P):
            model.addConstr(quicksum(Z[i, p, r] for r in range(R)) == 1, f"Assign_{i}_{p}")

    # (5) Each customer-product return assigned to one CDRC
    for i in range(I):
        for p in range(P):
            model.addConstr(quicksum(R_vars[i, p, r] for r in range(R)) == 1, f"Return_{i}_{p}")

    # (6) Forward assignment implies CDRC is constructed
    for i in range(I):
        for p in range(P):
            for r in range(R):
                model.addConstr(Z[i, p, r] <= Y[r], f"ZImpliesY_{i}_{p}_{r}")

    # (7/9) Return assignment only if CDRC is used in forward assignment
    #for i in range(I):
     #   for p in range(P):
      #      for r in range(R):
       #         model.addConstr(R_vars[i, p, r] <= Z[i, p, r], f"ReturnImpliesZ_{i}_{p}_{r}")

    # (12) Order quantity non-negativity already handled via lb=0 in Q

    # === Optional Emission Cap Constraint for Pareto generation ===
    total_emissions = quicksum(
    Z[i, p, r] * T * demand[i, p] * (
        5 * dist_CMRC_to_CDRC[r] + 
        5 * dist_CDRC_to_Customers[r, i]
        )
        for i in range(I) for p in range(P) for r in range(R)
        ) +quicksum(
        R_vars[i, p, r] * T * demand[i, p] * return_rate[p] * gamma * (
        5 * dist_Customers_to_CDRC[i, r] + 
        5 * dist_CDRC_to_CMRC[r]
          )
        for i in range(I) for p in range(P) for r in range(R)
        ) + quicksum(
        R_vars[i, p, r] * T * demand[i, p] * return_rate[p] * (1 - gamma) * (
        5 * dist_Customers_to_CDRC[i, r] + 
        5 * dist_CDRC_to_Customers[r, i]
          )
        for i in range(I) for p in range(P) for r in range(R)
        )

    if emission_cap is not None:
        model.addConstr(total_emissions <= emission_cap, name="NSGA2_EmissionCap")

    return model, Y, Z, R_vars, Q, total_emissions

#%%
import time

def run_nsga2_style_emission_loop():
    # Determine a loose upper bound for emissions
    model0, _, _, _, _, emissions_unconstrained = build_nsga2_like_model()
    model0.setParam('OutputFlag', 0)
    model0.optimize()

    if model0.status == GRB.OPTIMAL:
        emission_upper_estimate = emissions_unconstrained.getValue() * 1.1  # 10% buffer
    else:
        raise RuntimeError("Initial model to estimate emissions did not solve.")

    #emission_lower_limit = emission_upper_estimate * 0.4

    emission_caps = np.linspace(41_500_000, 36_000_000, num=1000)


    pareto_solutions = []

    print("⚙️ Generating NSGA2-style Pareto front using Gurobi MILP...\n")
    for idx, epsilon in enumerate(emission_caps):
        print(f"➡️ Run {idx+1:2}: Solving for emission cap = {epsilon:.2f}")
        model, Y, Z, R_vars, Q, emissions = build_nsga2_like_model(emission_cap=epsilon)

        start_time = time.time()
        model.setParam('OutputFlag', 0)  # suppress solver output
        model.optimize()
        elapsed = time.time() - start_time

        if model.status == GRB.OPTIMAL:
            z1 = model.ObjVal
            z2 = emissions.getValue()
            
            assignment = {
                "Y": [int(Y[r].X) for r in range(R)],
                "Z": {(i, p): r for i in range(I) for p in range(P)
                      for r in range(R) if Z[i, p, r].X > 0.5},
                "R": {(i, p): r for i in range(I) for p in range(P)
                      for r in range(R) if R_vars[i, p, r].X > 0.5},
                "Q": {(r, p): Q[r, p].X for r in range(R) for p in range(P)}
            }

            pareto_solutions.append((z1, z2, epsilon, elapsed, assignment))
            print(f"✅ Feasible | Z1 = {z1:.2f}, Z2 = {z2:.2f}, Time = {elapsed:.2f} s\n")
        else:
            print("❌ Infeasible at this ε — skipping.\n")

    return pareto_solutions
#%%
pareto_points = run_nsga2_style_emission_loop()
#%%
import matplotlib.pyplot as plt

def plot_nsga2_like_front(pareto_solutions):
    if not pareto_solutions:
        print("❌ No Pareto solutions found.")
        return

    # Sort by Z1 (Cost)
    pareto_sorted = sorted(pareto_solutions, key=lambda x: x[0])
    Z1_vals = [pt[0] for pt in pareto_sorted]
    Z2_vals = [pt[1] for pt in pareto_sorted]

    # Best solutions
    min_z1 = min(pareto_sorted, key=lambda x: x[0])
    min_z2 = min(pareto_sorted, key=lambda x: x[1])

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(Z1_vals, Z2_vals, '-o', color='darkgreen', label='MILP Pareto Front')
    plt.scatter(min_z1[0], min_z1[1], color='blue', s=100, edgecolors='k',
                label=f'Min Z1: ({min_z1[0]:.2e}, {min_z1[1]:.2e})')
    plt.scatter(min_z2[0], min_z2[1], color='red', s=100, edgecolors='k',
                label=f'Min Z2: ({min_z2[0]:.2e}, {min_z2[1]:.2e})')

    plt.xlabel("Z1: Total Cost")
    plt.ylabel("Z2: Carbon Emissions")
    plt.title("NSGA2-style Pareto Front (via Gurobi MILP)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Table of solutions
    print("\n📋 Final Pareto Points:")
    for i, sol in enumerate(pareto_sorted):
        z1, z2, eps, t = sol[:4]
        print(f"Point {i+1:2}: Z1 = {z1:,.2f}, Z2 = {z2:,.2f}, ε = {eps:,.2f}, Time = {t:.2f}s")
#%%
# Plot results
plot_nsga2_like_front(pareto_points)
#%%
def print_assignments(solution, label=""):
    z1, z2, epsilon, time_taken, assignment = solution
    print(f"\n📌 Assignment Summary for {label}")
    print(f"Z1 = {z1:.2f}, Z2 = {z2:.2f}, ε = {epsilon:.2f}, Time = {time_taken:.2f}s")

    print("\n🟦 Active CDRCs:")
    for r, status in enumerate(assignment["Y"]):
        if status:
            print(f"  - CDRC-{r+1} (Active)")

    print("\n🟩 Forward Assignments (CDRC → Customer):")
    for (i, p), r in assignment["Z"].items():
        print(f"  - Customer-{i+1} Product-{p+1} → CDRC-{r+1}")

    print("\n🟥 Return Assignments (Customer → CDRC):")
    for (i, p), r in assignment["R"].items():
        print(f"  - Customer-{i+1} Product-{p+1} ← CDRC-{r+1}")

    print("\n📦 Order Quantities (Q[r, p]):")
    for (r, p), qty in assignment["Q"].items():
        print(f"  - CDRC-{r+1} Product-{p+1}: Q = {qty:.2f}")
#%%
# Pick best by cost (Z1)
best_cost_sol = min(pareto_points, key=lambda x: x[0])
print_assignments(best_cost_sol, "Min Z1")

# Pick best by emissions (Z2)
best_emission_sol = min(pareto_points, key=lambda x: x[1])
print_assignments(best_emission_sol, "Min Z2")
#%%
import pandas as pd

def export_gurobi_pareto_summary(pareto_solutions):
    data = []

    for i, (z1, z2, eps, t, _) in enumerate(pareto_solutions):
        data.append({
            "Solution": f"ε-{i+1}",
            "Z1 (Total Cost)": z1,
            "Z2 (Emissions)": z2,
            "ε Used": eps,
            "Computation Time (s)": t
        })

    df = pd.DataFrame(data)
    df.set_index("Solution", inplace=True)
    
    print("📋 Pareto Front Summary Table (Gurobi ε-Constraint):")
    print(df)

    # Save it
    df.to_csv("gurobi_pareto_results.csv")
    print("\n✅ Exported to: gurobi_pareto_results.csv")

# Run this
export_gurobi_pareto_summary(pareto_points)#%%
#%%
import matplotlib.pyplot as plt

def plot_focused_pareto(pareto_solutions, resume_cost=327.10e6, resume_emission=36.92e6):
    # Sort by emissions for better line plot
    pareto_solutions.sort(key=lambda x: x[1])
    costs = [z1 for z1, z2, _, _, _ in pareto_solutions]
    emissions = [z2 for z1, z2, _, _, _ in pareto_solutions]

    plt.figure(figsize=(10, 6))
    plt.plot(emissions, costs, color='blue', label='Pareto Curve')
    plt.scatter([resume_emission], [resume_cost], color='red', label='Resume Point', zorder=5)
    plt.axhline(resume_cost, color='red', linestyle='--', linewidth=1)
    plt.axvline(resume_emission, color='red', linestyle='--', linewidth=1)

    plt.xlabel('Carbon Emissions (kg CO₂)')
    plt.ylabel('Total Cost (INR)')
    plt.title('Focused Pareto Front (36M – 38.5M kg CO₂)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
#%%
plot_focused_pareto(pareto_points)
