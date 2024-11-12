import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum

# Define Workers , Departments , Shifts , and Days
workers = [i for i in range(1, 101)]
departments = ['Battery', 'Body', 'Assembly', 'Paint', 'Quality']
shifts = ['Morning', 'Afternoon', 'Night']
days = ['Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun']

# Create Workers DataFrame
workers_df = pd.DataFrame({
'Worker_ID': np.repeat(workers , len(departments)*len(shifts)*len(days)),
'Department': np.tile(np.repeat(departments , len(shifts)*len(days)),len(workers)),
'Shift': np.tile(np.repeat(shifts , len(days)), len(workers)*len(departments)),
'Day': np.tile(days, len(workers)*len(departments)*len(shifts)),
'Availability': np.random.choice([0, 1], len(workers)*len(departments)*len(shifts)*len(days)),
'Preference_Score': np.random.randint(1, 10, len(workers)*len(departments)*len(shifts)*len(days)),
'Effectiveness_Score': np.random.randint(1, 10, len(workers)*len(departments)*len(shifts)*len(days))
})

# Create Department DataFrame
dept_df = pd.DataFrame({
'Department': np.repeat(departments , len(shifts)*len(days)),
'Shift': np.tile(np.repeat(shifts , len(days)), len(departments)),
'Day': np.tile(days, len(departments)*len(shifts)),
'Min_Workers': np.random.randint(1, 5, len(departments)*len(shifts)*len(days)),
'Max_Workers': np.random.randint(5, 10, len(departments)*len(shifts)*len(days))
})

# Create model
model = Model('q1')

x = model.addVars(workers, departments, shifts, days, vtype=GRB.BINARY, name='x')

model.setObjective(
    quicksum(
        workers_df.loc[
            (workers_df['Worker_ID'] == w) & 
            (workers_df['Department'] == d) & 
            (workers_df['Shift'] == s) & 
            (workers_df['Day'] == t), 
            'Preference_Score'
        ].values[0] * workers_df.loc[
            (workers_df['Worker_ID'] == w) & 
            (workers_df['Department'] == d) & 
            (workers_df['Shift'] == s) & 
            (workers_df['Day'] == t), 
            'Effectiveness_Score'
        ].values[0] * x[w, d, s, t]
        for w in workers for d in departments for s in shifts for t in days
    ),
    GRB.MAXIMIZE
)

# 1. Each worker can only work one shift per day
for w in workers:
    for t in days:
        model.addConstr(
            quicksum(x[w, d, s, t] for d in departments for s in shifts) <= 1,
            name=f"one_shift_per_day_{w}_{t}"
        )

# 2. Each worker can work a maximum of 5 days per week
for w in workers:
    model.addConstr(
        quicksum(x[w, d, s, t] for d in departments for s in shifts for t in days) <= 5,
        name=f"max_5_days_{w}"
    )

# 3. Workers can only be assigned to shifts they are available for
for w in workers:
    for d in departments:
        for s in shifts:
            for t in days:
                availability = workers_df.loc[
                    (workers_df['Worker_ID'] == w) & 
                    (workers_df['Department'] == d) & 
                    (workers_df['Shift'] == s) & 
                    (workers_df['Day'] == t),
                    'Availability'
                ].values[0]
                model.addConstr(
                    x[w, d, s, t] <= availability,
                    name=f"availability_{w}_{d}_{s}_{t}"
                )

# 4. Staffing requirements for each department shift
for d in departments:
    for s in shifts:
        for t in days:
            min_workers = dept_df.loc[
                (dept_df['Department'] == d) & 
                (dept_df['Shift'] == s) & 
                (dept_df['Day'] == t), 
                'Min_Workers'
            ].values[0]
            max_workers = dept_df.loc[
                (dept_df['Department'] == d) & 
                (dept_df['Shift'] == s) & 
                (dept_df['Day'] == t), 
                'Max_Workers'
            ].values[0]
            model.addConstr(
                quicksum(x[w, d, s, t] for w in workers) >= min_workers,
                name=f"min_staff_{d}_{s}_{t}"
            )
            model.addConstr(
                quicksum(x[w, d, s, t] for w in workers) <= max_workers,
                name=f"max_staff_{d}_{s}_{t}"
            )

# Solve model
model.optimize()

# Output results
if model.status == GRB.OPTIMAL:
    print("Optimal Solution Found")
    solution = []
    for w in workers:
        for d in departments:
            for s in shifts:
                for t in days:
                    if x[w, d, s, t].x > 0.5:  # Worker is assigned
                        solution.append((w, d, s, t))
    # Convert to DataFrame for easy visualization
    solution_df = pd.DataFrame(solution, columns=["Worker_ID", "Department", "Shift", "Day"])
    print(solution_df)
else:
    print("No optimal solution found.")