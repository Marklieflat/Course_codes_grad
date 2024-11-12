import pandas as pd
import numpy as np

childcare = pd.read_csv(r'Project_1\project1_new_datasets\new_child_care.csv')
employment = pd.read_csv(r'Project_1\project1_new_datasets\new_employment.csv')
income = pd.read_csv(r'Project_1\project1_new_datasets\new_income.csv')
population = pd.read_csv(r'Project_1\project1_new_datasets\new_population.csv')
potential_loc = pd.read_csv(r'Project_1\project1_new_datasets\new_potential_loc.csv')


population = population.iloc[:, :5].drop(['Total'], axis=1)
population['2w-12yrs'] = np.floor(population.iloc[:, 1:].sum(axis=1)*13/15).astype(int)
demand_desert = pd.merge(population, employment, on='zip_code', how = 'outer')
demand_desert = pd.merge(demand_desert, income, on='zip_code', how = 'outer')
demand_desert['high_demand'] = (demand_desert['employment rate'] >= 0.6)|(demand_desert['average income'] <= 60000)
demand_desert['high_demand'] = demand_desert['high_demand'].astype(int)

childcare_capacity = childcare.groupby('zip_code')['infant_capacity','toddler_capacity','preschool_capacity'
                                                   ,'children_capacity'].sum().reset_index()
childcare_capacity['2w_5yr_cap'] = np.floor(childcare_capacity.iloc[:, 1:4].sum(axis=1)+childcare_capacity['children_capacity']*5/12).astype(int)
childcare_capacity['2w_12yr_cap'] = np.floor(childcare_capacity.iloc[:, 1:5].sum(axis=1)).astype(int)

demand_desert = pd.merge(demand_desert, childcare_capacity, on='zip_code', how = 'outer')

def classify_desert(row):
    if row['high_demand'] == 1:
        return row['2w_12yr_cap'] <= row['2w-12yrs']*0.5
    else:
        return row['2w_12yr_cap'] <= row['2w-12yrs']*1/3

demand_desert['desert'] = demand_desert.apply(classify_desert, axis=1).astype(int)

print(demand_desert.head())
demand_desert.to_csv(r'Project_1\project1_new_datasets\demand_desert.csv', index=False)

