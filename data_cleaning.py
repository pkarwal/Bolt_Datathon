import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

DATA = {
    'employee'  : 'BOLT_Employees.csv',
    'applicant' : 'BOLT_Applicants.csv',
    'branch'    : 'BOLT_Branch.csv',
    'perf'      : 'BOLT_Performance.csv',
    'changes'   : 'BOLT_EmployeeChanges.csv',
}

REFERENCE_DATE = pd.Timestamp('2026-03-14')
HP_THRESHOLD   = 85.44   # 75th percentile of avg scores



# 1. LOAD
raw = {}
for name, fname in DATA.items():
    raw[name] = pd.read_csv(fname)
    print(f"  ✅ {name:12s} → {raw[name].shape[0]:>5d} rows × {raw[name].shape[1]} cols")



# 2. CLEAN: EMPLOYEE
emp = raw['employee'].copy()
emp['HiredOn']    = pd.to_datetime(emp['HiredOn'])
emp['emp_id']     = emp['EmployeeID']
emp['app_id']     = emp['ApplicantID']
emp['branch']     = emp['Branch#']
emp['status']     = emp['Current status'].str.strip()
emp['role']       = emp['Role'].str.strip()
emp['wage']       = emp['Wage'].str.strip()
emp['position']   = emp['Position'].str.strip()
emp['hours']      = emp['AvgWorkingHours/Week']
emp['is_exited']  = emp['status'].isin(['Left', 'Fired']).astype(int)
emp['is_fired']   = (emp['status'] == 'Fired').astype(int)
emp['tenure_days']   = (REFERENCE_DATE - emp['HiredOn']).dt.days
emp['tenure_months'] = (emp['tenure_days'] / 30.44).round(1)

role_level = {'Host':1,'Server Assistant':2,'Server':3,'Bartender':4,'Shift Lead':4,'Manager':5}
emp['role_level'] = emp['role'].map(role_level)

wage_num = {'Minimum':1,'Competitive':2,'Premium':3}
emp['wage_num'] = emp['wage'].map(wage_num)

def hours_bracket(h):
    if h < 20:  return 'Part-time (<20h)'
    if h < 32:  return 'Mid (20-32h)'
    return 'Full-time (32h+)'
emp['hours_bracket'] = emp['hours'].apply(hours_bracket)



# 3. CLEAN: PERFORMANCE
perf = raw['perf'].copy()
perf['DateReviewed'] = pd.to_datetime(perf['DateReviewed'])
perf['emp_id']       = perf['EmployeeID']

perf_agg = perf.groupby('emp_id').agg(
    avg_score   = ('PerformanceScore', 'mean'),
    max_score   = ('PerformanceScore', 'max'),
    num_reviews = ('PerformanceScore', 'count'),
    score_trend = ('PerformanceScore',
                   lambda x: round(x.iloc[-1]-x.iloc[0],2) if len(x)>1 else 0),
).reset_index()

median_score = perf_agg['avg_score'].median()

def perf_tier(s):
    if s >= HP_THRESHOLD: return 'High Performer'
    if s >= median_score: return 'Medium Performer'
    return 'Low Performer'

perf_agg['perf_tier'] = perf_agg['avg_score'].apply(perf_tier)
print(f"\nPerformance tiers:\n{perf_agg['perf_tier'].value_counts()}")

# 4. CLEAN: APPLICANTS
app = raw['applicant'].copy()
app['app_id'] = app['ApplicantID']
app['age']    = 2026 - app['YearOfBirth']

def age_group(a):
    if a < 23: return 'Under 23 (Student-age)'
    if a < 28: return '23-27'
    if a < 35: return '28-34'
    return '35+'
app['age_group'] = app['age'].apply(age_group)



# 5. CLEAN: BRANCH
branch = raw['branch'].copy()
branch['DatePosted'] = pd.to_datetime(branch['DatePosted'])
branch_names = {1:'UBC',2:'SFU',3:'Downtown',4:'Main St',5:'Richmond',6:'Surrey',7:'Metrotown'}
branch['branch_name'] = branch['BranchNo'].map(branch_names)



# 6. CLEAN: EMPLOYEE CHANGES
chg = raw['changes'].copy()
chg['DateChanged'] = pd.to_datetime(chg['DateChanged'])
chg['emp_id']      = chg['EmployeeID']
chg['new_role']    = chg['New Role'].str.strip()

EXIT_ROLES = ['Quit', 'Dismissed']
chg['is_exit']      = chg['new_role'].isin(EXIT_ROLES).astype(int)
chg['is_promotion'] = (~chg['new_role'].isin(EXIT_ROLES)).astype(int)
print(f"\nChanges breakdown:\n{chg['new_role'].value_counts()}")
print(f"\nReasons:\n{chg['ReasonForLeaving'].value_counts()}")



# 7. BUILD MASTER TABLE
promotions  = chg[chg['is_promotion']==1].copy()
first_promo = promotions.groupby('emp_id')['DateChanged'].min().reset_index()
first_promo.columns = ['emp_id','first_promo_date']
first_promo['was_promoted'] = 1

exits_chg = chg[chg['is_exit']==1][['emp_id','ReasonForLeaving']].copy()

master = emp.copy()
master = master.merge(perf_agg, on='emp_id', how='left')
master = master.merge(first_promo, on='emp_id', how='left')
master = master.merge(exits_chg, on='emp_id', how='left')
master = master.merge(
    app[['app_id','PastRelevantExperience','YearsOfRelevantExperience',
         'HighestEducationLevel','YearOfBirth','age','age_group']],
    on='app_id', how='left'
)

master['was_promoted']     = master['was_promoted'].fillna(0).astype(int)
master['first_promo_date'] = pd.to_datetime(master['first_promo_date'])
master['days_to_promo']    = (master['first_promo_date'] - master['HiredOn']).dt.days
master['months_to_promo']  = (master['days_to_promo'] / 30.44).round(1)
master['is_hp']            = (master['perf_tier']=='High Performer').astype(int)

def tenure_bracket(m):
    if pd.isna(m): return 'Unknown'
    if m<=6:  return '0-6 mo'
    if m<=12: return '6-12 mo'
    if m<=18: return '12-18 mo'
    if m<=24: return '18-24 mo'
    return '24+ mo'
master['tenure_bracket'] = master['tenure_months'].apply(tenure_bracket)



# 8. BRANCH SUMMARY TABLE

branch_stars = branch.groupby('BranchNo')['Stars'].agg(
    avg_stars='mean', num_reviews='count'
).reset_index().rename(columns={'BranchNo':'branch'})

hp_by_branch = master[master['is_hp']==1].groupby('branch').apply(
    lambda x: pd.Series({'hp_total':len(x), 'hp_exited':x['is_exited'].sum()})
).reset_index()
hp_by_branch['hp_exit_rate'] = hp_by_branch['hp_exited']/hp_by_branch['hp_total']*100
hp_by_branch['branch_name']  = hp_by_branch['branch'].map(branch_names)
hp_by_branch = hp_by_branch.merge(branch_stars, on='branch', how='left')



# 9. SAVE
master.to_csv('master.csv', index=False)
perf.to_csv('perf_clean.csv', index=False)
branch.to_csv('branch_clean.csv', index=False)
hp_by_branch.to_csv('branch_summary.csv', index=False)
chg.to_csv('changes_clean.csv', index=False)

print("\n✅ STAGE 1 COMPLETE")
print(f"   Total employees : {len(master)}")
print(f"   High performers : {master['is_hp'].sum()} ({master['is_hp'].mean():.0%})")
print(f"   HP exit rate    : {master[master['is_hp']==1]['is_exited'].mean():.1%}")
print(f"   HP promo rate   : {master[master['is_hp']==1]['was_promoted'].mean():.1%}")