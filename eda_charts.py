
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend — works without a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os

# ── Load cleaned data ─────────────────────────────────────────────────────────
master   = pd.read_csv('master.csv')
perf     = pd.read_csv('perf_clean.csv')
branch   = pd.read_csv('branch_clean.csv')
br_sum   = pd.read_csv('branch_summary.csv')
chg      = pd.read_csv('changes_clean.csv')

perf['DateReviewed'] = pd.to_datetime(perf['DateReviewed'])
chg['DateChanged']   = pd.to_datetime(chg['DateChanged'])
master['HiredOn']    = pd.to_datetime(master['HiredOn'])

HP_THRESHOLD = 85.44
os.makedirs('charts', exist_ok=True)

# ── DARK THEME SETUP ─────────────────────────────────────────────────────────
DARK_BG   = '#1a1a2e'
NAVY_MID  = '#16213e'
PURPLE    = '#7c5cbf'
ORANGE    = '#f5a623'
GREEN     = '#2ecc71'
RED       = '#e74c3c'
BLUE      = '#3498db'
WHITE     = '#ffffff'
GRAY      = '#95a5a6'

plt.rcParams.update({
    'figure.facecolor' : DARK_BG,
    'axes.facecolor'   : NAVY_MID,
    'axes.edgecolor'   : '#2a2a4a',
    'text.color'       : WHITE,
    'axes.labelcolor'  : WHITE,
    'xtick.color'      : WHITE,
    'ytick.color'      : WHITE,
    'grid.color'       : '#2a2a4a',
    'grid.alpha'       : 0.5,
    'font.family'      : 'DejaVu Sans',
})

branch_names = {1:'UBC',2:'SFU',3:'Downtown',4:'Main St',5:'Richmond',6:'Surrey',7:'Metrotown'}



# CHART 1 — Performance Score Distribution
# Answers: How is "high-performing" defined? (Guiding Q1)
fig, ax = plt.subplots(figsize=(10, 5), facecolor=DARK_BG)
ax.set_facecolor(NAVY_MID)

scores = master['avg_score'].dropna()
ax.hist(scores[scores < HP_THRESHOLD], bins=25, color=BLUE,   alpha=0.85, label='Medium / Low Performer', edgecolor='white', linewidth=0.3)
ax.hist(scores[scores >= HP_THRESHOLD], bins=15, color=ORANGE, alpha=0.85, label=f'High Performer (≥{HP_THRESHOLD})', edgecolor='white', linewidth=0.3)
ax.axvline(HP_THRESHOLD, color=ORANGE, linewidth=2.5, linestyle='--')
ax.text(HP_THRESHOLD+0.5, ax.get_ylim()[1]*0.85, f'HP Threshold\n{HP_THRESHOLD}',
        color=ORANGE, fontsize=10, fontweight='bold')

hp_pct = (scores >= HP_THRESHOLD).mean()
ax.set_title(f'Performance Score Distribution   |   {hp_pct:.0%} of workforce are High Performers',
             fontsize=13, fontweight='bold', color=WHITE, pad=12)
ax.set_xlabel('Average Performance Score (0–100)', fontsize=11)
ax.set_ylabel('Number of Employees', fontsize=11)
ax.legend(fontsize=10, facecolor=DARK_BG, edgecolor='gray', labelcolor=WHITE)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('charts/chart01_perf_dist.png', dpi=180, bbox_inches='tight', facecolor=DARK_BG)
plt.close()




# CHART 2 — Turnover Rate by Performance Tier
# Answers: What is the turnover rate of HPs compared to others? (Guiding Q1)
tier_order = ['High Performer','Medium Performer','Low Performer']
tier_stats = master.groupby('perf_tier').agg(
    total  = ('emp_id','count'),
    exited = ('is_exited','sum')
).reindex(tier_order)
tier_stats['exit_rate'] = tier_stats['exited'] / tier_stats['total']
print(f"\nTurnover by tier:\n{tier_stats}")

fig, ax = plt.subplots(figsize=(9, 5), facecolor=DARK_BG)
ax.set_facecolor(NAVY_MID)
bars = ax.bar(tier_stats.index, tier_stats['exit_rate']*100,
              color=[RED, BLUE, GREEN], edgecolor='white', linewidth=0.5, width=0.55)
for bar, val, total in zip(bars, tier_stats['exit_rate'], tier_stats['total']):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f'{val:.1%}\n(n={total})', ha='center', fontsize=11, fontweight='bold', color=WHITE)
ax.set_ylim(0, 110)
ax.set_title('Turnover Rate by Performance Tier', fontsize=13, fontweight='bold', color=WHITE, pad=12)
ax.set_ylabel('Exit Rate (%)', fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('charts/chart02_turnover_tier.png', dpi=180, bbox_inches='tight', facecolor=DARK_BG)
plt.close()




# CHART 3 — Tenure at Exit: Box Plot by Tier
# Answers: At what tenure point are HPs most likely to exit? (Guiding Q2)
exited = master[master['is_exited']==1].copy()
print(f"\nTenure at exit by tier:\n{exited.groupby('perf_tier')['tenure_months'].describe()}")

fig, ax = plt.subplots(figsize=(10, 5), facecolor=DARK_BG)
ax.set_facecolor(NAVY_MID)
tier_labels = ['High\nPerformer','Medium\nPerformer','Low\nPerformer']
data_by_tier = [exited[exited['perf_tier']==t]['tenure_months'].dropna().values for t in tier_order]
bp = ax.boxplot(data_by_tier, patch_artist=True,
                medianprops=dict(color='white', linewidth=2.5),
                whiskerprops=dict(color=GRAY), capprops=dict(color=GRAY),
                flierprops=dict(marker='o', markersize=3, alpha=0.4))
for patch, color in zip(bp['boxes'], [RED, BLUE, GREEN]):
    patch.set_facecolor(color); patch.set_alpha(0.8)
ax.set_xticklabels(tier_labels, fontsize=11)
for i, d in enumerate(data_by_tier):
    med = np.median(d)
    ax.text(i+1, med+1, f'Median\n{med:.0f} mo', ha='center', fontsize=9, color=WHITE, fontweight='bold')
ax.set_title('Tenure at Exit by Performance Tier', fontsize=13, fontweight='bold', color=WHITE, pad=12)
ax.set_ylabel('Tenure at Exit (months)', fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('charts/chart03_tenure_exit.png', dpi=180, bbox_inches='tight', facecolor=DARK_BG)
plt.close()




# CHART 4 — HP Exit Tenure Brackets (danger zone)
# Answers: Where exactly in tenure do HP exits cluster?
hp_exited = exited[exited['perf_tier']=='High Performer']
bracket_order = ['0-6 mo','6-12 mo','12-18 mo','18-24 mo','24+ mo']
hp_brackets = hp_exited['tenure_bracket'].value_counts().reindex(bracket_order).fillna(0)
print(f"\nHP exit tenure brackets:\n{hp_brackets}")

fig, ax = plt.subplots(figsize=(10, 5), facecolor=DARK_BG)
ax.set_facecolor(NAVY_MID)
bar_colors = [ORANGE if b=='24+ mo' else PURPLE for b in bracket_order]
bars = ax.bar(bracket_order, hp_brackets.values, color=bar_colors, edgecolor='white', linewidth=0.5, width=0.6)
for bar, val in zip(bars, hp_brackets.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            str(int(val)), ha='center', fontsize=12, fontweight='bold', color=WHITE)
ax.set_title('High Performer Exits by Tenure Bracket\n(Exit volume peaks at 24+ months — structured disillusionment)',
             fontsize=13, fontweight='bold', color=WHITE, pad=12)
ax.set_ylabel('Number of HP Exits', fontsize=11)
ax.set_xlabel('Tenure at Exit', fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('charts/chart04_hp_tenure_brackets.png', dpi=180, bbox_inches='tight', facecolor=DARK_BG)
plt.close()


# CHART 5 — Kaplan-Meier Style Retention Curve
# Answers: Retention probability over time by tier (Guiding Q2)
def km_curve(df_group, max_months=60):
    times  = df_group['tenure_months'].dropna().sort_values().values
    n      = len(times)
    months = np.arange(0, max_months+1, 1)
    surv   = np.array([(times > t).sum() / n for t in months])
    return months, surv

fig, ax = plt.subplots(figsize=(11, 6), facecolor=DARK_BG)
ax.set_facecolor(NAVY_MID)

for tier, color, lw in [
    ('High Performer',   RED,  2.5),
    ('Medium Performer', BLUE, 2.0),
    ('Low Performer',    GREEN,2.0),
]:
    subset = master[master['perf_tier']==tier]
    months, surv = km_curve(subset, max_months=56)
    ax.plot(months, surv*100, color=color, linewidth=lw, label=tier)
    idx_50 = np.where(surv <= 0.50)[0]
    if len(idx_50):
        m50 = months[idx_50[0]]
        ax.axvline(m50, color=color, linestyle=':', alpha=0.5, linewidth=1.2)
        ax.text(m50+0.5, 52, f'{m50}mo', color=color, fontsize=9, fontweight='bold')

ax.axhline(50, color='white', linestyle='--', alpha=0.3, linewidth=1)
ax.text(1, 51.5, '50% Retention', color='white', fontsize=9, alpha=0.6)
ax.set_title('Employee Retention Curve by Performance Tier\n(Kaplan-Meier Style — % still employed over time)',
             fontsize=13, fontweight='bold', color=WHITE, pad=12)
ax.set_xlabel('Months Since Hire', fontsize=11)
ax.set_ylabel('% Still Employed', fontsize=11)
ax.set_xlim(0, 56)
ax.set_ylim(0, 105)
ax.legend(fontsize=11, facecolor=DARK_BG, edgecolor='gray', labelcolor=WHITE)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('charts/chart05_km_curve.png', dpi=180, bbox_inches='tight', facecolor=DARK_BG)
plt.close()



# CHART 6 — Wage Stagnation: Exited vs Retained HPs
# Answers: Are HPs leaving on low wages? (Guiding Q7)
hp = master[master['is_hp']==1]
hp_exited_wage = hp[hp['is_exited']==1]['wage'].value_counts(normalize=True)*100
hp_retain_wage = hp[hp['is_exited']==0]['wage'].value_counts(normalize=True)*100
wage_order = ['Minimum','Competitive','Premium']
hp_exited_wage = hp_exited_wage.reindex(wage_order).fillna(0)
hp_retain_wage = hp_retain_wage.reindex(wage_order).fillna(0)
print(f"\nHP Exited wage: {hp_exited_wage.to_dict()}")
print(f"HP Retained wage: {hp_retain_wage.to_dict()}")

fig, ax = plt.subplots(figsize=(10, 5), facecolor=DARK_BG)
ax.set_facecolor(NAVY_MID)
x = np.arange(len(wage_order)); w = 0.35
b1 = ax.bar(x-w/2, hp_exited_wage.values, w, label='Exited HP',   color=RED,   alpha=0.85, edgecolor='white', linewidth=0.5)
b2 = ax.bar(x+w/2, hp_retain_wage.values, w, label='Retained HP', color=GREEN, alpha=0.85, edgecolor='white', linewidth=0.5)
for bar in list(b1)+list(b2):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f'{bar.get_height():.1f}%', ha='center', fontsize=10, fontweight='bold', color=WHITE)
ax.set_xticks(x); ax.set_xticklabels(wage_order, fontsize=12)
ax.set_title('Wage Category: Exited vs Retained High Performers',
             fontsize=13, fontweight='bold', color=WHITE, pad=12)
ax.set_ylabel('% of Group', fontsize=11)
ax.legend(fontsize=11, facecolor=DARK_BG, edgecolor='gray', labelcolor=WHITE)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('charts/chart06_wage_hp.png', dpi=180, bbox_inches='tight', facecolor=DARK_BG)
plt.close()




# CHART 7 — Branch HP Exit Rate + Stars Scatter
# Answers: Are there branch-level differences? (Guiding Q6)
hp_by_branch = master[master['is_hp']==1].groupby('branch').apply(
    lambda x: pd.Series({'hp_total':len(x),'hp_exited':x['is_exited'].sum()})
).reset_index()
hp_by_branch['hp_exit_rate'] = hp_by_branch['hp_exited']/hp_by_branch['hp_total']*100
hp_by_branch['name']         = hp_by_branch['branch'].map(branch_names)
hp_by_branch = hp_by_branch.sort_values('hp_exit_rate', ascending=False)
branch_stars = branch.groupby('BranchNo')['Stars'].mean().reset_index()
branch_stars.columns = ['branch','avg_stars']
hp_by_branch = hp_by_branch.merge(branch_stars, on='branch')
print(f"\nHP exit by branch:\n{hp_by_branch[['name','hp_exit_rate','avg_stars']].to_string(index=False)}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=DARK_BG)
for ax in axes: ax.set_facecolor(NAVY_MID)

avg_rate = hp_by_branch['hp_exit_rate'].mean()
bar_colors = [RED if r > avg_rate else BLUE for r in hp_by_branch['hp_exit_rate']]
bars = axes[0].bar(hp_by_branch['name'], hp_by_branch['hp_exit_rate'],
                   color=bar_colors, edgecolor='white', linewidth=0.5, width=0.6)
axes[0].axhline(avg_rate, color=ORANGE, linestyle='--', linewidth=1.5, label=f'Avg: {avg_rate:.1f}%')
for bar, val in zip(bars, hp_by_branch['hp_exit_rate']):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold', color=WHITE)
axes[0].set_title('High Performer Exit Rate by Branch', fontsize=12, fontweight='bold', color=WHITE)
axes[0].set_ylabel('HP Exit Rate (%)', fontsize=10)
axes[0].legend(fontsize=9, facecolor=DARK_BG, labelcolor=WHITE)
axes[0].grid(axis='y', alpha=0.3)

axes[1].scatter(hp_by_branch['hp_exit_rate'], hp_by_branch['avg_stars'],
                s=180, color=ORANGE, edgecolor='white', linewidth=1.5, zorder=5)
for _, row in hp_by_branch.iterrows():
    axes[1].annotate(row['name'], (row['hp_exit_rate'], row['avg_stars']),
                     textcoords='offset points', xytext=(6,4), fontsize=9, color=WHITE)
z = np.polyfit(hp_by_branch['hp_exit_rate'], hp_by_branch['avg_stars'], 1)
x_line = np.linspace(hp_by_branch['hp_exit_rate'].min(), hp_by_branch['hp_exit_rate'].max(), 50)
axes[1].plot(x_line, np.poly1d(z)(x_line), color=RED, linestyle='--', linewidth=1.5, alpha=0.7)
axes[1].set_title('HP Exit Rate vs Google Review Stars', fontsize=12, fontweight='bold', color=WHITE)
axes[1].set_xlabel('HP Exit Rate (%)', fontsize=10)
axes[1].set_ylabel('Avg Google Stars', fontsize=10)
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('charts/chart07_branch.png', dpi=180, bbox_inches='tight', facecolor=DARK_BG)
plt.close()




# CHART 8 — Hours Bracket vs HP Exit Rate
# Answers: Do working hours influence exit risk? (Guiding Q8)
hp_hours = master[master['is_hp']==1].groupby('hours_bracket').apply(
    lambda x: pd.Series({'total':len(x),'exited':x['is_exited'].sum()})
).reset_index()
hp_hours['exit_rate'] = hp_hours['exited']/hp_hours['total']*100
print(f"\nHP exit by hours:\n{hp_hours}")

fig, ax = plt.subplots(figsize=(9, 5), facecolor=DARK_BG)
ax.set_facecolor(NAVY_MID)
bars = ax.bar(hp_hours['hours_bracket'], hp_hours['exit_rate'],
              color=[ORANGE, PURPLE, BLUE], edgecolor='white', linewidth=0.5, width=0.5)
for bar, val, tot in zip(bars, hp_hours['exit_rate'], hp_hours['total']):
    if not np.isnan(val):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f'{val:.1f}%\n(n={int(tot)})', ha='center', fontsize=11, fontweight='bold', color=WHITE)
ax.set_title('High Performer Exit Rate by Working Hours', fontsize=13, fontweight='bold', color=WHITE, pad=12)
ax.set_ylabel('HP Exit Rate (%)', fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('charts/chart08_hours.png', dpi=180, bbox_inches='tight', facecolor=DARK_BG)
plt.close()




# CHART 9 — Age Group Exit Rates (challenge manager narrative)
# Answers: Is it really just students leaving? (Guiding Q9)
age_exit = master[master['is_hp']==1].groupby('age_group').apply(
    lambda x: pd.Series({'total':len(x),'exited':x['is_exited'].sum()})
).reset_index()
age_exit['exit_rate'] = age_exit['exited']/age_exit['total']*100
print(f"\nHP exit by age:\n{age_exit}")

fig, ax = plt.subplots(figsize=(10, 5), facecolor=DARK_BG)
ax.set_facecolor(NAVY_MID)
colors_age = [ORANGE, PURPLE, BLUE, GREEN]
bars = ax.bar(age_exit['age_group'], age_exit['exit_rate'],
              color=colors_age[:len(age_exit)], edgecolor='white', linewidth=0.5, width=0.55)
for bar, val, tot in zip(bars, age_exit['exit_rate'], age_exit['total']):
    if not np.isnan(val):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f'{val:.1f}%\n(n={int(tot)})', ha='center', fontsize=11, fontweight='bold', color=WHITE)
ax.set_title('High Performer Exit Rate by Age Group\n("Is it really just students leaving?")',
             fontsize=13, fontweight='bold', color=WHITE, pad=12)
ax.set_ylabel('HP Exit Rate (%)', fontsize=11)
ax.set_xlabel('Age Group at Hire', fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('charts/chart09_age_exit.png', dpi=180, bbox_inches='tight', facecolor=DARK_BG)
plt.close()




# CHART 10 — Promotion Rate by Tier
# Answers: Are HPs promoted faster / more? (Guiding Q4)
promo_data = master.groupby('perf_tier')['was_promoted'].mean()*100
promo_data = promo_data.reindex(tier_order)
print(f"\nPromotion rates:\n{promo_data}")

fig, ax = plt.subplots(figsize=(9, 5), facecolor=DARK_BG)
ax.set_facecolor(NAVY_MID)
bars = ax.bar(promo_data.index, promo_data.values,
              color=[ORANGE, BLUE, GREEN], edgecolor='white', linewidth=0.5, width=0.5)
for bar, val in zip(bars, promo_data.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f'{val:.1f}%', ha='center', fontsize=14, fontweight='bold', color=WHITE)
ax.set_title('Promotion Rate by Performance Tier\n(HPs ARE promoted more — but still only 31%)',
             fontsize=13, fontweight='bold', color=WHITE, pad=12)
ax.set_ylabel('% Ever Promoted', fontsize=11)
ax.set_ylim(0, 45)
ax.grid(axis='y', alpha=0.3)
ax.annotate('69% of High Performers\nNEVER received a promotion',
            xy=(0, promo_data.iloc[0]), xytext=(1.2, 38),
            color=RED, fontsize=10, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=RED, lw=1.5))
plt.tight_layout()
plt.savefig('charts/chart10_promo_rate.png', dpi=180, bbox_inches='tight', facecolor=DARK_BG)
plt.close()




# CHART 11 — Time to Promotion by Tier (box plot)
# Answers: Does performance speed up promotion? (Guiding Q4)
promo_only = master[master['was_promoted']==1].copy()
data_promo = [promo_only[promo_only['perf_tier']==t]['months_to_promo'].dropna().values for t in tier_order]
for t, d in zip(tier_order, data_promo):
    print(f"  {t}: median={np.median(d):.1f}mo, mean={np.mean(d):.1f}mo, n={len(d)}")

fig, ax = plt.subplots(figsize=(9, 5), facecolor=DARK_BG)
ax.set_facecolor(NAVY_MID)
tier_labels = ['High\nPerformer','Medium\nPerformer','Low\nPerformer']
bp = ax.boxplot(data_promo, patch_artist=True,
                medianprops=dict(color='white', linewidth=2.5),
                whiskerprops=dict(color=GRAY), capprops=dict(color=GRAY),
                flierprops=dict(marker='o', markersize=3, alpha=0.4, color=GRAY))
for patch, color in zip(bp['boxes'], [ORANGE, BLUE, GREEN]):
    patch.set_facecolor(color); patch.set_alpha(0.8)
ax.set_xticklabels(tier_labels, fontsize=11)
for i, d in enumerate(data_promo):
    med = np.median(d)
    ax.text(i+1, med+0.3, f'{med:.1f} mo', ha='center', fontsize=10, color=WHITE, fontweight='bold')
ax.set_title("Time to First Promotion by Performance Tier\n(No meaningful difference — performance doesn't speed up promotion)",
             fontsize=13, fontweight='bold', color=WHITE, pad=12)
ax.set_ylabel('Months to First Promotion', fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('charts/chart11_time_to_promo.png', dpi=180, bbox_inches='tight', facecolor=DARK_BG)
plt.close()




# CHART 12 — Retention: Promoted vs Not Promoted
# Answers: Do promotions show higher retention? (Guiding Q5)
retention_data = {
    'HP\nPromoted'       : master[(master['is_hp']==1)&(master['was_promoted']==1)]['is_exited'].mean()*100,
    'HP\nNot Promoted'   : master[(master['is_hp']==1)&(master['was_promoted']==0)]['is_exited'].mean()*100,
    'Non-HP\nPromoted'   : master[(master['is_hp']==0)&(master['was_promoted']==1)]['is_exited'].mean()*100,
    'Non-HP\nNot Promoted': master[(master['is_hp']==0)&(master['was_promoted']==0)]['is_exited'].mean()*100,
}
print(f"\nRetention data:\n{retention_data}")

fig, ax = plt.subplots(figsize=(10, 5), facecolor=DARK_BG)
ax.set_facecolor(NAVY_MID)
bars = ax.bar(retention_data.keys(), retention_data.values(),
              color=[ORANGE, RED, BLUE, PURPLE], edgecolor='white', linewidth=0.5, width=0.55)
for bar, val in zip(bars, retention_data.values()):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f'{val:.1f}%', ha='center', fontsize=12, fontweight='bold', color=WHITE)
ax.set_title("Exit Rate: Promoted vs Not Promoted\n(Promotion alone isn't saving high performers)",
             fontsize=13, fontweight='bold', color=WHITE, pad=12)
ax.set_ylabel('Exit Rate (%)', fontsize=11)
ax.set_ylim(0, 105)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('charts/chart12_promo_retention.png', dpi=180, bbox_inches='tight', facecolor=DARK_BG)
plt.close()




# CHART 13 — Reasons for Leaving: HP vs Non-HP
# Answers: Why do HPs cite different reasons? (Guiding Q9)
hp_r   = master[master['is_hp']==1]['ReasonForLeaving'].value_counts(normalize=True)*100
nhp_r  = master[master['is_hp']==0]['ReasonForLeaving'].value_counts(normalize=True)*100
reason_order = ['Better Offer','Lack of Growth','Burntout','Insufficient Wages',
                'Poor Management','Relocation','Performance','Attendance','Policy Violation']
hp_r  = hp_r.reindex(reason_order).fillna(0)
nhp_r = nhp_r.reindex(reason_order).fillna(0)
print(f"\nHP reasons:\n{hp_r}")

fig, ax = plt.subplots(figsize=(12, 6), facecolor=DARK_BG)
ax.set_facecolor(NAVY_MID)
x = np.arange(len(reason_order)); w = 0.38
b1 = ax.bar(x-w/2, hp_r.values,  w, label='High Performers',     color=ORANGE, alpha=0.9, edgecolor='white', linewidth=0.4)
b2 = ax.bar(x+w/2, nhp_r.values, w, label='Non-High Performers', color=BLUE,   alpha=0.9, edgecolor='white', linewidth=0.4)
ax.set_xticks(x)
ax.set_xticklabels(reason_order, rotation=20, ha='right', fontsize=9)
ax.set_title('Reasons for Leaving: High Performers vs Others\n(HPs disproportionately cite "Better Offer" & "Lack of Growth")',
             fontsize=13, fontweight='bold', color=WHITE, pad=12)
ax.set_ylabel('% of Group', fontsize=11)
ax.legend(fontsize=11, facecolor=DARK_BG, edgecolor='gray', labelcolor=WHITE)
ax.grid(axis='y', alpha=0.3)
for i, (hv, nv) in enumerate(zip(hp_r.values, nhp_r.values)):
    diff = hv - nv
    if abs(diff) > 3:
        color = ORANGE if diff > 0 else GREEN
        ax.annotate(f'+{diff:.0f}%' if diff > 0 else f'{diff:.0f}%',
                    xy=(x[i]-w/2, hv), xytext=(x[i]-w/2, hv+1.5),
                    ha='center', fontsize=8, color=color, fontweight='bold')
plt.tight_layout()
plt.savefig('charts/chart13_reasons.png', dpi=180, bbox_inches='tight', facecolor=DARK_BG)
plt.close()




# CHART 14 — HP Reasons Donut
hp_reasons_clean = master[master['is_hp']==1]['ReasonForLeaving'].value_counts()
top5 = hp_reasons_clean.head(5).copy()
other_val = hp_reasons_clean.iloc[5:].sum()
if other_val > 0: top5['Other'] = other_val

fig, ax = plt.subplots(figsize=(8, 6), facecolor=DARK_BG)
ax.set_facecolor(DARK_BG)
colors_donut = [ORANGE, RED, BLUE, PURPLE, GREEN, GRAY]
wedges, texts, autotexts = ax.pie(
    top5.values, labels=top5.index, autopct='%1.1f%%',
    colors=colors_donut[:len(top5)], startangle=90,
    wedgeprops=dict(edgecolor='white', linewidth=1.5), pctdistance=0.75)
for t in texts: t.set_color(WHITE); t.set_fontsize(10)
for at in autotexts: at.set_color('white'); at.set_fontsize(9); at.set_fontweight('bold')
centre_circle = plt.Circle((0,0), 0.55, fc=DARK_BG)
ax.add_artist(centre_circle)
ax.text(0, 0, f'n={int(hp_reasons_clean.sum())}\nHP exits',
        ha='center', va='center', fontsize=11, color=WHITE, fontweight='bold')
ax.set_title('Why High Performers Leave Palm Club', fontsize=13, fontweight='bold', color=WHITE, pad=15)
plt.tight_layout()
plt.savefig('charts/chart14_hp_reasons_donut.png', dpi=180, bbox_inches='tight', facecolor=DARK_BG)
plt.close()




# CHART 15 — Part-time vs Full-time HP exit
pos_exit = master[master['is_hp']==1].groupby('position').apply(
    lambda x: pd.Series({'total':len(x),'exited':x['is_exited'].sum()})
).reset_index()
pos_exit['exit_rate'] = pos_exit['exited']/pos_exit['total']*100
print(f"\nHP exit by position:\n{pos_exit}")

fig, ax = plt.subplots(figsize=(8, 5), facecolor=DARK_BG)
ax.set_facecolor(NAVY_MID)
bars = ax.bar(pos_exit['position'], pos_exit['exit_rate'],
              color=[RED, ORANGE], edgecolor='white', linewidth=0.5, width=0.4)
for bar, val, tot in zip(bars, pos_exit['exit_rate'], pos_exit['total']):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f'{val:.1f}%\n(n={int(tot)})', ha='center', fontsize=13, fontweight='bold', color=WHITE)
ax.set_title('HP Exit Rate: Full-time vs Part-time', fontsize=13, fontweight='bold', color=WHITE, pad=12)
ax.set_ylabel('HP Exit Rate (%)', fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('charts/chart15_position.png', dpi=180, bbox_inches='tight', facecolor=DARK_BG)
plt.close()




# KEY STATS PRINTOUT — all numbers referenced in the deck
print("\n" + "="*60)
print("COMPLETE KEY STATS SUMMARY")
print("="*60)

hp   = master[master['is_hp']==1]
nhp  = master[master['is_hp']==0]

print(f"\n── WORKFORCE ──")
print(f"Total employees           : {len(master)}")
print(f"High performers (≥{HP_THRESHOLD}) : {master['is_hp'].sum()} ({master['is_hp'].mean():.0%})")
print(f"Overall exit rate         : {master['is_exited'].mean():.1%}")
print(f"HP exit rate              : {hp['is_exited'].mean():.1%}")
print(f"Non-HP exit rate          : {nhp['is_exited'].mean():.1%}")

print(f"\n── TENURE ──")
print(f"HP median tenure at exit  : {hp[hp['is_exited']==1]['tenure_months'].median():.1f} months")
print(f"NonHP median tenure exit  : {nhp[nhp['is_exited']==1]['tenure_months'].median():.1f} months")

print(f"\n── PROMOTION ──")
print(f"HP promotion rate         : {hp['was_promoted'].mean():.1%}")
print(f"Non-HP promotion rate     : {nhp['was_promoted'].mean():.1%}")
hp_promo = master[(master['is_hp']==1)&(master['was_promoted']==1)]
hp_nopromo = master[(master['is_hp']==1)&(master['was_promoted']==0)]
print(f"HP+Promoted exit rate     : {hp_promo['is_exited'].mean():.1%}")
print(f"HP+NotPromoted exit rate  : {hp_nopromo['is_exited'].mean():.1%}")
print(f"HP median months to promo : {hp[hp['was_promoted']==1]['months_to_promo'].median():.1f}")
print(f"NonHP median to promo     : {nhp[nhp['was_promoted']==1]['months_to_promo'].median():.1f}")

print(f"\n── WAGES ──")
print(f"HP on Min wage (exited)   : {(hp[hp['is_exited']==1]['wage']=='Minimum').mean():.1%}")
print(f"HP on Min wage (retained) : {(hp[hp['is_exited']==0]['wage']=='Minimum').mean():.1%}")

print(f"\n── REASONS ──")
print(f"HP Better Offer           : {hp_r.get('Better Offer',0):.1f}%")
print(f"HP Lack of Growth         : {hp_r.get('Lack of Growth',0):.1f}%")
print(f"Combined avoidable exits  : {hp_r.get('Better Offer',0)+hp_r.get('Lack of Growth',0):.1f}%")

print(f"\n── BRANCHES ──")
print(f"Worst branch HP exit rate : Downtown {hp_by_branch.iloc[0]['hp_exit_rate']:.1f}%")
print(f"Best branch HP exit rate  : UBC {hp_by_branch.iloc[-1]['hp_exit_rate']:.1f}%")
print(f"Gap                       : {hp_by_branch.iloc[0]['hp_exit_rate']-hp_by_branch.iloc[-1]['hp_exit_rate']:.1f}pp")

print("\n✅ ALL 15 CHARTS SAVED TO charts/ folder")