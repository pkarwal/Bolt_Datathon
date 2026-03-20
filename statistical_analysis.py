
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

master = pd.read_csv('master.csv')
HP_THRESHOLD = 85.44



# 1. T-TEST: Is HP exit rate significantly different from non-HP?
print("=" * 60)
print("1. T-TEST: HP vs Non-HP exit rates")
print("=" * 60)

hp_exits  = master[master['is_hp']==1]['is_exited'].values
nhp_exits = master[master['is_hp']==0]['is_exited'].values

t_stat, p_val = stats.ttest_ind(hp_exits, nhp_exits)
print(f"HP exit rate    : {hp_exits.mean():.3f}")
print(f"Non-HP exit rate: {nhp_exits.mean():.3f}")
print(f"T-statistic     : {t_stat:.4f}")
print(f"P-value         : {p_val:.4f}")
print(f"Significant (p<0.05): {'YES' if p_val < 0.05 else 'NO'}")



# 2. T-TEST: Does time-to-promotion differ by performance?
print("\n" + "=" * 60)
print("2. T-TEST: Time to promotion — HP vs non-HP")
print("=" * 60)

hp_promo  = master[(master['is_hp']==1)&(master['was_promoted']==1)]['months_to_promo'].dropna()
nhp_promo = master[(master['is_hp']==0)&(master['was_promoted']==1)]['months_to_promo'].dropna()

t2, p2 = stats.ttest_ind(hp_promo, nhp_promo)
print(f"HP  median months to promo : {hp_promo.median():.2f}")
print(f"NHP median months to promo : {nhp_promo.median():.2f}")
print(f"T-statistic                : {t2:.4f}")
print(f"P-value                    : {p2:.4f}")
print(f"Significant (p<0.05)       : {'YES' if p2 < 0.05 else 'NO'}")
print("→ If NOT significant: performance has NO effect on promotion speed")



# 3. CHI-SQUARE: Are HPs and non-HPs promoted at different rates
print("\n" + "=" * 60)
print("3. CHI-SQUARE: HP promotion rate vs non-HP")
print("=" * 60)

ct = pd.crosstab(master['is_hp'], master['was_promoted'])
print(f"Contingency table:\n{ct}")
chi2, p3, dof, expected = stats.chi2_contingency(ct)
print(f"\nChi2 statistic : {chi2:.4f}")
print(f"P-value        : {p3:.6f}")
print(f"Degrees freedom: {dof}")
print(f"Significant    : {'YES' if p3 < 0.05 else 'NO'}")



# 4. T-TEST: Tenure at exit — HPs vs non-HPs
print("\n" + "=" * 60)
print("4. T-TEST: Tenure at exit — HP vs non-HP")
print("=" * 60)

hp_tenure  = master[(master['is_hp']==1)&(master['is_exited']==1)]['tenure_months'].dropna()
nhp_tenure = master[(master['is_hp']==0)&(master['is_exited']==1)]['tenure_months'].dropna()

t4, p4 = stats.ttest_ind(hp_tenure, nhp_tenure)
print(f"HP  median tenure at exit  : {hp_tenure.median():.1f} months")
print(f"NHP median tenure at exit  : {nhp_tenure.median():.1f} months")
print(f"T-statistic                : {t4:.4f}")
print(f"P-value                    : {p4:.4f}")
print(f"Significant                : {'YES' if p4 < 0.05 else 'NO'}")



# 5. LOGISTIC REGRESSION: What predicts HP exit?
print("\n" + "=" * 60)
print("5. LOGISTIC REGRESSION: Predictors of HP exit")
print("=" * 60)

hp_data = master[master['is_hp']==1].copy()

# Encode wage
wage_map = {'Minimum':1,'Competitive':2,'Premium':3}
hp_data['wage_enc'] = hp_data['wage'].map(wage_map).fillna(2)

# Encode position
hp_data['is_parttime'] = (hp_data['position']=='part-time').astype(int)

# Encode branch (dummy — use branch number)
hp_data['branch_enc'] = hp_data['branch'].fillna(4)

# Features
features = ['avg_score','wage_enc','hours','is_parttime','was_promoted','branch_enc']
hp_model_data = hp_data[features + ['is_exited']].dropna()

X = hp_model_data[features].values
y = hp_model_data['is_exited'].values

if len(np.unique(y)) > 1:
    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X, y)
    print(f"\nLogistic Regression trained on {len(X)} HP employees")
    print(f"Features: {features}")
    print(f"\nCoefficients (positive = higher exit risk):")
    for feat, coef in zip(features, lr.coef_[0]):
        direction = '↑ exit risk' if coef > 0 else '↓ exit risk'
        print(f"  {feat:20s}: {coef:+.4f}  {direction}")
    from sklearn.metrics import accuracy_score
    y_pred = lr.predict(X)
    print(f"\nModel accuracy: {accuracy_score(y, y_pred):.1%}")
    print("Note: With 90% base exit rate, model accuracy is limited.")
    print("Coefficients are more informative than accuracy here.")
else:
    print("Insufficient variation in target variable for logistic regression.")



# 6. CORRELATION MATRIX — key numeric variables
print("\n" + "=" * 60)
print("6. CORRELATION MATRIX")
print("=" * 60)

corr_vars = master[['avg_score','wage_num','hours','was_promoted',
                     'is_exited','is_hp','tenure_months']].dropna()
corr_matrix = corr_vars.corr()
print(f"\nCorrelation with is_exited:")
print(corr_matrix['is_exited'].sort_values())

print(f"\nCorrelation with avg_score:")
print(corr_matrix['avg_score'].sort_values())



# 7. ANOVA: Does avg performance score differ across branches?
print("\n" + "=" * 60)
print("7. ONE-WAY ANOVA: Performance scores across branches")
print("=" * 60)

branch_groups = [
    master[master['branch']==b]['avg_score'].dropna().values
    for b in sorted(master['branch'].dropna().unique())
]
f_stat, p7 = stats.f_oneway(*branch_groups)
print(f"F-statistic : {f_stat:.4f}")
print(f"P-value     : {p7:.4f}")
print(f"Significant : {'YES' if p7 < 0.05 else 'NO'}")
print("→ If YES: there ARE performance differences across branches")


print("\n" + "=" * 60)
print("✅ STAGE 4 STATISTICAL ANALYSIS COMPLETE")
print("=" * 60)