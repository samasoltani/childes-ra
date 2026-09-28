# -*- coding: utf-8 -*-
"""
گام ۵: برازش مدل رگرسیون لجستیک با خطای استاندارد خوشه‌ای
(GEE با خوشه‌بندی بر اساس جمله، برای کنترل عدم استقلال مشاهدات)
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

df = pd.read_csv("perdt_objects_categorized.csv")
print(f" مجموع رکوردها: {len(df)}")
print(f"   نشانه‌گذاری‌شده: {df['label'].sum()} | بی‌نشان: {(df['label']==0).sum()}\n")

# ============================================================
# مدل ۱: رگرسیون لجستیک معمولی (fixed-effects, برای مقایسه)
# ============================================================
model_fixed = smf.logit(
    "label ~ C(category, Treatment(reference='inanimate_noun'))",
    data=df
).fit(disp=0)

print("=" * 70)
print(" مدل ۱: رگرسیون لجستیک معمولی (بدون کنترل خوشه‌بندی)")
print("=" * 70)
print(model_fixed.summary())

# ============================================================
# مدل ۲: GEE با خطای استاندارد خوشه‌ای (بر اساس sent_id)
# ============================================================
df["sent_id_code"] = df["sent_id"].astype("category").cat.codes

model_gee = smf.gee(
    "label ~ C(category, Treatment(reference='inanimate_noun'))",
    groups="sent_id_code",
    data=df,
    family=sm.families.Binomial(),
    cov_struct=sm.cov_struct.Independence()
).fit()

print("\n" + "=" * 70)
print(" مدل ۲: GEE با خطای استاندارد خوشه‌ای (کنترل عدم استقلال جمله‌ای)")
print("=" * 70)
print(model_gee.summary())

# ============================================================
# جدول Odds Ratio نهایی (بر اساس مدل GEE - مدل اصلیِ گزارش‌شونده)
# ============================================================
print("\n" + "=" * 70)
print(" جدولِ Odds Ratio نهایی (مدل GEE، برای گزارش در مقاله)")
print("=" * 70)

conf = model_gee.conf_int()
conf.columns = ["CI_lower", "CI_upper"]
conf["coef"] = model_gee.params
conf["OR"] = np.exp(conf["coef"])
conf["CI_lower_OR"] = np.exp(conf["CI_lower"])
conf["CI_upper_OR"] = np.exp(conf["CI_upper"])
conf["p_value"] = model_gee.pvalues

print(f"\n{'متغیر':<55}{'OR':<10}{'CI 95%':<24}{'p-value':<10}")
print("-" * 100)
for idx in conf.index:
    label = idx.replace("C(category, Treatment(reference='inanimate_noun'))[T.", "دسته=").replace("]", "")
    or_val = conf.loc[idx, "OR"]
    ci_l = conf.loc[idx, "CI_lower_OR"]
    ci_u = conf.loc[idx, "CI_upper_OR"]
    p = conf.loc[idx, "p_value"]
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "(n.s.)"))
    print(f"{label:<55}{or_val:<10.3f}[{ci_l:.3f}, {ci_u:.3f}]{'':<8}{p:<10.4f}{sig}")

print("\n گام ۵ کامل شد.")