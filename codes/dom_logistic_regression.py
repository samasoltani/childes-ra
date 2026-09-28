import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# ==========================================
# گام ۱: بارگذاری همان دیتاستِ قبلی (marked/unmarked + دسته + کودک)
# ==========================================
with open('dom_neural_dataset.json', encoding='utf-8') as f:
    dataset = json.load(f)

df = pd.DataFrame(dataset)[['category', 'child', 'label']]
print(f" مجموع نمونه‌ها (پیش از فیلتر): {len(df)}")

# ==========================================
# فیلتر: فقط هفت دسته‌ای که صورتِ بی‌نشانِ قابل‌شمارش دارند
# (همان هفت دسته‌ی بخشِ ۴.۵؛ چهار دستهٔ دیگر، چون هرگز نمونهٔ بی‌نشان برایشان
#  جمع‌آوری نشده (۱۰۰٪ نشانه‌گذاری‌شده)، باعثِ تفکیکِ کاملِ آماری می‌شوند)
# ==========================================
VALID_CATEGORIES = ['pronoun', 'demonstrative', 'animate_noun', 'body_part_noun',
                     'inanimate_noun', 'abstract_noun', 'food_noun']
excluded = set(df['category'].unique()) - set(VALID_CATEGORIES)
print(f"⚠️ دسته‌های حذف‌شده (فاقدِ نمونه‌ی بی‌نشان): {excluded}")

df = df[df['category'].isin(VALID_CATEGORIES)].copy()
print(f" مجموع نمونه‌ها (پس از فیلتر): {len(df)}")
print(f"   نشانه‌گذاری‌شده: {df['label'].sum()} | بی‌نشان: {(df['label']==0).sum()}\n")

print("توزیع نمونه‌ها بر اساس دسته:")
print(df.groupby('category')['label'].agg(['sum', 'count']))
print()

# ==========================================
# گام ۲: آماده‌سازی متغیرها
# دسته‌ی پایه (reference) را «اسم عام غیرجاندار» انتخاب می‌کنیم
# (یک دستهٔ بزرگ و از نظر نظری کم‌پروتوتایپی، مناسب برای مقایسه)
# کودک را هم به‌صورتِ عددی (Lilia=0, Minu=1) کدگذاری می‌کنیم
# ==========================================
df['child_bin'] = (df['child'] == 'Minu').astype(int)

# ==========================================
# گام ۳: مدلِ رگرسیونِ لجستیک (اثرات اصلیِ دسته + کودک، بدون تعامل)
# ==========================================
model = smf.logit(
    "label ~ C(category, Treatment(reference='inanimate_noun')) + child_bin",
    data=df
).fit(disp=0)

print("=" * 70)
print(" خلاصه‌ی کاملِ مدل (برای بررسیِ خودتان):")
print("=" * 70)
print(model.summary())

# ==========================================
# گام ۴: تبدیل ضرایب به Odds Ratio با فاصلهٔ اطمینان ۹۵٪
# ==========================================
print("\n" + "=" * 70)
print(" جدولِ Odds Ratio (برای گزارش در مقاله)")
print("=" * 70)

conf = model.conf_int()
conf['OR'] = model.params
conf.columns = ['CI_lower', 'CI_upper', 'coef']
conf['OR'] = np.exp(conf['coef'])
conf['CI_lower_OR'] = np.exp(conf['CI_lower'])
conf['CI_upper_OR'] = np.exp(conf['CI_upper'])
conf['p_value'] = model.pvalues

print(f"\n{'متغیر':<55}{'OR':<10}{'CI 95%':<20}{'p-value':<10}")
print("-" * 95)
for idx in conf.index:
    label = idx.replace("C(category, Treatment(reference='inanimate_noun'))[T.", "دسته=").replace("]", "")
    or_val = conf.loc[idx, 'OR']
    ci_l = conf.loc[idx, 'CI_lower_OR']
    ci_u = conf.loc[idx, 'CI_upper_OR']
    p = conf.loc[idx, 'p_value']
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    print(f"{label:<55}{or_val:<10.3f}[{ci_l:.3f}, {ci_u:.3f}]{'':<5}{p:<10.4f}{sig}")

# ==========================================
# گام ۵: کیفیتِ برازشِ کلیِ مدل
# ==========================================
print(f"\n McFadden's pseudo R²: {model.prsquared:.3f}")
print(f" Log-Likelihood: {model.llf:.2f}  (مدل صفر: {model.llnull:.2f})")
lr_stat = 2 * (model.llf - model.llnull)
from scipy.stats import chi2
lr_p = chi2.sf(lr_stat, model.df_model)
print(f" آزمونِ نسبتِ درست‌نمایی (کلِ مدل در برابرِ مدلِ صفر): LR χ²({model.df_model:.0f}) = {lr_stat:.2f}, p = {lr_p:.6f}")

print("\n گام ۱ کامل شد.")
