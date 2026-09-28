# -*- coding: utf-8 -*-
"""
تحلیل حساسیت: بررسی اینکه آیا OR پایینِ pronoun و demonstrative
محصول یک ریشه‌ی پرتکرار (mæn / in) است یا یک الگوی واقعی دسته‌ای
"""
import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

with open('dom_neural_dataset.json', encoding='utf-8') as f:
    dataset = json.load(f)

df_all = pd.DataFrame(dataset)

VALID_CATEGORIES = ['pronoun', 'demonstrative', 'animate_noun', 'body_part_noun',
                     'inanimate_noun', 'abstract_noun', 'food_noun']
df_all = df_all[df_all['category'].isin(VALID_CATEGORIES)].copy()
df_all['child_bin'] = (df_all['child'] == 'Minu').astype(int)


def fit_and_report(data, label, targets=("pronoun", "demonstrative")):
    model = smf.logit(
        "label ~ C(category, Treatment(reference='inanimate_noun')) + child_bin",
        data=data
    ).fit(disp=0)

    conf = model.conf_int()
    conf.columns = ['lo', 'hi']
    conf['OR'] = np.exp(model.params)
    conf['lo'] = np.exp(conf['lo'])
    conf['hi'] = np.exp(conf['hi'])
    conf['p'] = model.pvalues

    print(f"\n=== {label} ===")
    print(f"    (n کل={len(data)}, نشانه‌گذاری‌شده={data['label'].sum()})")
    for t in targets:
        key = f"C(category, Treatment(reference='inanimate_noun'))[T.{t}]"
        if key in conf.index:
            row = conf.loc[key]
            print(f"  {t:15s} OR={row['OR']:.3f}  CI=[{row['lo']:.3f}, {row['hi']:.3f}]  p={row['p']:.4f}")
    return model


# ============================================================
# مدل ۱: دیتای کامل (برای راستی‌آزمایی - باید با خروجی قبلی یکی باشد)
# ============================================================
fit_and_report(df_all, "مدل اصلی (کامل)")

# ============================================================
# مدل ۲: حذف کامل ریشه‌ی 'mæn' از pronoun (نشانه‌گذاری‌شده + بی‌نشان)
# ============================================================
mask_man = (df_all['category'] == 'pronoun') & (df_all['root'] == 'mæn')
print(f"\nتعداد کل رخدادهای ریشه‌ی 'mæn' که حذف می‌شود: {mask_man.sum()}")
df_no_man = df_all[~mask_man].copy()
fit_and_report(df_no_man, "حذف کامل ریشه‌ی 'mæn'")

# ============================================================
# مدل ۳: حذف کامل ریشه‌ی 'in' از demonstrative (نشانه‌گذاری‌شده + بی‌نشان)
# ============================================================
mask_in = (df_all['category'] == 'demonstrative') & (df_all['root'] == 'in')
print(f"\nتعداد کل رخدادهای ریشه‌ی 'in' که حذف می‌شود: {mask_in.sum()}")
df_no_in = df_all[~mask_in].copy()
fit_and_report(df_no_in, "حذف کامل ریشه‌ی 'in'")