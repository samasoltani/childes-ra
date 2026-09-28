# -*- coding: utf-8 -*-
"""
گام ۲: استخراج تمام مفعول‌ها (deprel='obj') از پیکره‌ی PerDT
و تعیین وضعیت نشانه‌گذاری (آیا با "را" همراه است یا نه)
"""
import conllu
import pandas as pd
import glob
import os

# مسیر پوشه‌ای که فایل‌های .conllu در آن هستند
CORPUS_DIR = "UD_Persian-PerDT-master"

records = []

conllu_files = glob.glob(os.path.join(CORPUS_DIR, "*.conllu"))
print(f" تعداد فایل‌های پیدا‌شده: {len(conllu_files)}")
for f in conllu_files:
    print("   -", os.path.basename(f))

sent_counter = 0

for filepath in conllu_files:
    split_name = os.path.basename(filepath)  # train/dev/test را از نام فایل نگه می‌داریم
    with open(filepath, encoding="utf-8") as f:
        data = f.read()

    sentences = conllu.parse(data)

    for sent in sentences:
        sent_counter += 1
        sent_id = f"{split_name}_{sent_counter}"

        # ایندکس‌کردن توکن‌ها بر اساس id برای پیدا کردن فرزندان (head)
        tokens_by_id = {tok["id"]: tok for tok in sent if isinstance(tok["id"], int)}

        for tok in sent:
            if not isinstance(tok["id"], int):
                continue  # رد کردن multi-word tokenها

            if tok["deprel"] == "obj":
                # بررسی: آیا این مفعول فرزندی با deprel='case' و متنِ "را"/"رو" دارد؟
                marked = 0
                case_form = None
                for other in sent:
                    if not isinstance(other["id"], int):
                        continue
                    if other["head"] == tok["id"] and other["deprel"] == "case":
                        if other["form"] in ("را", "رو", "رو‌"):
                            marked = 1
                            case_form = other["form"]

                records.append({
                    "sent_id": sent_id,
                    "split": split_name,
                    "token_id": tok["id"],
                    "form": tok["form"],
                    "lemma": tok["lemma"],
                    "upos": tok["upos"],
                    "xpos": tok.get("xpos"),
                    "head_lemma": tokens_by_id.get(tok["head"], {}).get("lemma") if tok["head"] else None,
                    "label": marked,
                    "case_form": case_form,
                })

df = pd.DataFrame(records)
print(f"\n مجموع مفعول‌های پیدا‌شده: {len(df)}")
print(f"   نشانه‌گذاری‌شده (را/رو): {df['label'].sum()}")
print(f"   بی‌نشان: {(df['label']==0).sum()}")

df.to_csv("perdt_objects.csv", index=False, encoding="utf-8-sig")
print("\n فایل perdt_objects.csv ذخیره شد.")

# جدول فراوانی لماها (برای گام ۳)
freq = df.groupby(["lemma", "upos"]).agg(
    total=("label", "count"),
    marked=("label", "sum")
).reset_index().sort_values("total", ascending=False)

freq["marked_rate"] = (freq["marked"] / freq["total"] * 100).round(1)
freq.to_csv("perdt_lemma_frequency.csv", index=False, encoding="utf-8-sig")

print(f"\n فایل perdt_lemma_frequency.csv ذخیره شد ({len(freq)} لمای منحصربه‌فرد).")
print("\nنمونه‌ی ۲۰ لمای پربسامدترین:")
print(freq.head(20).to_string(index=False))

# چند آماره‌ی کلی برای اطمینان
cum_coverage = freq["total"].cumsum() / freq["total"].sum() * 100
top300_coverage = cum_coverage.iloc[299] if len(freq) > 300 else cum_coverage.iloc[-1]
print(f"\n📊 پوششِ ۳۰۰ لمای پربسامدترین: {top300_coverage:.1f}٪ از کل رخدادها")