import os
import glob
import re
from collections import Counter

CORPUS_PATHS = {
    "Lilia": r"C:\Users\sama\PycharmProjects\PythonProject\farsi_corpus_extracted\Lilia",
    "Minu": r"C:\Users\sama\PycharmProjects\PythonProject\farsi_corpus_extracted\Minu",
}

stop_words = {'xxx', 'laugh', 'singing', 'yell', 'whine', 'giggle', 'crying', 'yyy'}


def extract_age_months(lines):
    for line in lines:
        if line.startswith('@ID:') and '|CHI|' in line:
            fields = line.strip().split('|')
            for field in fields:
                m = re.match(r'^(\d+);(\d+)\.?(\d*)$', field.strip())
                if m:
                    return int(m.group(1)) * 12 + int(m.group(2))
    return None


def extract_chi_utterances(lines):
    """هم متنِ خام (برای بافت) و هم واژه‌های پاک‌سازی‌شده (برای جست‌وجو) را برمی‌گرداند."""
    utterances = []
    for raw_line in lines:
        line = raw_line.strip()
        if line.startswith('*CHI:') or line.startswith('CHI:'):
            raw_context = re.sub(r'^\*?CHI:\s*', '', line).strip()
            raw_context = re.sub(r'\d+_\d+\s*$', '', raw_context).strip()

            text = raw_context
            text = re.sub(r'&~\w+', '', text)
            text = re.sub(r'@\w+', '', text)
            text = re.sub(r'<.*?>', '', text)
            text = re.sub(r'\[.*?\]', '', text)
            text_for_words = re.sub(r'[?!.,:;\"_\-]', ' ', text)

            words = [
                w.lower()
                for w in re.findall(r'[a-zA-ZæøåæɑɒɛɪʊʌəŋʃʒʧʤɣxχRq]+', text_for_words)
                if w.lower() not in stop_words and len(w) > 1
            ]
            if words:
                utterances.append({'words': words, 'context': raw_context})
    return utterances


all_records = []
for child_name, path in CORPUS_PATHS.items():
    if not os.path.isdir(path):
        print(f"⚠️ مسیر {child_name} پیدا نشد: {path}")
        continue
    for filepath in glob.glob(os.path.join(path, "*.cha")):
        with open(filepath, 'r', encoding='utf-8-sig', errors='ignore') as f:
            lines = f.readlines()
        age = extract_age_months(lines)
        utterances = extract_chi_utterances(lines)
        if age is not None:
            for u in utterances:
                all_records.append({'child': child_name, 'age_months': age,
                                     'words': u['words'], 'context': u['context']})

print(f" مجموع گفته‌های معتبر: {len(all_records)}\n")

# ==========================================
# جست‌وجوی صورتِ رسمی و جداگانه‌ی «را»
# چند صورتِ محتمل را با هم چک می‌کنیم تا مطمئن شویم چیزی از قلم نمی‌افتد
# ==========================================
CANDIDATE_FORMS = ['ra', 'ræ', 'raa']

found_rows = []
total_word_counts = Counter()

for rec in all_records:
    for idx, w in enumerate(rec['words']):
        total_word_counts[w] += 1
        if w in CANDIDATE_FORMS:
            found_rows.append({
                'child': rec['child'],
                'age_months': rec['age_months'],
                'form': w,
                'position': idx,
                'context': rec['context'],
            })

print(f" نتیجه‌ی جست‌وجوی صورت‌های محتمل {CANDIDATE_FORMS}:\n")
for form in CANDIDATE_FORMS:
    count = total_word_counts.get(form, 0)
    print(f"   صورتِ «{form}»: {count} رخداد در کل واژگان")

print(f"\n مجموع کل رخدادهای یافت‌شده (هر سه صورت): {len(found_rows)}\n")

if found_rows:
    print(" نمونه‌ی بافت‌های واقعی (حداکثر ۳۰ مورد اول)، برای بررسیِ دستی:\n")
    for row in found_rows[:30]:
        print(f"   [{row['child']}, {row['age_months']} ماهگی] «{row['form']}» در جایگاه {row['position']}: {row['context']}")

    child_counts = Counter(r['child'] for r in found_rows)
    print(f"\n به تفکیک کودک: {dict(child_counts)}")
else:
    print(" هیچ رخدادی از صورتِ جداگانه‌ی «را» در گفتار کودک پیدا نشد.")
    print("   این خودش یک یافته‌ی گزارش‌پذیر است: یعنی کودک منحصراً از صورتِ محاوره‌ای")
    print("   (-ro/-o) استفاده کرده، نه صورتِ رسمی.")

print("\n جست‌وجوی اکتشافی کامل شد.")
