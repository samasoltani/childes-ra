import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# داده‌ی نهایی و اصلاح‌شده (بعد از حذف موردِ کاذبِ "sho"، N=268)
# ==========================================
categories = [
    "ضمیر شخصی", "کلیتیک ضمیری\n(سوم‌شخص)", "ضمیر مبهم/شمول", "ضمیر پرسشی",
    "ضمیر اشاره", "اسم خاص", "اسم عام جاندار", "اسم عضو بدن",
    "اسم عام غیرجاندار", "اسم انتزاعی", "اسم خوراکی",
]
lilia_counts = [6, 1, 4, 6, 64, 2, 2, 3, 3, 0, 0]
minu_counts  = [16, 7, 13, 0, 38, 2, 11, 19, 55, 8, 7]

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(13, 6.5))
bars_l = ax.bar(x - width/2, lilia_counts, width, label='Lilia (1;11-2;10)', color='tab:blue', edgecolor='black')
bars_m = ax.bar(x + width/2, minu_counts, width, label='Minu (4;0-5;2)', color='tab:red', edgecolor='black')

for bar, val in zip(bars_l, lilia_counts):
    ax.annotate(str(val), (bar.get_x() + bar.get_width()/2, bar.get_height()),
                textcoords="offset points", xytext=(0, 3), ha='center', fontsize=8)
for bar, val in zip(bars_m, minu_counts):
    ax.annotate(str(val), (bar.get_x() + bar.get_width()/2, bar.get_height()),
                textcoords="offset points", xytext=(0, 3), ha='center', fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=8.5)
ax.set_ylabel("Number of marked tokens")
ax.set_title("Differential Object Marking (-ro/-o) by Referential Type\n(Persian Child Speech, Lilia + Minu Corpora, N=267)", fontsize=12)
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("dom_figure1_raw_counts.png", dpi=150, bbox_inches='tight')
plt.show()

print(" شکل ۱ ذخیره شد: dom_figure1_raw_counts.png")
