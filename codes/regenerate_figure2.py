import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# داده‌ی نهایی و اصلاح‌شده (بعد از حذف موردِ کاذبِ "sho"، N=268)
# ==========================================
LILIA_WORDS = 9668
MINU_WORDS = 46153

categories = [
    "ضمیر شخصی", "کلیتیک ضمیری\n(سوم‌شخص)", "ضمیر مبهم/شمول", "ضمیر پرسشی",
    "ضمیر اشاره", "اسم خاص", "اسم عام جاندار", "اسم عضو بدن",
    "اسم عام غیرجاندار", "اسم انتزاعی", "اسم خوراکی",
]
lilia_counts = [6, 1, 4, 6, 64, 2, 2, 3, 3, 0, 0]
minu_counts  = [16, 7, 13, 0, 38, 2, 11, 19, 55, 8, 7]

lilia_rates = [c / LILIA_WORDS * 1000 for c in lilia_counts]
minu_rates = [c / MINU_WORDS * 1000 for c in minu_counts]

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(13, 6.5))
bars_l = ax.bar(x - width/2, lilia_rates, width, label='Lilia (1;11-2;10)', color='tab:blue', edgecolor='black')
bars_m = ax.bar(x + width/2, minu_rates, width, label='Minu (4;0-5;2)', color='tab:red', edgecolor='black')

for bar, val in zip(bars_l, lilia_rates):
    ax.annotate(f"{val:.2f}", (bar.get_x() + bar.get_width()/2, bar.get_height()),
                textcoords="offset points", xytext=(0, 3), ha='center', fontsize=8)
for bar, val in zip(bars_m, minu_rates):
    ax.annotate(f"{val:.2f}", (bar.get_x() + bar.get_width()/2, bar.get_height()),
                textcoords="offset points", xytext=(0, 3), ha='center', fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=8.5)
ax.set_ylabel("Rate per 1,000 child word-tokens")
ax.set_title("Normalized DOM Rate by Referential Type\n(controls for corpus-size difference between Lilia and Minu, N=267)", fontsize=12)
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("dom_figure2_normalized_rate.png", dpi=150, bbox_inches='tight')
plt.show()

print(" شکل ۲ ذخیره شد: dom_figure2_normalized_rate.png")
