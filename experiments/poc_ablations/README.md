# NeuroBottleneck - Proof of Concept (PoC) Ablation Studies

## 🎯 هدف این پوشه
این ماژول برای انجام آزمایش‌های سریع (MVP) جهت اثبات تأثیر مثبت «تزریق دانش ساختاری گوموری-هو» بر روی یادگیری تقویتی (PPO) در شبکه‌های مخابراتی طراحی شده است.

## 🧪 آزمایش‌های تعریف‌شده (4 حالت Ablation)
1. **Baseline**: PPO ساده (بدون GNN، بدون ماسک)
2. **Telecom Mask**: PPO + ماسک مخابراتی (محدودیت ظرفیت فیزیکی)
3. **GNN+PPO**: PPO + استخراج ویژگی با GraphSAGE (بدون ماسک ساختاری)
4. **Neuro (Full)**: PPO + GraphSAGE + ماسک مخابراتی + ماسک گوموری-هو (نوآوری اصلی)

## 📂 ساختار فایل‌ها
- `env_poc.py` : محیط اصلی Gymnasium با منطق ماسک‌ها
- `gnn_encoder_poc.py` : مدل GraphSAGE برای تبدیل گراف به بردار ویژگی
- `run_ablation.py` : اسکریپت اصلی اجرای ۴ آزمایش با ۵ Seed مختلف
- `run_india35_analysis.py` : اسکریپت اختصاصی برای تحلیل گلوگاه‌های India35 (درخواست استاد)
- `notebooks/visualize_results.ipynb` : تحلیل بصری نتایج با نمودارهای تعاملی

## 🚀 نحوه اجرا
```bash
# ۱. نصب پیش‌نیازها
pip install -r requirements_poc.txt

# ۲. اجرای آزمایش‌های Ablation (روی گراف Abilene)
python run_ablation.py --timesteps 30000

# ۳. تحلیل گلوگاه‌های India35 برای استاد راهنما
python run_india35_analysis.py

# ۴. مشاهده نتایج در TensorBoard
tensorboard --logdir ./outputs/logs/