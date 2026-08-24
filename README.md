# 🇮🇷 Persian Expense Classifier

یک مدل یادگیری عمیق برای **دسته‌بندی خودکار هزینه‌ها از روی متن فارسی**.

این پروژه با استفاده از **Python، TensorFlow/Keras و پردازش متن** ساخته شده است و هدف آن این است که یک عبارت فارسی مربوط به یک هزینه را دریافت کرده و آن را در یکی از دسته‌های هزینه قرار دهد.

برای مثال:

```text
اجاره خونه 10 میلیون تومان
```

به دسته‌ی:

```text
housing
```

طبقه‌بندی می‌شود.

---

## ✨ ویژگی‌ها

- 📝 پشتیبانی از متن فارسی
- 🔢 تبدیل اعداد فارسی و انگلیسی به یک توکن مشترک (`NUM`)
- 💰 نرمال‌سازی عبارت‌های مربوط به واحد پول
- 🔤 تبدیل متن به sequence عددی با Keras Tokenizer
- 📏 Padding و محدود کردن طول ورودی به ۲۰ توکن
- 🧠 استفاده از مدل Neural Network مبتنی بر `Embedding`
- 📊 پشتیبانی از ۷ دسته‌ی مختلف هزینه
- 💾 ذخیره‌ی Tokenizer و Label Encoder
- 📱 امکان تبدیل مدل Keras به **TensorFlow Lite** برای استفاده در اپلیکیشن‌های موبایل
- 🐍 ساختار پروژه به صورت Python Package با `pyproject.toml`

---

## 🏷️ دسته‌بندی‌ها

مدل در حال حاضر هزینه‌ها را در ۷ دسته طبقه‌بندی می‌کند:

| ID | دسته | توضیح |
|---:|---|---|
| `0` | `entertainment` | تفریح و سرگرمی |
| `1` | `food` | غذا و خوراک |
| `2` | `health` | سلامت و درمان |
| `3` | `housing` | مسکن و هزینه‌های مربوط به آن |
| `4` | `other` | سایر هزینه‌ها |
| `5` | `shopping` | خرید |
| `6` | `transport` | حمل‌ونقل |

---

## 🧠 معماری مدل

مدل اصلی در فایل `custom_model.py` تعریف شده و از یک معماری نسبتاً سبک برای طبقه‌بندی متن استفاده می‌کند.

```text
Input
  │
  ▼
Embedding
  │
  ▼
GlobalAveragePooling1D
  │
  ▼
Dense(16, ReLU)
  │
  ▼
Dense(7, Softmax)
  │
  ▼
Predicted Expense Category
```

جزئیات مدل:

- `Embedding`
  - vocabulary size: `1000`
  - embedding dimension: `32`
  - maximum sequence length: `20`
- `GlobalAveragePooling1D`
- `Dense(16, activation="relu")`
- `Dense(7, activation="softmax")`
- Optimizer: `Adam`
- Loss: `SparseCategoricalCrossentropy`
- Metric: `Accuracy`

---

## 🔄 جریان پردازش داده

Pipeline اصلی پروژه به شکل زیر است:

```text
Persian Text
     │
     ▼
Text Normalization
     │
     ├── تبدیل حروف به lowercase
     ├── تبدیل اعداد فارسی/انگلیسی به NUM
     ├── تبدیل «تومن» به «تومان»
     └── حذف فاصله‌های اضافی
     │
     ▼
Keras Tokenizer
     │
     ▼
Integer Sequences
     │
     ▼
Padding / Truncation
     │
     ▼
Embedding
     │
     ▼
Neural Network
     │
     ▼
Expense Category
```

در preprocessing، اعداد فارسی و انگلیسی با عبارت `NUM` جایگزین می‌شوند و عبارت `تومن` نیز به `تومان` نرمال می‌شود.

Tokenizer نیز با vocabulary حداکثر ۱۰۰۰ کلمه ساخته شده و sequenceها به طول ۲۰ محدود می‌شوند؛ در صورت کوتاه بودن متن، padding و در صورت طولانی بودن، truncation انجام می‌شود.

---

## 📂 ساختار پروژه

ساختار فعلی repository به صورت کلی به شکل زیر است:

```text
persian_expense_classifier/
│
├── build/
│   └── ...
│
├── data/
│   ├── raw/
│   └── labels.json
│
├── src/
│   └── persian_expense_classifier/
│       │
│       ├── data/
│       │   └── data_loader.py
│       │
│       ├── preprocessing/
│       │   ├── tokenizer_and_padder.py
│       │   └── custom_label_encoder.py
│       │
│       └── models/
│           └── custom_model.py
│
├── .gitignore
├── main.py
├── pipline.md
├── pyproject.toml
└── requirements.yml
```

پروژه از ساختار `src` برای Python Package استفاده می‌کند و `setuptools` نیز در `pyproject.toml` برای پیدا کردن packageها از پوشه‌ی `src` تنظیم شده است.

---

## ⚙️ نیازمندی‌ها

محیط Conda پروژه در `requirements.yml` تعریف شده است و شامل موارد زیر است:

- Python `3.10.18`
- NumPy
- Pandas
- Scikit-learn
- TensorFlow
- Keras
- Matplotlib

---

## 🚀 نصب و راه‌اندازی

ابتدا repository را clone کنید:

```bash
git clone https://github.com/Jafar-Rezazadeh/persian_expense_classifier.git

cd persian_expense_classifier
```

### ساخت محیط Conda

```bash
conda env create -f requirements.yml
```

سپس محیط را فعال کنید:

```bash
conda activate persian_classifier
```

### نصب پروژه

برای نصب package در حالت development:

```bash
pip install -e .
```

---

## 🏋️ آموزش مدل

فرآیند آموزش در `main.py` انجام می‌شود.

ابتدا داده‌ها بارگذاری شده و سپس به مجموعه‌های train و test تقسیم می‌شوند. تقسیم داده‌ها با `stratify` انجام می‌شود تا توزیع کلاس‌ها حفظ شود.

برای اجرای pipeline:

```bash
python main.py
```

در فرآیند آموزش:

1. داده‌ها بارگذاری می‌شوند.
2. داده‌ها به train و test تقسیم می‌شوند.
3. متن‌ها normalize می‌شوند.
4. Tokenizer روی داده‌های آموزشی ساخته می‌شود.
5. متن‌ها به sequence عددی تبدیل می‌شوند.
6. sequenceها padding می‌شوند.
7. labelها encode می‌شوند.
8. مدل ساخته و آموزش داده می‌شود.
9. مدل روی داده‌های test ارزیابی می‌شود.
10. در صورت نیاز مدل به TensorFlow Lite تبدیل می‌شود.

---

## 🔤 Tokenizer

برای تبدیل متن فارسی به داده‌ای که مدل بتواند پردازش کند، از Keras `Tokenizer` استفاده شده است.

Tokenizer با این تنظیمات ساخته می‌شود:

```python
Tokenizer(
    num_words=1000,
    oov_token="<OOV>"
)
```

و حداکثر طول ورودی مدل:

```python
maxLen = 20
```

است.

Tokenizer قابلیت ذخیره شدن به صورت JSON را نیز دارد:

```text
build/tokenizer.json
```

که برای استفاده‌ی مجدد در زمان inference بسیار مهم است.

---

## 🏷️ Label Encoder

برای تبدیل labelهای متنی به مقادیر عددی از `LabelEncoder` استفاده شده است.

برای مثال:

```text
entertainment → 0
food          → 1
health        → 2
housing       → 3
other         → 4
shopping      → 5
transport     → 6
```

Label mapping نیز در:

```text
data/labels.json
```

نگهداری می‌شود.

همچنین encoder امکان ذخیره‌ی mapping تولیدشده در:

```text
build/labels.json
```

را دارد.

---

## 📱 خروجی TensorFlow Lite

یکی از اهداف پروژه امکان استفاده از مدل در محیط‌های mobile مانند Android و Flutter است.

مدل Keras با استفاده از `TFLiteConverter` به TensorFlow Lite تبدیل می‌شود و با optimization پیش‌فرض TensorFlow Lite ساخته می‌شود. خروجی در مسیر زیر قرار می‌گیرد:

```text
build/expense_classifier_model.tflite
```



بنابراین معماری کلی برای استفاده در یک اپلیکیشن می‌تواند به صورت زیر باشد:

```text
                   Python / Training
                         │
                         ▼
                  Train Keras Model
                         │
                         ▼
                expense_classifier
                         │
                         ▼
                TensorFlow Lite
                         │
                         ▼
                Flutter / Android
```

> نکته: برای inference در محیطی مانند Flutter، علاوه بر فایل `.tflite` باید دقیقاً همان preprocessing، tokenizer/vocabulary و mapping مربوط به labelها نیز حفظ شود.

---

## 🧪 نمونه ورودی

یک ورودی متنی فارسی مانند:

```text
اجاره خونه 10 میلیون تومان
```

ابتدا normalize می‌شود:

```text
اجاره خونه NUM میلیون تومان
```

سپس توسط tokenizer به sequence عددی تبدیل شده و با padding به طول مورد نیاز مدل می‌رسد.

در نهایت مدل احتمال هر یک از ۷ کلاس را محاسبه کرده و کلاس با بیشترین احتمال به عنوان نتیجه انتخاب می‌شود.

---

## 📊 ارزیابی

مدل دارای متد `evaluate` است و می‌توان با استفاده از داده‌های test عملکرد آن را بررسی کرد:

```python
model.evaluate(xTestPad, yTestEnc)
```

معیار اصلی فعلی:

```text
Accuracy
```

است. مدل با `SparseCategoricalCrossentropy` آموزش داده می‌شود.

---

## 🛠️ تکنولوژی‌های استفاده‌شده

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-ML-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?logo=keras)
![NumPy](https://img.shields.io/badge/NumPy-Data-blue?logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-purple?logo=pandas)
![Scikit Learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![TFLite](https://img.shields.io/badge/TensorFlow%20Lite-Mobile-lightgrey)

</div>

---

## 🎯 هدف پروژه

هدف اصلی این پروژه ساخت یک مدل سبک برای **تشخیص خودکار نوع هزینه از روی توضیحات فارسی تراکنش‌ها** است.

چنین مدلی می‌تواند در برنامه‌های مدیریت مالی شخصی استفاده شود تا کاربر بدون انتخاب دستی دسته‌بندی، تنها توضیح تراکنش را وارد کند و سیستم به صورت خودکار دسته‌ی مناسب را تشخیص دهد.

برای مثال:

```text
"خرید کفش ۲ میلیون"
        ↓
shopping

"ویزیت دکتر ۵۰۰ هزار"
        ↓
health

"بنزین ۳۰۰ هزار"
        ↓
transport

"خرید پیتزا ۴۰۰ هزار"
        ↓
food
```

---

## 🔮 ایده‌های توسعه

برخی مسیرهای مناسب برای توسعه‌ی پروژه:

- [ ] افزایش حجم و تنوع dataset
- [ ] بهبود preprocessing مخصوص زبان فارسی
- [ ] پشتیبانی بهتر از نیم‌فاصله و حروف عربی/فارسی
- [ ] استفاده از `TextVectorization`
- [ ] بررسی مدل‌های CNN و RNN برای متن
- [ ] بررسی مدل‌های Transformer و BERT فارسی
- [ ] اضافه کردن confidence score به prediction
- [ ] اضافه کردن تست‌های جامع برای pipeline
- [ ] بهینه‌سازی مدل برای موبایل
- [ ] ایجاد API برای inference
- [ ] استفاده مستقیم از مدل TFLite در Flutter
- [ ] benchmark کردن مدل روی دستگاه‌های موبایل

---

## 🤝 مشارکت

اگر پیشنهادی برای بهبود مدل، preprocessing، dataset یا معماری دارید، می‌توانید:

1. Repository را fork کنید.
2. یک branch جدید ایجاد کنید.
3. تغییرات خود را اعمال کنید.
4. یک Pull Request ارسال کنید.

همچنین برای گزارش bug یا پیشنهاد قابلیت جدید می‌توانید از بخش Issues استفاده کنید.

---

## 📄 مجوز

در حال حاضر repository فایل License مشخصی ندارد. بنابراین قبل از استفاده‌ی تجاری یا انتشار مجدد کد، شرایط استفاده از پروژه را با صاحب repository بررسی کنید.

---

## 👨‍💻 توسعه‌دهنده

ساخته‌شده توسط **Jafar Rezazadeh**

GitHub:

[Jafar-Rezazadeh/persian_expense_classifier](https://github.com/Jafar-Rezazadeh/persian_expense_classifier?utm_source=chatgpt.com)

---

<div align="center">

### 🇮🇷 ساخته‌شده برای پردازش و درک بهتر متن فارسی

⭐ اگر پروژه برایتان مفید بود، خوشحال می‌شوم با یک Star از آن حمایت کنید.

</div>