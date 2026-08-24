# 🇮🇷 Persian Expense Classifier

یک پروژه‌ی یادگیری ماشین برای **تشخیص خودکار دسته‌بندی هزینه‌ها از روی توضیحات متنی فارسی** است.

هدف پروژه این است که کاربر بتواند توضیح یک هزینه را به زبان طبیعی وارد کند و مدل، دسته‌ی مناسب آن هزینه را تشخیص دهد.

برای مثال:

```text
خرید کفش ۲ میلیون تومان
        ↓
     Shopping
```

یا:

```text
اجاره خونه ۱۰ میلیون تومان
        ↓
      Housing
```

---

## 🎯 مسئله

مدل برای طبقه‌بندی توضیحات هزینه در **۷ دسته** آموزش داده شده است:

| دسته            | توضیح          |
| --------------- | -------------- |
| `entertainment` | تفریح و سرگرمی |
| `food`          | غذا و خوراک    |
| `health`        | سلامت و درمان  |
| `housing`       | مسکن           |
| `other`         | سایر هزینه‌ها  |
| `shopping`      | خرید           |
| `transport`     | حمل‌ونقل       |

بنابراین ورودی مدل یک **متن فارسی** و خروجی آن احتمال تعلق متن به هر یک از این هفت دسته است.

---

## 🧠 معماری مدل

مدل اصلی پروژه یک **CNN برای طبقه‌بندی متن (Text Classification)** است.

ابتدا متن توسط `TextVectorization` به یک دنباله‌ی عددی با طول ثابت تبدیل می‌شود. سپس این دنباله وارد شبکه‌ی عصبی می‌شود.

معماری کلی:

```text
Persian Text
     │
     ▼
Text Standardization
     │
     ▼
TextVectorization
     │
     ▼
Embedding
     │
     ▼
Conv1D
     │
     ▼
GlobalAveragePooling1D
     │
     ▼
Dense
     │
     ▼
Dense + Softmax
     │
     ▼
7 Expense Classes
```

### لایه‌های اصلی

- **TextVectorization**  
  متن را به sequence عددی تبدیل می‌کند.

- **Embedding**  
  کلمات را به بردارهای عددی با ابعاد بالاتر تبدیل می‌کند.

- **Conv1D**  
  الگوهای محلی موجود در متن را استخراج می‌کند.

- **GlobalAveragePooling1D**  
  خروجی convolution را به یک بردار ثابت تبدیل می‌کند.

- **Dense**  
  ویژگی‌های استخراج‌شده را برای classification ترکیب می‌کند.

- **Softmax**  
  احتمال هر یک از ۷ دسته را تولید می‌کند.

تنظیمات مدل و فرآیند آموزش در فایل:

```text
config/train_config.yml
```

قرار دارند.

---

## 📝 پردازش متن فارسی

قبل از ورود متن به مدل، یک مرحله‌ی استانداردسازی انجام می‌شود.

برای مثال:

```text
خرید کفش ۲,۵۰۰,۰۰۰ تومان!!!
```

به شکلی مناسب برای مدل تبدیل می‌شود و اعداد به یک token مشترک مانند `NUMBER` تبدیل می‌شوند.

این کار باعث می‌شود مدل به جای یادگیری مقدارهای عددی مختلف، بیشتر روی **معنای توضیح هزینه** تمرکز کند.

پردازش متن در بخشی از pipeline با TensorFlow انجام می‌شود تا preprocessing با مدل قابل استفاده و قابل انتقال به محیط inference باشد.

---

## 🔄 Pipeline

کل فرآیند پروژه به صورت خلاصه:

```text
Dataset
   │
   ▼
Text Preprocessing
   │
   ▼
TextVectorization
   │
   ▼
Train / Test Split
   │
   ▼
CNN Model
   │
   ▼
Training
   │
   ▼
Evaluation
   │
   ▼
Saved Model
```

---

## 📱 هدف نهایی

یکی از اهداف مهم پروژه این است که مدل آموزش‌دیده بتواند در **اپلیکیشن‌های موبایل، مخصوصاً Flutter** مورد استفاده قرار گیرد.

برای این منظور، مدل و بخش vectorization به صورت جداگانه قابل export هستند تا بتوان pipeline آموزش‌دیده را در محیطی خارج از Python نیز استفاده کرد.

معماری موردنظر برای استفاده در اپلیکیشن:

```text
Flutter App
     │
     ▼
Persian Expense Text
     │
     ▼
Text Preprocessing
     │
     ▼
ML Model
     │
     ▼
Expense Category
```

---

## 🛠️ تکنولوژی‌ها

- Python
- TensorFlow / Keras
- Scikit-learn
- NumPy
- Pandas
- Pytest
- TensorFlow Lite
- Conda

---

## 🚀 اجرا

ابتدا repository را clone کنید:

```bash
git clone https://github.com/Jafar-Rezazadeh/persian_expense_classifier.git
cd persian_expense_classifier
```

محیط Conda را ایجاد کنید:

```bash
conda env create -f requirements.yml
conda activate persian_classifier
```

سپس پروژه را نصب کنید:

```bash
pip install -e .
```

و برای اجرای pipeline:

```bash
python main.py
```

برای اجرای تست‌ها:

```bash
pytest
```

---

## 📂 ساختار کلی

```text
persian_expense_classifier/
│
├── config/                 # تنظیمات مدل و training
├── data/                   # داده‌ها و label mapping
├── src/
│   └── persian_expense_classifier/
│       ├── data/           # Data loading
│       ├── models/         # Model architectures
│       ├── preprocessing/  # Text preprocessing
│       ├── training/       # Training & evaluation
│       ├── exports/        # Model export
│       └── experiments/    # ML experiments
│
├── tests/                  # Unit tests
├── main.py                 # Entry point
├── pyproject.toml
└── requirements.yml
```

---

## 👨‍💻 توسعه‌دهنده

**Jafar Rezazadeh**

[GitHub Repository](https://github.com/Jafar-Rezazadeh/persian_expense_classifier)

---

<div align="center">

### 🇮🇷 Persian Expense Classifier

**طبقه‌بندی خودکار هزینه‌ها از روی متن فارسی**

</div>
