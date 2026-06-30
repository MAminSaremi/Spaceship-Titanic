# 🚀 Spaceship Titanic — Transported Prediction

پیش‌بینی اینکه آیا یک مسافر در حادثه‌ی فضاپیمای تایتانیک به بُعد دیگری منتقل (Transported) شده یا نه، با استفاده از مهندسی ویژگی پیشرفته، Imputation مبتنی بر EDA، و یک مدل Stacking ترکیبی از چند الگوریتم یادگیری ماشین.

## 📋 ویژگی‌های پروژه

### مهندسی ویژگی (Feature Engineering)
- استخراج `deck`, `num`, `side` از ستون `Cabin`
- استخراج `groupId` از `PassengerId` برای شناسایی مسافران هم‌گروه
- محاسبه‌ی `spending` (مجموع هزینه‌های `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`)
- استخراج نام خانوادگی (`Surname`) و شناسایی اعضای یک خانواده با ترکیب `groupId` + `Surname`
- ساخت ویژگی‌های مشتق‌شده: `log_spending` (تبدیل لگاریتمی به‌دلیل چولگی شدید)، `is_child` (سن ≤ ۱۲)، `group_size` (تعداد اعضای هر گروه)

### Imputation هوشمند مبتنی بر EDA
به‌جای پر کردن ساده‌ی مقادیر گمشده با میانگین/مد سراسری، این پروژه از الگوهای کشف‌شده در تحلیل اکتشافی داده استفاده می‌کند، از جمله:
- `HomePlanet`: بر اساس مد گروه (`groupId`)، سپس بر اساس شرط هم‌زمانی `VIP` و `CryoSleep`
- `CryoSleep`: بر اساس رابطه با `spending` (هیچ مسافری با هزینه‌ی مثبت در CryoSleep نبوده)
- `deck` و `side`: بر اساس مد درون هر گروه/خانواده، و سپس بر اساس ترکیب `HomePlanet` + `CryoSleep`
- `Age`: با ترکیب چند قانون شرطی (بر اساس خانواده، الگوی هزینه، و میانه‌ی گروهی `HomePlanet`/`CryoSleep`)
- `VIP`, `Destination`: بر اساس مد درون خانواده و سپس بر اساس `HomePlanet`

> ⚠️ نکته‌ی مهم: در طول فرآیند Imputation، از استفاده‌ی مستقیم متغیر هدف (`Transported`) برای پر کردن مقادیر گمشده اجتناب شده تا از **Target Leakage** جلوگیری شود (حتی در مواردی که الگوی آماری قوی هم وجود داشت).

### مدل‌سازی
- پیش‌پردازش متغیرهای کتگوریکال (`HomePlanet`, `Destination`, `deck`, `side`) با `OneHotEncoder` در یک `ColumnTransformer`
- جست‌وجوی هایپرپارامتر با `GridSearchCV` برای چهار مدل پایه:
  - CatBoost
  - XGBoost
  - Random Forest
  - LightGBM
- ترکیب مدل‌ها با `StackingClassifier` (با meta-model از نوع Random Forest)
- ارزیابی نهایی با Cross Validation (۵-Fold Stratified) برای تخمین پایدارتر از عملکرد مدل

## 📊 دیتاست

فایل مورد نیاز: `train.csv` (دیتاست استاندارد مسابقه‌ی [Spaceship Titanic در Kaggle](https://www.kaggle.com/competitions/spaceship-titanic))

ستون‌های اصلی:

| ستون | توضیح |
| --- | --- |
| `PassengerId` | شناسه‌ی مسافر (شامل گروه و شماره) |
| `HomePlanet` | سیاره‌ی مبدأ مسافر |
| `CryoSleep` | آیا مسافر در خواب سرمایی بوده |
| `Cabin` | شماره‌ی کابین (deck/num/side) |
| `Destination` | مقصد سفر |
| `Age` | سن مسافر |
| `VIP` | عضویت VIP |
| `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck` | هزینه‌های مسافر در بخش‌های مختلف فضاپیما |
| `Name` | نام کامل مسافر |
| `Transported` | متغیر هدف — آیا مسافر منتقل شده یا نه |

## ⚙️ نصب پیش‌نیازها

```bash
pip install pandas numpy scikit-learn catboost xgboost lightgbm
```

## 🚀 نحوه اجرا

1. فایل `train.csv` را در همان مسیر اسکریپت قرار دهید.
2. اسکریپت را اجرا کنید:

```bash
python main.py
```

3. مدل Stacking روی داده‌ی train آموزش داده و سپس با Cross Validation ارزیابی می‌شود.
4. خروجی شامل دقت (accuracy) هر Fold، میانگین دقت، انحراف معیار، و بازه‌ی اطمینان ۹۵٪ است.

## 📈 نتایج مدل‌ها (GridSearchCV)

| مدل | Train Accuracy | Test Accuracy | Gap |
| --- | --- | --- | --- |
| CatBoost | 77.54% | 76.02% | 0.0152 |
| XGBoost | 78.04% | 76.02% | 0.0202 |
| Random Forest | 82.84% | 76.14% | 0.0671 |
| LightGBM | 82.59% | 76.31% | 0.0628 |

### مدل نهایی: Stacking (CatBoost + XGBoost + LightGBM → Random Forest)

| معیار | مقدار |
| --- | --- |
| Train Accuracy | 79.12% |
| Test Accuracy | 76.54% |
| Gap | 0.0258 |

> 📌 از بین مدل‌های آزمایش‌شده برای meta-model، **Random Forest** بهترین عملکرد را در ترکیب نهایی داشت.
> 📌 آزمایش‌ها نشان داد تفاوت معناداری بین CatBoost با و بدون One-Hot Encoding وجود ندارد؛ با این حال برای حفظ سازگاری با چارچوب Stacking، از Pipeline شامل `OneHotEncoder` استفاده شد.

## 🛠️ تکنولوژی‌ها

- Python
- Pandas / NumPy
- Scikit-learn (Pipeline, ColumnTransformer, StackingClassifier, GridSearchCV)
- CatBoost
- XGBoost
- LightGBM

## 📁 ساختار پروژه

```
.
├── main.py        # اسکریپت اصلی پروژه (پیش‌پردازش، مهندسی ویژگی، مدل‌سازی)
├── train.csv       # دیتاست آموزش (باید جداگانه فراهم شود)
└── README.md
```

## 📝 توضیحات تکمیلی

- بخش‌های مربوط به `GridSearchCV` برای هر چهار مدل پایه در کد به‌صورت کامنت نگه داشته شده‌اند تا در صورت نیاز قابل اجرای مجدد باشند (زمان‌بر هستند)؛ بهترین پارامترهای یافت‌شده مستقیماً در تعریف هر Pipeline اعمال شده‌اند.
- پیش‌پردازش متغیرهای کتگوریکال از طریق `ColumnTransformer` داخل خود Pipeline انجام می‌شود، نه به‌صورت دستی روی دیتافریم — این رویکرد از تغییر مستقیم داده‌ی خام جلوگیری کرده و امکان اعمال یکسان تبدیل‌ها روی داده‌ی جدید (مثل `test.csv`) را فراهم می‌کند.
- ارزیابی نهایی مدل با `cross_val_score` بر مبنای ۵-Fold Stratified انجام شده تا تخمین دقت پایدارتر و کمتر وابسته به یک تقسیم‌بندی خاص از داده باشد.

## 📄 لایسنس

این پروژه برای اهداف آموزشی/شخصی ایجاد شده است.
