# Customer Churn Prediction

Bu layihədə telekom şirkətinin müştəri datası üzərində maşın öyrənməsi modeli qurub, hansı müştərinin xidmətdən imtina edə biləcəyini (churn) proqnozlaşdırmağa çalışdım. Məqsəd data ilə əvvəldən sona qədər işləməyi (yükləmə, təmizləmə, model qurma, yoxlama) öyrənmək idi.

Data IBM-in açıq Telco Customer Churn dataseti (Kaggle-də də var) - 7043 müştəri, hər biri üçün 21 xüsusiyyət (yaş, kontrakt növü, aylıq ödəniş və s).

## Qovluq strukturu

- `data/` - orijinal CSV fayl
- `notebooks/customer_churn_analysis.ipynb` - əsas iş burdadır, data analizi və qrafiklər
- `src/preprocessing.py` - datanı təmizləyən və modelə hazır edən kod
- `src/train.py` - modeli təlim edən skript
- `src/predict.py` - hazır modellə yeni data üzərində proqnoz vermək üçün
- `models/` - train.py işlədikdən sonra model buraya yazılır

## Nə etdim

Əvvəlcə datanı araşdırdım - `TotalCharges` sütununda boş dəyərlər var idi, onları doldurdum. Kateqorik sütunları (gender, contract növü və s) modelin başa düşəcəyi rəqəmlərə çevirdim (one-hot encoding).

Sonra 3 fərqli model sınadım ki, hansı daha yaxşı nəticə verir:
- Logistic Regression
- Random Forest  
- XGBoost

Hər birini GridSearchCV ilə tuning etdim (ən yaxşı parametrləri tapmaq üçün). Data balanslı deyildi (churn edənlər ~26%), ona görə təkcə accuracy-ə deyil, ROC-AUC və recall-a da baxdım.

## Nəticələr

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.805 | 0.655 | 0.559 | 0.603 | 0.841 |
| Random Forest | 0.795 | 0.672 | 0.444 | 0.535 | 0.843 |
| XGBoost | 0.801 | 0.659 | 0.516 | 0.579 | 0.844 |

Üçü də bir-birinə yaxın çıxdı, XGBoost bir az öndə oldu ona görə onu final model kimi saxladım.

Ən çox təsir edən amillər: müqavilə nə qədər qısamüddətlidirsə (month-to-month), müştəri nə qədər az müddətdir xidmətdədirsə (tenure), o qədər çox churn ehtimalı var idi. Bu məntiqli də görünür.

## İşlətmək üçün

```bash
pip install -r requirements.txt
python src/train.py
python src/predict.py data/Telco-Customer-Churn.csv
```

Notebook üçün Jupyter və ya VS Code + Jupyter extension lazımdır.

## İstifadə olunanlar

Python, pandas, scikit-learn, XGBoost, matplotlib, seaborn
