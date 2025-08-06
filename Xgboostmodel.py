import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score
# 1. Veriyi yükle
df = pd.read_csv("bundesliga_with_efficiency.csv")

# 2. Hedef ve giriş değişkenlerini ayır
X = df.drop(columns=["Result"])
y = df["Result"]

# 3. Veriyi eğitim ve test olarak böl
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. XGBoost modelini tanımla ve eğit
model = xgb.XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'  # çok sınıflı sınıflandırma için uygun
)
model.fit(X_train, y_train)

# 5. Tahmin yap ve başarıyı ölç
y_pred = model.predict(X_test)


# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Precision, Recall, F1-score (makro ortalama)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# Tablo
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-score'],
    'Score': [accuracy, precision, recall, f1]
})

print(metrics_df)

# Görselleştirme
sns.barplot(x='Metric', y='Score', data=metrics_df)
plt.ylim(0, 1)
plt.title("Model Başarı Metrikleri - XGBoost")
plt.show()
