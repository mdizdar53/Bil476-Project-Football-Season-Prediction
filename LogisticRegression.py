import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
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

# 4. Lojistik Regresyon modelini tanımla ve eğit
model = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=500,
    class_weight="balanced",
    random_state=42
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
plt.title("Model Başarı Metrikleri - Logistic Regression")
plt.show()

# Özellik önemleri (katsayılar)
coefficients = model.coef_
feature_names = X.columns

# Çok sınıflı olduğu için ortalama alalım
avg_importance = coefficients.mean(axis=0)

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': avg_importance
}).sort_values(by='Importance', ascending=False)

print("\n🔍 Logistic Regression - Öznitelik Önemi:")
print(importance_df)

# Görselleştirme
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
plt.title("Logistic Regression - En Önemli 15 Özellik")
plt.tight_layout()
plt.show()


