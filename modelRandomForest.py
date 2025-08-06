import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
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

# 4. Random Forest modelini tanımla ve eğit
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# 5. Tahmin yap ve başarıyı ölç
y_pred = model.predict(X_test)


# Mevcut accuracy
accuracy = accuracy_score(y_test, y_pred)

# Precision, Recall ve F1-score (makro ortalama kullanıyoruz)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# Tablolaştır
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-score'],
    'Score': [accuracy, precision, recall, f1]
})

print(metrics_df)

# İsteğe bağlı görselleştirme
sns.barplot(x='Metric', y='Score', data=metrics_df)
plt.ylim(0, 1)
plt.title("Model Başarı Metrikleri - Random Forest")
plt.show()

# 6. Özniteliklerin önem derecelerini al
importances = model.feature_importances_
feature_names = X.columns

# 7. DataFrame olarak sırala
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\n🔍 Özniteliklerin Maç Sonucuna Etkisi:")
print(importance_df)

# 8. Görselleştirme
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
plt.title("Random Forest - En Önemli 15 Özellik")
plt.xlabel("Özellik Önemi")
plt.ylabel("Özellik")
plt.tight_layout()
plt.show()



