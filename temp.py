import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle

# 1. تحميل البيانات
data = pd.read_csv(r'C:\Users\Elbostan\Desktop\New folder (2)\breast-cancer.csv')

# 2. ترميز المتغيرات النصية (لو موجودة)
encoder = LabelEncoder()
for col in data.select_dtypes(include=['object']).columns:
    data[col] = encoder.fit_transform(data[col])

# 3. تقسيم البيانات إلى ميزات وهدف
X = data.drop(['id', 'diagnosis'], axis=1)
y = data['diagnosis']

# 4. تقسيم مجموعة البيانات إلى تدريب واختبار
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. تدريب نموذج Decision Tree
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(x_train, y_train)

# 6. تقييم النموذج
y_pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"دقة النموذج: {accuracy:.2f}")

# 7. حفظ النموذج في ملف
model_path = r'C:\Users\Elbostan\Desktop\New folder (2)\model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(clf, f)
print(f"تم حفظ النموذج في {model_path}")

# 8. قراءة النموذج المحفوظ واستخدامه للتنبؤ
with open(model_path, 'rb') as f:
    loaded_model = pickle.load(f)

# اختبار التنبؤ على بيانات الاختبار
loaded_pred = loaded_model.predict(x_test)
loaded_accuracy = accuracy_score(y_test, loaded_pred)
print(f"دقة النموذج بعد التحميل: {loaded_accuracy:.2f}")

print(X.columns.tolist())
