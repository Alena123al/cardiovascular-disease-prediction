import pandas as pd
import joblib

# загрузка моделей
preprocessor = joblib.load("models/preprocessor.pkl")
rf_model = joblib.load("models/best_random_forest.pkl")

# загрузка данных
data = pd.read_csv("data/test.csv")

# если есть ID — удаляем
if "ID" in data.columns:
    data = data.drop(columns=["ID"])

# предсказания
predictions = rf_model.predict(data)

# сохраняем результат
output = pd.DataFrame({
    "prediction": predictions
})

output.to_csv("predictions.csv", index=False)

print("Предсказания сохранены в predictions.csv")
