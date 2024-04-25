import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# تولید داده‌های مصنوعی برای مثال
np.random.seed(42)
dates = pd.date_range('20230101', periods=100)
prices = np.random.randint(100, 200, size=100)
volatility = np.random.uniform(0.5, 2.0, size=100)

data = pd.DataFrame({'Date': dates, 'Price': prices, 'Volatility': volatility})
data['Implied_Volatility'] = data['Volatility'] + np.random.normal(0, 0.1, size=100)

# محاسبه تفاوت ولاتیلیته ضمنی و واقعی
data['Delta_Volatility'] = data['Implied_Volatility'] - data['Volatility']

# تقسیم داده‌ها به داده‌های آموزشی و آزمون
X = data[['Price', 'Delta_Volatility']]
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# اعمال مدل رگرسیون خطی
model = LinearRegression()
model.fit(X_train, y_train)

# پیش‌بینی قیمت‌های سهام
predictions = model.predict(X_test)

# نمایش نمودار پیش‌بینی‌ها
plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test)), y_test, label='Actual Prices')
plt.plot(range(len(predictions)), predictions, label='Predicted Prices')
plt.xlabel('Index')
plt.ylabel('Price')
plt.title('Actual vs. Predicted Stock Prices')
plt.legend()
plt.show()

plt.savefig("stock_price_plot.png")
print("نمودار ذخیره شد.")
