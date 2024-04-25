import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# داده‌های قیمت روز نفت برنت
dates = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
prices = np.array([50, 52, 55, 53, 56])

# مدل رگرسیون خطی برای پیش‌بینی قیمت آینده
model = LinearRegression()
model.fit(dates, prices)

# پیش‌بینی قیمت آینده بر اساس مدل
future_date = np.array([6]).reshape(-1, 1)
future_price = model.predict(future_date)

# ارایه پیشنهاد برای خرید در قیمت مناسب
if future_price < 60:
    print("پیشنهاد: بخرید در قیمت مناسب")
else:
    print("پیشنهاد: صبر کنید تا قیمت مناسب‌تر شود")

