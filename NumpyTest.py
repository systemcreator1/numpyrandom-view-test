# Step 1: Install Necessary Libraries
!pip install pandas numpy scikit-learn matplotlib

# Step 2: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Step 3: Simulate IoT Data
def generate_iot_data(num_readings):
    timestamps = pd.date_range(start='2024-10-01', periods=num_readings, freq='H')

    # Simulating different sensor readings
    temperature = np.random.normal(loc=22.0, scale=2.0, size=num_readings)  # Temperature in °C
    humidity = np.random.normal(loc=50.0, scale=5.0, size=num_readings)      # Humidity in %
    radiation = np.random.uniform(low=0.0, high=100.0, size=num_readings)   # Radiation in µSv/h
    pressure = np.random.normal(loc=1013.0, scale=10.0, size=num_readings)  # Pressure in hPa

    return pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temperature,
        'humidity': humidity,
        'radiation': radiation,
        'pressure': pressure
    })

# Generate 100 readings
iot_data = generate_iot_data(100)
print(iot_data.head())

# Step 4: Visualize the Data
plt.figure(figsize=(15, 10))

# Temperature
plt.subplot(2, 2, 1)
plt.plot(iot_data['timestamp'], iot_data['temperature'], label='Temperature', color='blue')
plt.title('Temperature Readings')
plt.xlabel('Timestamp')
plt.ylabel('Temperature (°C)')
plt.grid()

# Humidity
plt.subplot(2, 2, 2)
plt.plot(iot_data['timestamp'], iot_data['humidity'], label='Humidity', color='green')
plt.title('Humidity Readings')
plt.xlabel('Timestamp')
plt.ylabel('Humidity (%)')
plt.grid()

# Radiation
plt.subplot(2, 2, 3)
plt.plot(iot_data['timestamp'], iot_data['radiation'], label='Radiation', color='orange')
plt.title('Radiation Readings')
plt.xlabel('Timestamp')
plt.ylabel('Radiation (µSv/h)')
plt.grid()

# Pressure
plt.subplot(2, 2, 4)
plt.plot(iot_data['timestamp'], iot_data['pressure'], label='Pressure', color='purple')
plt.title('Pressure Readings')
plt.xlabel('Timestamp')
plt.ylabel('Pressure (hPa)')
plt.grid()

plt.tight_layout()
plt.show()

# Step 5: Build Machine Learning Model
# Prepare the data for modeling
X = iot_data[['humidity', 'pressure']]
y = iot_data['temperature']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
score = model.score(X_test, y_test)
print(f'Model R^2 Score: {score:.2f}')

# Step 6: Make Predictions
# Simulate future readings
future_humidity = np.random.normal(loc=50.0, scale=5.0, size=24)  # Next 24 hours
future_pressure = np.random.normal(loc=1013.0, scale=10.0, size=24)

future_X = pd.DataFrame({'humidity': future_humidity, 'pressure': future_pressure})
predicted_temperatures = model.predict(future_X)

# Display predictions
future_data = pd.DataFrame({'predicted_temperature': predicted_temperatures})
print(future_data)
