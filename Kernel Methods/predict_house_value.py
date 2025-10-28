import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from colorama import Fore, Style, init

# Initialize colorama for colored text
init(autoreset=True)

print(Fore.CYAN + "\n🏡 California Housing Price Predictor (Kernel Ridge Regression)\n")

# =======================
# 1️⃣ Load dataset
# =======================
try:
    df = pd.read_csv("housing.csv")
except FileNotFoundError:
    print(Fore.RED + "❌ Error: housing.csv not found. Place it in the same folder as this script.")
    exit()

# Encode categorical column
label = LabelEncoder()
df["ocean_proximity"] = label.fit_transform(df["ocean_proximity"])

# Separate features and target
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standard scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
print(Fore.YELLOW + "🔄 Training model...")
model = KernelRidge(alpha=1.0, kernel='rbf')
model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(Fore.GREEN + f"\n✅ Model trained successfully!")
print(Fore.WHITE + f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f}")

# Ocean proximity mapping
print(Fore.CYAN + "\n🌊 Ocean Proximity Options:")
for i, name in enumerate(label.classes_):
    print(f"{i} → {name}")

print("\nNow, enter housing details below 👇\n")

# =======================
# 2️⃣ Get user input safely
# =======================
def get_float(prompt):
    while True:
        try:
            return float(input(Fore.WHITE + prompt))
        except ValueError:
            print(Fore.RED + "❌ Please enter a valid number.")

longitude = get_float("Longitude: ")
latitude = get_float("Latitude: ")
housing_median_age = get_float("Housing Median Age: ")
total_rooms = get_float("Total Rooms: ")
total_bedrooms = get_float("Total Bedrooms: ")
population = get_float("Population: ")
households = get_float("Households: ")
median_income = get_float("Median Income: ")

while True:
    ocean_proximity = input(Fore.WHITE + "Ocean Proximity (NEAR BAY, INLAND, <1H OCEAN, NEAR OCEAN, ISLAND): ").strip()
    if ocean_proximity in label.classes_:
        break
    print(Fore.RED + f"❌ Invalid value! Choose from: {', '.join(label.classes_)}")

# Convert to numeric encoding
ocean_encoded = label.transform([ocean_proximity])[0]

# Create new data for prediction
new_data = pd.DataFrame([[
    longitude, latitude, housing_median_age, total_rooms,
    total_bedrooms, population, households, median_income, ocean_encoded
]], columns=X.columns)

# Scale and predict
new_data_scaled = scaler.transform(new_data)
predicted_value = model.predict(new_data_scaled)[0]

# =======================
# 3️⃣ Display result
# =======================
print(Fore.MAGENTA + "\n---------------------------------------------")
if predicted_value < 150000:
    color = Fore.RED
elif predicted_value < 300000:
    color = Fore.YELLOW
else:
    color = Fore.GREEN

print(color + f"🏠 Predicted Median House Value: ${predicted_value:,.2f}")
print(Fore.MAGENTA + "---------------------------------------------\n")
print(Style.RESET_ALL)
