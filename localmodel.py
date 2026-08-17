import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Load the local CSV files
print("Loading datasets")
weather_path = r"C:\Users\jacey\Downloads\all_datasets(1)\full_weather_places.csv"
air_quality_path = r"C:\Users\jacey\Downloads\all_datasets(1)\air_quality.csv"

df_weather = pd.read_csv(weather_path)
df_air = pd.read_csv(air_quality_path)

# Clean and cast types
df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])
df_weather['temperature'] = pd.to_numeric(df_weather['temperature'], errors='coerce')
df_weather['humidity'] = pd.to_numeric(df_weather['humidity'], errors='coerce')
df_weather['wind_speed'] = pd.to_numeric(df_weather['wind_speed'], errors='coerce')

df_air['datetime'] = pd.to_datetime(df_air['datetime'])
df_air['pollutant_value'] = pd.to_numeric(df_air['pollutant_value'], errors='coerce')
start_time = time.time()

# 2. Define the Scaling Logic
MULTIPLIER = 2
print(f"Scaling datasets by {MULTIPLIER}x in memory")

def scale_dataset(df, time_col, numeric_cols, multiplier):
    scaled_chunks = []
    for i in range(multiplier):
        df_copy = df.copy()
        # Shift datetime forward by i months
        df_copy[time_col] = df_copy[time_col] + pd.DateOffset(months=i)
        
        # Inject 2% statistical noise into numeric features
        for col in numeric_cols:
            noise = np.random.normal(1.0, 0.02, size=len(df_copy))
            df_copy[col] = df_copy[col] * noise
            
        scaled_chunks.append(df_copy)
    return pd.concat(scaled_chunks, ignore_index=True)

# Scale Weather
weather_features = ['temperature', 'humidity', 'wind_speed']
df_weather_large = scale_dataset(df_weather, 'datetime', weather_features, MULTIPLIER)

# Scale Air Quality
df_air_large = scale_dataset(df_air, 'datetime', ['pollutant_value'], MULTIPLIER)

# 3. SAVE THE SCALED DATA FIRST
print("Saving scaled datasets to disk...")
large_weather_out = r"C:\Users\jacey\Downloads\all_datasets(1)\weather_large_local.csv"
large_air_out = r"C:\Users\jacey\Downloads\all_datasets(1)\air_quality_large_local.csv"

df_weather_large.to_csv(large_weather_out, index=False)
df_air_large.to_csv(large_air_out, index=False)
print("Data saved successfully!\n")

# Free up memory before starting the actual benchmark
del df_weather
del df_air
del df_weather_large
del df_air_large


# 4. Load the Massive Datasets from Disk
df_w_large = pd.read_csv(large_weather_out)
df_a_large = pd.read_csv(large_air_out)

df_w_large['datetime'] = pd.to_datetime(df_w_large['datetime'])
df_a_large['datetime'] = pd.to_datetime(df_a_large['datetime'])

# 5. Perform the Massive In-Memory Join

df_joined = pd.merge(
    df_w_large, 
    df_a_large, 
    on=['datetime', 'place'], 
    how='inner',
    suffixes=('_w', '_a')
)

# pollutant stays normal, but pollutant_value uses the _a suffix
cols_to_keep = ['datetime', 'place', 'temperature', 'humidity', 'wind_speed', 'pollutant', 'pollutant_value_a']
df_joined = df_joined[cols_to_keep].dropna()

# Rename ONLY pollutant_value_a back to normal so the ML steps don't break
df_joined = df_joined.rename(columns={
    'pollutant_value_a': 'pollutant_value'
})

print(f"Total Rows for ML Training: {len(df_joined):,}")
# 6. Feature Engineering
# Convert categorical 'pollutant' string into a numerical code
df_joined['pollutant_idx'] = df_joined['pollutant'].astype('category').cat.codes

# Define feature matrix (X) and target label (y)
features = ['temperature', 'humidity', 'wind_speed', 'pollutant_idx']
X = df_joined[features]
y = df_joined['pollutant_value']

# 7. Train-Test Split (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Train the Random Forest Model locally using Scikit-Learn
rf_model = RandomForestRegressor(
    n_estimators=20, 
    max_depth=10, 
    random_state=42, 
    n_jobs=-1  # -1 tells Scikit-Learn to use all available CPU cores
)
rf_model.fit(X_train, y_train)

# 9. Evaluate Algorithm Performance

predictions = rf_model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)


end_time = time.time()
elapsed_seconds = end_time - start_time

print(f"\nNon big data metrics")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Squared Error (MSE):       {mse:.4f}")
print(f"Mean Absolute Error (MAE):      {mae:.4f}")
print(f"R-Squared (R2) Score:           {r2:.4f}")
print(f"Total Execution Time (Scale + Join + Train): {elapsed_seconds:.2f} seconds ({elapsed_seconds / 60:.2f} minutes)")

# 10. Audit Sample Predictions
results_df = df_joined.iloc[X_test.index][['datetime', 'place', 'pollutant']].copy()
results_df['label'] = y_test
results_df['prediction'] = predictions

print("\nSample Model Predictions:")
print(results_df.head(5).to_string(index=False))