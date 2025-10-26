#CODE BY BHAVESH G
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data/energy_data.csv")


df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")
df = df.astype({"household_id":"int16", "consumption_kwh":"float32", "temperature_celsius":"float32", "occupancy":"int8"})
df.index = df["timestamp"]

df["consumption_kwh"] = df["consumption_kwh"].ffill()
df["temperature_celsius"] = df["temperature_celsius"].fillna(df["temperature_celsius"].mean())

df["hour_of_day"]= df["timestamp"].dt.hour
df["peak_hours"] = (df["hour_of_day"]>=18) & (df["hour_of_day"]<=22)


off_peak = ((df['hour_of_day'] >= 0) & (df['hour_of_day'] < 6)) | (df['hour_of_day'] >= 22)
normal = (df['hour_of_day'] >= 6) & (df['hour_of_day'] < 18)
peak = (df['hour_of_day'] >= 18) & (df['hour_of_day'] < 22)
conditions = [off_peak, normal, peak]
rates = [5, 7, 10]

df["tariff_inr"] = np.select(conditions, rates, default=7)
df["cost_inr"] = df["consumption_kwh"] * df["tariff_inr"]
df["cost_inr"] = df["cost_inr"].astype("float32")
df = df.drop(columns=["tariff_inr"])

df["consumption_per_degree"] = df["consumption_kwh"]/df["temperature_celsius"]


df["consumption_kwh"] = df["consumption_kwh"]*0.95
df2 = df.groupby("household_id")["consumption_kwh"].agg(["mean", "median", "std"]).reset_index()
df2.columns = ["household_id","Mean", "Median", "Standard Deviation"]
df2.index = df2["household_id"]
df["consumption_kwh"] = df["consumption_kwh"]/0.95

df2.to_csv("data/Statistical_Analysis.csv", index=False)


dftop5 = df2.sort_values(by="Mean", ascending=False).head(5).copy()
dftop5 = dftop5.drop(columns=["Median", "Standard Deviation"])
dftop5["Monthy_cost"] = dftop5["Mean"]*30
print("Monthly cost of Highest Daily Average Consumers:\n\n", dftop5)


coeffs = np.corrcoef(df["consumption_kwh"],df["temperature_celsius"])[0][1]
print("\n\nCorrelation coefficient between temperature_celsius and consumption_kwh is", coeffs)


#Calculates %percentage increase from off-peak and peak hours
peak2 = df[df['hour_of_day'].between(18, 21)]['consumption_kwh'].sum()
off_peak2 = df[(df['hour_of_day'] < 6) | (df['hour_of_day'] >= 22)]['consumption_kwh'].sum()
percent_increase = ((peak2 - off_peak2) / off_peak2) * 100
print("\nPerncentage Increase between peak hours and off-peak hours is", percent_increase,"\n\n")


daily_mean = df.groupby("household_id")["consumption_kwh"].resample("D").mean().reset_index()

daily_mean.columns = ["household_id","Date","Daily Average Consumption"]
daily_mean['7_day_rolling_avg'] = (
    daily_mean.groupby('household_id')['Daily Average Consumption']
    .transform(lambda x: x.rolling(window=7, min_periods=1).mean())
)
print(daily_mean)

df['day_of_week'] = df['timestamp'].dt.dayofweek
day_avg = df.groupby('day_of_week')['consumption_kwh'].mean()
max_day = day_avg.idxmax()
Days = np.array(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
print("\n\nDay with highest average consumption:", Days[max_day])
print("Average consumption on that day:", day_avg[max_day], "\n\n")


avg_consumption_by_occupancy = df.groupby("occupancy")["consumption_kwh"].mean().reset_index()
avg_consumption_by_occupancy.columns = ["Occupancy", "Average Consumption"]
occupancy_map = {0: "Vacant", 1: "Occupied"}
avg_consumption_by_occupancy["Occupancy"] = avg_consumption_by_occupancy["Occupancy"].replace(occupancy_map)
print("Vacant Vs Occupied Average Consumption is\n", avg_consumption_by_occupancy)



df2["Inefficient"] = df2["Mean"] > (1.5 * df2["Median"])
inefficient_households = df2[df2["Inefficient"]]["Inefficient"]
print("\n\nNumber of Households with Energy inefficiency are",inefficient_households.count(),".\n")


hourly_avg = df.groupby('hour_of_day')['consumption_kwh'].mean()
colors = ["red" if (18 <= hour < 22) else "blue" for hour in hourly_avg.index]

plt.figure(figsize=(12, 6))
plt.plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2)
for i, (hour, val) in enumerate(hourly_avg.items()):
    plt.scatter(hour, val, color=colors[i], s=100, zorder=5)
plt.xlabel("Hour of Day")
plt.ylabel("Average Consumption (kWh)")
plt.title("Average Hourly Consumption Pattern (Peak Hours in Red)")
plt.grid(True, alpha=0.3)
plt.xticks(range(0, 24))
plt.tight_layout()
plt.savefig("figs/hourly_consumption_pattern.png")
plt.show()

df["day_of_week"] = df["timestamp"].dt.dayofweek
heatmap_data = df.groupby(["day_of_week", "hour_of_day"])["consumption_kwh"].mean().unstack()

plt.figure(figsize=(14, 6))
plt.imshow(heatmap_data, cmap="YlOrRd", aspect="auto")
plt.colorbar(label="Average Consumption (kWh)")
plt.xlabel("Hour of Day")
plt.ylabel("Day of Week")
plt.yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
plt.xticks(range(24))
plt.title("Heatmap: Average Consumption by Day of Week and Hour of Day")
plt.tight_layout()
plt.savefig("figs/consumption_heatmap.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(df["temperature_celsius"], df["consumption_kwh"], alpha=0.3, s=10)

z = np.polyfit(df["temperature_celsius"].dropna(), 
df.loc[df["temperature_celsius"].notna(), "consumption_kwh"], 1)
p = np.poly1d(z)
temp_range = np.linspace(df["temperature_celsius"].min(), df["temperature_celsius"].max(), 100)
plt.plot(temp_range, p(temp_range), "r-", linewidth=2, label=f"Best fit: y={z[0]:.3f}x+{z[1]:.3f}")

plt.xlabel("Temperature (°C)")
plt.ylabel("Consumption (kWh)")
plt.title("Relationship between Temperature and Consumption")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figs/temp_vs_consumption_scatter.png")
plt.show()

df["is_weekend"] = df["day_of_week"].isin([5, 6])
df["day_type"] = df["is_weekend"].map({True: "Weekend", False: "Weekday"})


df.boxplot(column="consumption_kwh", by="day_type", figsize=(10, 6))
plt.suptitle('')
plt.title("Consumption Distribution: Weekdays vs Weekends")
plt.xlabel("Day Type")
plt.ylabel("Consumption (kWh)")
plt.tight_layout()
plt.savefig("figs/weekday_vs_weekend_boxplot.png")
plt.show()


peak_consumption = df[df["hour_of_day"].between(18, 21)]["consumption_kwh"].sum()
peak_cost = df[df["hour_of_day"].between(18, 21)]["cost_inr"].sum()

shifted_consumption = peak_consumption * 0.30
original_cost_of_shifted = shifted_consumption * 10  
new_cost_of_shifted = shifted_consumption * 5
savings = original_cost_of_shifted - new_cost_of_shifted

print(f"Potential Savings from Shifting 30% Peak Consumption to Off-Peak:")
print(f"  - Shifted consumption: {shifted_consumption:.2f} kWh")
print(f"  - Potential savings: ₹{savings:.2f}")
print(f"  - Percentage savings: {(savings/peak_cost)*100:.2f}%\n")

df["temp_bin"] = pd.cut(df["temperature_celsius"], bins=10)
temp_efficiency = df.groupby("temp_bin", observed=False)["consumption_kwh"].mean()
most_efficient_temp_range = temp_efficiency.idxmin()

efficient_temps = df[df["consumption_kwh"] < df["consumption_kwh"].quantile(0.25)]["temperature_celsius"]
optimal_temp_range = (np.percentile(efficient_temps, 25), np.percentile(efficient_temps, 75))

print(f"Optimal Temperature Range (Most Efficient Consumption):")
print(f"  - Temperature range: {optimal_temp_range[0]:.2f}°C to {optimal_temp_range[1]:.2f}°C")
print(f"  - This range corresponds to the lowest 25% of consumption values\n")

total_energy = df["consumption_kwh"].sum()
total_cost = df["cost_inr"].sum()
num_households = df["household_id"].nunique()
avg_cost_per_household = total_cost / num_households
co2_emissions = total_energy * 0.82


print("SUMMARY REPORT")
print(f"Total Energy Consumed: {total_energy:.2f} kWh")
print(f"Total Cost: ₹{total_cost:.2f}")
print(f"Number of Households: {num_households}")
print(f"Average Cost per Household: ₹{avg_cost_per_household:.2f}")
print(f"Total CO₂ Emissions: {co2_emissions:.2f} kg CO₂")


#df.to_csv("file.csv")`
