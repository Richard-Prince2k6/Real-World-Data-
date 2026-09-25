"""End-to-end analysis of the UCI Online Retail transaction dataset."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Online_Retail.xlsx"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")

raw = pd.read_excel(DATA)
raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"])
raw["Revenue"] = raw["Quantity"] * raw["UnitPrice"]
cancelled = raw["InvoiceNo"].astype(str).str.startswith("C")
valid = (~cancelled) & raw["CustomerID"].notna() & (raw["Quantity"] > 0) & (raw["UnitPrice"] > 0)
sales = raw.loc[valid].copy()
sales["Date"] = sales["InvoiceDate"].dt.normalize()
sales["Month"] = sales["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
sales["LineRevenue"] = sales["Quantity"] * sales["UnitPrice"]
daily = sales.groupby("Date")["LineRevenue"].sum().asfreq("D", fill_value=0)

# KPI and customer summaries
invoice = sales.groupby("InvoiceNo").agg(revenue=("LineRevenue", "sum"), customer=("CustomerID", "first"), date=("InvoiceDate", "min"))
customer = sales.groupby("CustomerID").agg(revenue=("LineRevenue", "sum"), orders=("InvoiceNo", "nunique"), last_purchase=("InvoiceDate", "max"))
snapshot = sales["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
customer["recency_days"] = (snapshot - customer["last_purchase"].dt.normalize()).dt.days
customer["segment"] = "Regular"
customer.loc[(customer["orders"] >= 3) & (customer["revenue"] >= customer["revenue"].quantile(.75)), "segment"] = "Champions"
customer.loc[(customer["orders"] == 1), "segment"] = "One-time"
monthly = sales.groupby("Month")["LineRevenue"].sum()
country = sales.groupby("Country")["LineRevenue"].sum().sort_values(ascending=False)
products = sales.groupby("Description")["LineRevenue"].sum().sort_values(ascending=False).head(10)

# Time-aware daily revenue model: hold out final 20% of dates. Lag and rolling features use past values only.
features = pd.DataFrame(index=daily.index)
for lag in (1, 7, 14):
    features[f"lag_{lag}"] = daily.shift(lag)
features["rolling_mean_7"] = daily.shift(1).rolling(7).mean()
features["day_of_week"] = daily.index.dayofweek
features["month"] = daily.index.month
model_data = features.join(daily.rename("target")).dropna()
split = int(len(model_data) * .8)
train, test = model_data.iloc[:split], model_data.iloc[split:]
model = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, max_features=.8, random_state=42, n_jobs=-1)
model.fit(train.drop(columns="target"), train.target)
pred = model.predict(test.drop(columns="target"))
baseline = test["lag_7"].to_numpy()
metrics = {
    "holdout_days": int(len(test)),
    "model_mae": float(mean_absolute_error(test.target, pred)),
    "weekly_naive_mae": float(mean_absolute_error(test.target, baseline)),
    "model_rmse": float(np.sqrt(mean_squared_error(test.target, pred))),
    "model_mape_nonzero_days_pct": float((np.abs(test.target.to_numpy()[test.target.to_numpy() > 0] - pred[test.target.to_numpy() > 0]) / test.target.to_numpy()[test.target.to_numpy() > 0]).mean() * 100),
}

# Charts
fig, ax = plt.subplots(figsize=(11, 4.8))
monthly.plot(ax=ax, marker="o", color="#176B87")
ax.set(title="Monthly net sales (positive, non-cancelled lines)", xlabel="Month", ylabel="Sales (dataset currency units)")
fig.tight_layout(); fig.savefig(FIG / "monthly_sales.png", dpi=180); plt.close(fig)

fig, ax = plt.subplots(figsize=(11, 4.8))
ax.plot(daily.index, daily.rolling(7).mean(), color="#176B87", label="7-day moving average")
ax.set(title="Daily sales trend", xlabel="Date", ylabel="Sales (dataset currency units)")
ax.legend(); fig.tight_layout(); fig.savefig(FIG / "daily_sales.png", dpi=180); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
country.head(10).sort_values().plot.barh(ax=ax, color="#64A6BD")
ax.set(title="Top 10 countries by net sales", xlabel="Sales (dataset currency units)", ylabel="")
fig.tight_layout(); fig.savefig(FIG / "country_sales.png", dpi=180); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
products.sort_values().plot.barh(ax=ax, color="#F4A261")
ax.set(title="Top 10 products by net sales", xlabel="Sales (dataset currency units)", ylabel="")
fig.tight_layout(); fig.savefig(FIG / "top_products.png", dpi=180); plt.close(fig)

fig, ax = plt.subplots(figsize=(11, 4.8))
ax.plot(test.index, test.target, label="Actual", color="#176B87", linewidth=1.8)
ax.plot(test.index, pred, label="Random forest", color="#E76F51", alpha=.85)
ax.plot(test.index, baseline, label="7-day naive baseline", color="#6C757D", alpha=.75)
ax.set(title="Daily revenue forecast: chronological holdout", xlabel="Date", ylabel="Sales (dataset currency units)")
ax.legend(); fig.tight_layout(); fig.savefig(FIG / "forecast_holdout.png", dpi=180); plt.close(fig)

summary = {
    "raw_rows": int(len(raw)), "cancelled_invoice_lines": int(cancelled.sum()),
    "valid_sales_lines": int(len(sales)), "missing_customer_id_rows": int(raw.CustomerID.isna().sum()),
    "date_start": str(sales.InvoiceDate.min().date()), "date_end": str(sales.InvoiceDate.max().date()),
    "unique_customers": int(sales.CustomerID.nunique()), "unique_orders": int(sales.InvoiceNo.nunique()),
    "countries": int(sales.Country.nunique()), "net_sales": float(sales.LineRevenue.sum()),
    "average_order_value": float(invoice.revenue.mean()), "median_order_value": float(invoice.revenue.median()),
    "repeat_customer_share_pct": float((customer.orders >= 2).mean() * 100),
    "top_country": str(country.index[0]), "top_country_sales_share_pct": float(country.iloc[0] / country.sum() * 100),
    "top_month": str(monthly.idxmax().strftime("%B %Y")), "top_month_sales": float(monthly.max()),
    "customer_segments": {str(k): int(v) for k, v in customer.segment.value_counts().items()},
    "forecast_metrics": metrics,
    "top_countries": {str(k): float(v) for k, v in country.head(10).items()},
    "top_products": {str(k): float(v) for k, v in products.items()},
}
(ROOT / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
print(f"\nCharts saved to {FIG}")
