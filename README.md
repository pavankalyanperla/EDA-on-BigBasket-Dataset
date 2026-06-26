# EDA on BigBasket Dataset

An exploratory data analysis and price-prediction project on BigBasket, India's largest online grocery platform. The project examines ~27,500 products across categories ranging from Beauty & Hygiene to Gourmet & World Food, uncovering trends in pricing, discounts, and customer ratings — and closes with four regression models that predict sale price from product features.

---

## Dataset

| Attribute | Detail |
|---|---|
| Source | BigBasket product catalog |
| Rows | 27,555 (raw); 27,439 after cleaning |
| Columns | 10 |
| Format | CSV (~16 MB) |

**Column descriptions:**

| Column | Type | Description |
|---|---|---|
| `index` | int | Row identifier |
| `product` | str | Product name |
| `category` | str | Top-level category (e.g. Beauty & Hygiene) |
| `sub_category` | str | More specific grouping within category |
| `brand` | str | Manufacturer / brand |
| `sale_price` | float | Price at which product is sold (INR) |
| `market_price` | float | Original listed / MRP price (INR) |
| `type` | str | Product type (e.g. Hair Oil & Serum) |
| `rating` | float | Customer rating (0–5 scale) |
| `description` | str | Full product description |

---

## Folder Structure

```
EDA-on-BigBasket-Dataset/
├── notebooks/
│   └── EDA on BigBasket DataSet.ipynb   # Main analysis notebook
├── Dataset/
│   └── BigBasket Products.csv           # Raw dataset (16 MB)
├── docs/
│   └── BigBasket EDA Final Report end.pdf
├── EDA PROJECT.pptx                     # Presentation slides (15 slides)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Key EDA Findings

All findings below are drawn directly from notebook outputs and markdown cells.

- **Missing values:** Four columns (`product`, `brand`, `rating`, `description`) contained null values. Rows with null `product` or `brand` were dropped. Rows where both `rating` and `description` were null simultaneously were dropped. Remaining null ratings were imputed with the column mean.

- **Engineered features:** Two new columns were derived — `diff_in_prices` (market price minus sale price) and `discount_percentage` ((diff / market price) × 100).

- **Outliers:** Boxplots confirmed outliers exist in `sale_price`, `market_price`, and `diff_in_prices`. They were intentionally retained as they represent real high-priced products, not data errors.

- **Sale price distribution:** The `sale_price` histogram is approximately bell-shaped (normal distribution).

- **Price relationship:** Scatter plots show that sale price and market price are strongly positively correlated. Many products cluster on the diagonal, indicating sale price ≈ market price (zero or near-zero discount). Points below the diagonal represent discounted products.

- **Category analysis:** Revenue contribution varies significantly by category. Correlation heatmap confirms high correlation between `sale_price` and `market_price`; `rating` and `discount_percentage` are weakly correlated with price.

- **Advanced EDA:** PCA reduces the numerical feature space to 2 components; t-SNE and K-Means clustering (k=3 on PCA space) reveal product groupings by category.

---

## Models

Target variable: `sale_price`  
Features: `market_price`, `category`, `sub_category`, `brand`, `type`, `rating`, `discount_percentage`  
Train/test split: 80/20 (`random_state=42`)  
Preprocessing: `StandardScaler` on numeric features + `OneHotEncoder` on categorical features, via `ColumnTransformer` + `Pipeline`

| Model | R² Score | RMSE (INR) |
|---|---|---|
| Linear Regression | 0.9729 | 80.46 |
| Random Forest | **0.9990** | **15.83** |
| Gradient Boosting | 0.9987 | 17.86 |
| XGBoost | 0.9914 | 45.34 |

**Random Forest** achieves the best performance (R² = 0.999, RMSE = ₹15.83). The high R² values across all models reflect that `market_price` is the dominant predictor — for most products the sale price is close to or equal to the market price.

---

## Setup & Running

**1. Clone the repo**
```bash
git clone https://github.com/pavankalyanperla/EDA-on-BigBasket-Dataset.git
cd EDA-on-BigBasket-Dataset
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Launch Jupyter and open the notebook**
```bash
jupyter notebook notebooks/EDA\ on\ BigBasket\ DataSet.ipynb
```
Or open Jupyter Lab and navigate to `notebooks/EDA on BigBasket DataSet.ipynb`.

> The notebook loads the dataset with a path relative to its own location: `../Dataset/BigBasket Products.csv`. As long as you open it from within the `notebooks/` folder (the default when launching `jupyter notebook` from the repo root), the path resolves correctly.

---

## Possible Next Steps

- **Hyperparameter tuning:** Grid search or Bayesian optimization for Random Forest/XGBoost (n_estimators, max_depth, learning_rate).
- **Text features:** Use TF-IDF or sentence embeddings on the `description` column to extract additional predictive signal.
- **Discount prediction:** Reframe the target as `discount_percentage` and treat it as a classification problem (Low / High / No discount).
- **Elbow / Silhouette analysis:** Determine the optimal K for the K-Means clustering step instead of hardcoding k=3.
- **Deployment:** Wrap the Random Forest pipeline in a Flask / FastAPI endpoint so price estimates can be served programmatically.
- **More recent data:** The current catalog reflects a snapshot in time; re-running on a live scrape would test whether the patterns hold.
