# REPO AUDIT — EDA-on-BigBasket-Dataset
**GitHub URL:** https://github.com/pavankalyanperla/EDA-on-BigBasket-Dataset  
**Audit Date:** 2026-06-26  
**Auditor:** Claude Code (read-only reconnaissance — no files were modified)  
**Purpose:** Final-year CS portfolio review

---

## 1. FULL DIRECTORY TREE

```
EDA-on-BigBasket-Dataset/                                   (repo root)
│
├── README.md                                               295 bytes
├── EDA PROJECT.pptx                                        733,068 bytes  (716 KB)
│
├── Dataset/
│   └── BigBasket Products.csv                              16,739,247 bytes  (16.0 MB)
│
├── Docs or Reports/
│   └── BigBasket EDA Final Report end.pdf                  1,833,372 bytes  (1.74 MB)
│
└── Script or Codefile/
    ├── EDA on BigBasket DataSet.ipynb                      13,063,159 bytes  (12.5 MB)
    ├── EDA on BigBasket DataSet.py                         27,469 bytes      (nbconvert — generated during audit)
    ├── EDA on BigBasket DataSet.md                         5,672,344 bytes   (nbconvert — generated during audit)
    └── EDA on BigBasket DataSet_files/                     directory of embedded plot images (nbconvert output)

TOTAL: 4 meaningful project files + README
```

---

## 2. NOTEBOOK: `Script or Codefile/EDA on BigBasket DataSet.ipynb`

### 2a. Summary Statistics

| Metric | Value |
|---|---|
| File size | 13,063,159 bytes (12.5 MB, large due to embedded plot images) |
| Total cells | 157 |
| Code cells | 117 |
| Markdown cells | 40 |
| Raw cells | 0 |
| Empty code cells | **15** (positions 47, 48, 50, 107, 110, 113, 115, 134, 139, 141, 143, 145, 147, 153, 157) |
| Cells with runtime errors in stored output | **0** |
| Max execution count in stored outputs | 106 |
| Has markdown explanation cells | YES (40 cells, though several are incomplete — see Section 6) |

### 2b. Converted Output Files

Both conversions were run during this audit and saved alongside the original in `Script or Codefile/`:

- `EDA on BigBasket DataSet.py` — Python script (27 KB)
- `EDA on BigBasket DataSet.md` — Markdown with inline base64 images (5.4 MB)

### 2c. Full Code — All Code Cells (in order)

> Cells numbered by sequential position (1-based) in the notebook. `[In N]` = stored execution count.

---

**Cell #1 [In 1] — Imports**
```python
import numpy as np #mathematical calculations
import pandas as pd # cleaning exploring and manupulation of data
import matplotlib.pyplot as plt # it is used for data visualisation which is a ML lib
import seaborn as sns #data viz library used to show attractive statistical graphs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
```

**Cell #2 [In 2] — Load Data**
```python
df=pd.read_csv('BigBasket Products.csv')
```

**Cell #3 [In 3] — Display DataFrame**
```python
df
```

**Cell #4 [In 4] — DataFrame Info**
```python
df.info()
```

**Cell #5 [In 5] — Null Check**
```python
df.isnull().sum()
```

**Cell #6 [In 6] — Missing Values Bar Chart**
```python
# Count of missing values per column
null_counts = df.isnull().sum()

# Filter only columns with null values
null_counts = null_counts[null_counts > 0]

# Check if null_counts is empty
if not null_counts.empty:
    # Plot the bar chart
    plt.figure(figsize=(12, 6))
    null_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Count of Missing Values in BigBasket Dataset', fontsize=16)
    plt.xlabel('Columns', fontsize=12)
    plt.ylabel('Number of Missing Values', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
else:
    print("No missing values found in the dataset.")
```

**Cell #8 [In 8] — View null product rows**
```python
df[(df['product'].isnull())]
```

**Cell #9 [In 9] — Drop null product rows**
```python
df = df[~(df['product'].isnull())]
```

**Cell #10 [In 10] — Verify product nulls removed**
```python
df[(df['product'].isnull())]
```

**Cell #11 [In 11] — View null brand rows**
```python
df[(df['brand'].isnull())]
```

**Cell #12 [In 12] — Drop null brand rows**
```python
df = df[~(df['brand'].isnull())]
```

**Cell #13 [In 13] — Verify brand nulls removed**
```python
df[(df['brand'].isnull())]
```

**Cell #14 [In 14] — Null percentage**
```python
# percentage of null values
df.isnull().sum()*100/len(df)
```

**Cell #15 [In 15] — Rows where both rating AND description are null**
```python
df[df[['rating', 'description',]].isnull().all(axis=1)]
```

**Cell #16 [In 16] — Drop rows where both rating and description are null**
```python
df = df[~df[['rating', 'description',]].isnull().all(axis=1)]
```

**Cell #17 [In 17] — Verify combined nulls removed**
```python
df[df[['rating', 'description',]].isnull().all(axis=1)]
```

**Cell #18 [In 18] — Updated null percentages**
```python
# percentage of null values
df.isnull().sum()*100/len(df)
```

**Cell #19 [In 19] — View null description rows**
```python
df[(df['description'].isnull())]
```

**Cell #20 [In 20] — Drop null description rows**
```python
df = df[~(df['description'].isnull())]
```

**Cell #21 [In 21] — Verify description nulls removed**
```python
df[(df['description'].isnull())]
```

**Cell #22 [In 22] — Final null percentages**
```python
# percentage of null values
df.isnull().sum()*100/len(df)
```

**Cell #23 [In 23] — View null rating rows**
```python
df[(df['rating'].isnull())]
```

**Cell #24 [In 24] — DataFrame info after cleaning**
```python
df.info()
```

**Cell #25 [In 25] — Rating skewness**
```python
df.rating.skew()
```

**Cell #26 [In 26] — Fill rating nulls with mean**
```python
df.rating.fillna(df.rating.mean(), inplace=True)
df.isnull().sum()
```

**Cell #27 [In 27] — Descriptive statistics**
```python
df.describe()
```

**Cell #29 [In 29] — Feature: price difference**
```python
df['diff_in_prices'] = df['market_price'] - df['sale_price']
```

**Cell #30 [In 30] — Display df**
```python
df
```

**Cell #32 [In 32] — Feature: discount percentage**
```python
df['discount_percentage'] = (df['diff_in_prices'] / df['market_price']) * 100
```

**Cell #33 [In 33] — Display df**
```python
df
```

**Cell #34 [In 34] — DataFrame info**
```python
df.info()
```

**Cell #36 [In 36] — Describe (outlier context)**
```python
df.describe()
```

**Cell #37 [In 37] — 95th percentile sale price**
```python
df['sale_price'].quantile(0.95)
```

**Cell #38 [In 38] — 99th percentile sale price**
```python
df['sale_price'].quantile(0.99)
```

**Cell #40 [In 40] — Boxplots for all numerical features**
```python
# List of numerical columns
numerical_columns = ['sale_price', 'market_price', 'rating', 'diff_in_prices', 'discount_percentage']

# Plot boxplots for each numerical column
for column in numerical_columns:
    plt.figure(figsize=(8, 4))
    plt.boxplot(df[column].dropna(), vert=False, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='black'),
                medianprops=dict(color='red'))
    plt.title(f'Boxplot of {column}', fontsize=14)
    plt.xlabel('Values')
    plt.show()
```

**Cell #43 [In 43] — Univariate: list numerical columns**
```python
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

print("Numerical Columns:")
print(numerical_columns)
```

**Cell #44 [In 44] — Sale price histogram**
```python
df['sale_price'].hist(bins=50)
plt.title('Distribution of Sale Price')
plt.show()
```

**Cell #45 [In 45] — Category value counts**
```python
df['category'].value_counts()
```

**Cell #46 [In 46] — Category countplot**
```python
# Countplot for Category
plt.figure(figsize=(12, 8))
sns.countplot(y=df['category'], order=df['category'].value_counts().index)
plt.title('Category Selling Distribution')
plt.show()
```

**Cells #47, #48 [In ?, ?] — EMPTY**

**Cell #49 [In 49] — Category pie chart**
```python
# Calculate category frequencies
category_frequencies = df['category'].value_counts()

# Create a pie chart
plt.figure(figsize=(12, 12))

# Pie chart
plt.pie(category_frequencies, labels=category_frequencies.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.tab20.colors)

# Add a legend on the right side
plt.legend(category_frequencies.index, title='Categories', loc='center left', bbox_to_anchor=(1, 0.5))

# Title
plt.title('Category Distribution')

# Show the plot
plt.show()
```

**Cell #50 [In ?] — EMPTY**

**Cell #51 [In 51] — Rating KDE (density) plot**
```python
# Create a density plot
plt.figure(figsize=(10, 6))
sns.kdeplot(df['rating'], shade=True, color='blue')
plt.title('Density Plot of Ratings')
plt.xlabel('Rating')
plt.ylabel('Density')
plt.grid(True)
plt.show()
```

**Cell #52 [In 52] — Rating violin plot**
```python
# Create a violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(x='rating', data=df, inner='quartile', color='blue')
plt.title('Violin Plot of Ratings')
plt.xlabel('Rating')
plt.grid(True)
plt.show()
```

**Cell #56 [In 56] — Bivariate scatter: sale vs market price**
```python
# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['market_price'], df['sale_price'], alpha=0.5)
plt.title('Scatter Plot of Sale Price vs. Market Price')
plt.xlabel('Market Price')
plt.ylabel('Sale Price')
plt.grid(True)
plt.show()
```

**Cell #58 [In 58] — Correlation heatmap**
```python
# Select numerical columns
numerical_columns = ['sale_price', 'market_price', 'rating', 'diff_in_prices', 'discount_percentage']

# Create a correlation matrix
correlation_matrix = df[numerical_columns].corr()

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Variables')
plt.show()
```

**Cell #60 [In 60] — Pair plot (first)**
```python
# Select numerical columns
numerical_columns = ['sale_price', 'market_price', 'rating', 'diff_in_prices', 'discount_percentage']

# Create a pair plot
sns.pairplot(df[numerical_columns])
plt.suptitle('Pair Plot of Numerical Variables', y=1.02)
plt.show()
```

**Cell #61 [In 61] — Scatter: rating vs discount percentage**
```python
# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['discount_percentage'], df['rating'], alpha=0.5)
plt.title('Scatter Plot of Rating vs. Discount Percentage')
plt.xlabel('Discount Percentage')
plt.ylabel('Rating')
plt.grid(True)
plt.show()
```

**Cell #63 [In 63] — Num-Cat boxplot: sale price by category**
```python
# Create a box plot
plt.figure(figsize=(15, 6))
sns.boxplot(x='category', y='sale_price', data=df)
plt.title('Box Plot of Sale Price by Category')
plt.xlabel('Category')
plt.ylabel('Sale Price')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
```

**Cell #64 [In 64] — Bar: avg sale price by category**
```python
# Group by category and calculate the mean sale price
mean_sale_price_by_category = df.groupby('category')['sale_price'].mean().reset_index()

# Create a bar plot
plt.figure(figsize=(16, 6))
sns.barplot(x='category', y='sale_price', data=mean_sale_price_by_category)
plt.title('Average Sale Price by Category')
plt.xlabel('Category')
plt.ylabel('Average Sale Price')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
```

**Cell #65 [In 65] — Boxplot: discount % by category**
```python
# Create a box plot
plt.figure(figsize=(12, 6))
sns.boxplot(x='category', y='discount_percentage', data=df)
plt.title('Box Plot of Discount Percentage by Category')
plt.xlabel('Category')
plt.ylabel('Discount Percentage')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
```

**Cell #66 [In 66] — Bar: avg discount % by category (mislabeled as "by brand")**
```python
# Group by brand and calculate the mean discount percentage
mean_discount_by_brand = df.groupby('category')['discount_percentage'].mean().reset_index()

# Create a bar plot
plt.figure(figsize=(16, 6))
sns.barplot(x='category', y='discount_percentage', data=mean_discount_by_brand)
plt.title('Average Discount Percentage by category')
plt.xlabel('category Name')
plt.ylabel('Average Discount Percentage')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
```

**Cell #67 [In 67] — Line: avg rating by category**
```python
# Group by category and calculate the mean rating
mean_rating_by_category = df.groupby('category')['rating'].mean().reset_index()

# Create a line plot
plt.figure(figsize=(12, 6))
sns.lineplot(x='category', y='rating', data=mean_rating_by_category, marker='o')
plt.title('Average Rating by Category')
plt.xlabel('Category')
plt.ylabel('Average Rating')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
```

**Cell #68 [In 68] — Line: avg sale price by sub-category**
```python
# Group by sub category and calculate the mean sale price
mean_sale_price_by_sub_category = df.groupby('sub_category')['sale_price'].mean().reset_index()

# Create a line plot
plt.figure(figsize=(35, 6))
sns.lineplot(x='sub_category', y='sale_price', data=mean_sale_price_by_sub_category, marker='o')
plt.title('Average Sale Price by Sub Category')
plt.xlabel('Sub Category')
plt.ylabel('Average Sale Price')
plt.xticks(rotation=90)
plt.grid(True)
plt.show()
```

**Cell #71 [In 71] — Cat-Cat count plot: category by sub-category**
```python
# Create a count plot
plt.figure(figsize=(20, 6))
sns.countplot(x='category', hue='sub_category', data=df)
plt.title('Count Plot of Category by Sub Category')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.legend(title='Sub Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()
```

**Cell #73 [In 73] — Pair plot (second, near-duplicate)**
```python
# Select numerical columns
numerical_columns = ['sale_price', 'market_price', 'rating', 'diff_in_prices', 'discount_percentage']

# Create a pair plot
sns.pairplot(df[numerical_columns], diag_kind='kde', markers='o')
plt.suptitle('Pair Plot of Numerical Variables', y=1.02)
plt.show()
```

**Cell #74 [In 74] — 3D scatter: sale price, market price, rating**
```python
# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

x = df['sale_price']
y = df['market_price']
z = df['rating']

ax.scatter(x, y, z, c='blue', marker='o', alpha=0.5)

ax.set_xlabel('Sale Price')
ax.set_ylabel('Market Price')
ax.set_zlabel('Rating')
ax.set_title('3D Scatter Plot of Sale Price, Market Price, and Rating')

plt.show()
```

**Cell #75 [In 75] — Multivariate boxplot: sale price by category and rating range**
```python
# Create rating ranges
df['rating_range'] = pd.cut(df['rating'], bins=[0, 1, 2, 3, 4, 5], labels=['0-1', '1-2', '2-3', '3-4', '4-5'])

# Create a box plot
plt.figure(figsize=(20, 6))
sns.boxplot(x='category', y='sale_price', hue='rating_range', data=df)
plt.title('Box Plot of Sale Price by Category and Rating Range')
plt.xlabel('Category')
plt.ylabel('Sale Price')
plt.xticks(rotation=45)
plt.legend(title='Rating Range')
plt.grid(True)
plt.show()
```

**Cell #76 [In 76] — Violin: sale price by category and discount bin**
```python
# Discretize the discount_percentage into two bins
df['discount_bin'] = pd.cut(df['discount_percentage'], bins=[0, 10, 100], labels=['Low', 'High'])

# Create a violin plot
plt.figure(figsize=(12, 6))
sns.violinplot(x='category', y='sale_price', hue='discount_bin', data=df, inner='quartile', split=True)
plt.title('Violin Plot of Sale Price by Category and Discount Percentage')
plt.xlabel('Category')
plt.ylabel('Sale Price')
plt.xticks(rotation=45)
plt.legend(title='Discount Bin')
plt.grid(True)
plt.show()
```

**Cell #77 [In 77] — Bubble chart**
```python
plt.figure(figsize=(10, 6))
plt.scatter(df['market_price'], df['sale_price'], 
            s=df['discount_percentage']*10, c=df['rating'], cmap='viridis', alpha=0.7)
plt.colorbar(label="Rating")
plt.title("Bubble Chart: Market Price vs Sale Price with Discount Percentage and Rating", fontsize=16)
plt.xlabel("Market Price")
plt.ylabel("Sale Price")
plt.show()
```

**Cell #78 [In 78] — Violin: discount % across categories**
```python
plt.figure(figsize=(12, 6))
sns.violinplot(data=df, x='category', y='discount_percentage', palette='muted')
plt.xticks(rotation=45)
plt.title("Distribution of Discount Percentage Across Categories", fontsize=16)
plt.show()
```

**Cell #79 [In 79] — Stacked bar: subcategories within categories**
```python
# Create a pivot table for stacked bar chart
stacked_data = df.groupby(['category', 'sub_category']).size().unstack()

# Plot the stacked bar chart
stacked_data.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab20')
plt.title("Stacked Bar Plot of Subcategories within Categories", fontsize=16)
plt.xlabel("Category")
plt.ylabel("Count")
plt.legend(title="Subcategories", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
```

**Cell #80 [In 80] — Scatter: market vs sale price with discount hue and rating size**
```python
# Scatter plot with 'discount_percentage' as hue
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='market_price', y='sale_price', hue='discount_percentage', palette='coolwarm', size='rating', sizes=(20, 200))
plt.title("Market Price vs Sale Price with Discount Percentage and Rating", fontsize=16)
plt.xlabel("Market Price")
plt.ylabel("Sale Price")
plt.legend(title="Discount Percentage", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
```

**Cell #82 [In 82] — Set seaborn style**
```python
sns.set(style="whitegrid")
```

**Cell #83 [In 83] — Scale features for PCA/t-SNE**
```python
# Scaling the data for PCA and t-SNE
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.select_dtypes(include=[np.number]))
```

**Cell #84 [In 84] — PCA (2 components)**
```python
# PCA Analysis
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)
pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
pca_df['category'] = df['category']
```

**Cell #85 [In 85] — PCA scatter plot**
```python
#Multivariate Scatter Plot with PCA
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='category', data=pca_df, palette='Set1', alpha=0.7)
plt.title('PCA Result: Multivariate Scatter Plot')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend()
plt.show()
```

**Cell #86 [In 86] — t-SNE**
```python
# Explicitly set init and learning_rate to avoid FutureWarnings
tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='random', learning_rate=200.0)
tsne_result = tsne.fit_transform(scaled_features)

tsne_df = pd.DataFrame(data=tsne_result, columns=['TSNE1', 'TSNE2'])
tsne_df['category'] = df['category']
```

**Cell #87 [In 87] — t-SNE scatter plot**
```python
# Plotting t-SNE Results
plt.figure(figsize=(10, 6))
sns.scatterplot(x='TSNE1', y='TSNE2', hue='category', data=tsne_df, palette='Set2', alpha=0.7)
plt.title('t-SNE Result: Clustering of Products')
plt.xlabel('TSNE1')
plt.ylabel('TSNE2')
# To Move the legend to the right side
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Category")
plt.tight_layout()
plt.show()
```

**Cell #88 [In 88] — Import plotly**
```python
import plotly.express as px
```

**Cell #89 [In 89] — Interactive 3D PCA scatter (Plotly)**
```python
# Assuming df still contains the rating column and hasn't lost alignment
pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])

# Include the 'category' and 'rating' columns from the original DataFrame
# First, ensure df and pca_df have matching indices
df_filtered = df[df['rating'].notnull()]  # Filter rows where rating is not null

# Reset index to maintain alignment
df_filtered.reset_index(drop=True, inplace=True)

# Create pca_df with relevant columns
pca_df['category'] = df_filtered['category']
pca_df['rating'] = df_filtered['rating']  # Add rating to pca_df

# 3D Scatter Plot using Plotly
fig = px.scatter_3d(pca_df, x='PCA1', y='PCA2', z='rating', color='category',
                    title='3D Scatter Plot of PCA Results',
                    labels={'PCA1': 'PCA1', 'PCA2': 'PCA2', 'rating': 'Rating'},
                    hover_name='category')
fig.show()
```

**Cell #90 [In 90] — KMeans (4 clusters) — RESULT UNUSED**
```python
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)
```

**Cell #91 [In 91] — Print pca_df columns**
```python
print(pca_df.columns)
```

**Cell #92 [In 92] — KMeans on PCA (3 clusters) + scatter**
```python
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(pca_df[['PCA1', 'PCA2']])

# Add the cluster labels to the pca_df DataFrame
pca_df['Cluster'] = clusters

# Plot the scatterplot with clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=pca_df, palette='Set3', alpha=0.7)
plt.title('Clusters based on PCA Results')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(title='Cluster')
plt.show()
```

**Cell #94 [In 94] — Print df.columns**
```python
df.columns
```

**Cell #97 [In 97] — Feature deep-dive: index**
```python
df["index"].head()
```

**Cell #100 [In 100] — Feature deep-dive: product**
```python
df["product"].head()
```

**Cell #103 [In 103] — Feature deep-dive: category**
```python
df["category"].head()
```

**Cell #105 [In 105] — Revenue by category pie chart**
```python
# Revenue contribution by category
revenue_by_category = df.groupby('category')['sale_price'].sum()

# Pie chart
plt.figure(figsize=(20, 15))
revenue_by_category.plot(kind='pie', autopct='%1.1f%%', startangle=90, colormap='coolwarm')
plt.title('Revenue Contribution by Category')
plt.ylabel('')  # Hide y-axis label
plt.show()
```

**Cell #106 [In 106] — Avg sale price by category (horizontal bar)**
```python
# Calculate average sale price by category
avg_sale_price = df.groupby('category')['sale_price'].mean().sort_values()

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=avg_sale_price.values, y=avg_sale_price.index, palette='viridis')
plt.title('Average Sale Price by Category')
plt.xlabel('Average Sale Price')
plt.ylabel('Category')
plt.show()
```

**Cell #109 [In ?] — Avg price by sub-category bar**
```python
# Group by sub_category and calculate average sale_price
avg_price_sub_category = df.groupby('sub_category')['sale_price'].mean().sort_values(ascending=False)

# Plot bar chart
plt.figure(figsize=(25, 10))
avg_price_sub_category.plot(kind='bar', color='teal')
plt.title("Average Sale Price per Sub-Category")
plt.ylabel("Average Sale Price")
plt.xlabel("Sub-Category")
plt.xticks(rotation=90, ha='right')
plt.show()
```

**Cell #112 [In ?] — Top 10 brands by product count (duplicate imports)**
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Group by 'brand' and count the number of products for each brand
brand_counts = df['brand'].value_counts().head(10)

# Create a bar plot for the top 10 brands by product count
plt.figure(figsize=(12, 6))
sns.barplot(x=brand_counts.index, y=brand_counts.values, palette='Set2')
plt.title('Top 10 Brands by Number of Products')
plt.xlabel('Brand')
plt.ylabel('Number of Products')
plt.xticks(rotation=45)
plt.show()
```

**Cell #114 [In ?] — Top 10 brands by avg sale price (duplicate imports)**
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Group by 'brand' and calculate the average sale price for each brand
avg_sale_price_by_brand = df.groupby('brand')['sale_price'].mean().sort_values(ascending=False).head(10)

# Create a bar plot for the top 10 brands by average sale price
plt.figure(figsize=(12, 6))
sns.barplot(x=avg_sale_price_by_brand.index, y=avg_sale_price_by_brand.values, palette='Set2')
plt.title('Top 10 Brands by Average Sale Price')
plt.xlabel('Brand')
plt.ylabel('Average Sale Price')
plt.xticks(rotation=45)
plt.show()
```

**Cell #116 [In ?] — Bottom 10 brands by avg sale price (duplicate imports)**
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Group by 'brand' and calculate the average sale price for each brand
avg_sale_price_by_brand = df.groupby('brand')['sale_price'].mean().sort_values(ascending=False).tail(10)

# Create a bar plot for the Bottom 10 brands by average sale price
plt.figure(figsize=(12, 6))
sns.barplot(x=avg_sale_price_by_brand.index, y=avg_sale_price_by_brand.values, palette='Set2')
plt.title('Bottom 10 Brands by Average Sale Price')
plt.xlabel('Brand')
plt.ylabel('Average Sale Price')
plt.xticks(rotation=45)
plt.show()
```

**Cell #119 [In ?] — Sale price describe**
```python
print(df['sale_price'].describe())
```

**Cell #121 [In ?] — Sale price histplot**
```python
plt.figure(figsize=(10, 6))
sns.histplot(df['sale_price'], bins=30, kde=True, color='blue')
plt.title("Distribution of Sale Price", fontsize=16)
plt.xlabel("Sale Price")
plt.ylabel("Frequency")
plt.show()
```

**Cell #123 [In ?] — Scatter: sale vs market price with rating hue**
```python
sns.scatterplot(data=df, x='market_price', y='sale_price', hue='rating', palette='coolwarm')
plt.title("Sale Price vs Market Price by Rating", fontsize=16)
plt.xlabel("Market Price")
plt.ylabel("Sale Price")
plt.show()
```

**Cell #125 [In ?] — Boxplot: sale price outliers**
```python
sns.boxplot(data=df, y='sale_price', color='orange')
plt.title("Outliers in Sale Price", fontsize=16)
plt.ylabel("Sale Price")
plt.show()
```

**Cell #128 [In ?] — Market price describe**
```python
print(df['market_price'].describe())
```

**Cell #129 [In ?] — Market price histplot**
```python
plt.figure(figsize=(10, 6))
sns.histplot(df['market_price'], bins=30, kde=True, color='green')
plt.title("Distribution of Market Price", fontsize=16)
plt.xlabel("Market Price")
plt.ylabel("Frequency")
plt.show()
```

**Cell #130 [In ?] — Scatter: market vs sale price (axis labels swapped)**
```python
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, y='market_price', x='sale_price', color='blue', alpha=0.6)
plt.title("Market Price vs Sale Price", fontsize=16)
plt.xlabel("Sale Price")
plt.ylabel("market Price")
plt.show()
```

**Cell #133 [In ?] — Type value counts**
```python
print(df['type'].value_counts())
```

**Cell #135 [In ?] — Top 10 product types**
```python
# Top 10 product types by count
top_10_types = df['type'].value_counts().head(10)

# Display the result
print(top_10_types)
```

**Cell #136 [In ?] — Top 10 types barplot (duplicate imports)**
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Plot the top 10 product types
plt.figure(figsize=(10, 6))
sns.barplot(x=top_10_types.values, y=top_10_types.index, palette='Blues_r')
plt.title("Top 10 Product Types by Count", fontsize=16)
plt.xlabel("Count")
plt.ylabel("Product Type")
plt.show()
```

**Cell #138 [In ?] — MODEL CREATION: sklearn imports**
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
```

**Cell #140 [In ?] — Feature/target split + preprocessor pipeline**
```python
# Splitting the data into features (X) and target (y)
X = df[['market_price', 'category', 'sub_category', 'brand', 'type', 'rating', 'discount_percentage']]
y = df['sale_price']

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['market_price', 'rating', 'discount_percentage']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['category', 'sub_category', 'brand', 'type'])
    ]
)
```

**Cell #142 [In ?] — Linear Regression — MISSING IMPORT (will fail on fresh run)**
```python
# Define the model
linear_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train the model
linear_model.fit(X_train, y_train)

# Evaluate the model
y_pred_linear = linear_model.predict(X_test)
linear_r2 = r2_score(y_test, y_pred_linear)
linear_rmse = np.sqrt(mean_squared_error(y_test, y_pred_linear))

print("Linear Regression - R^2 Score:", linear_r2)
print("Linear Regression - RMSE:", linear_rmse)
```

**Cell #144 [In ?] — Random Forest**
```python
# Define the model
rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the model
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred_rf = rf_model.predict(X_test)
rf_r2 = r2_score(y_test, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print("Random Forest - R^2 Score:", rf_r2)
print("Random Forest - RMSE:", rf_rmse)
```

**Cell #146 [In ?] — Gradient Boosting — MISSING IMPORT (will fail on fresh run)**
```python
# Define the model
gb_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42))
])

# Train the model
gb_model.fit(X_train, y_train)

# Evaluate the model
y_pred_gb = gb_model.predict(X_test)
gb_r2 = r2_score(y_test, y_pred_gb)
gb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_gb))

print("Gradient Boosting - R^2 Score:", gb_r2)
print("Gradient Boosting - RMSE:", gb_rmse)
```

**Cell #148 [In ?] — XGBoost — MISSING IMPORT (will fail on fresh run)**
```python
# Define the model
xgb_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42))
])

# Train the model
xgb_model.fit(X_train, y_train)

# Evaluate the model
y_pred_xgb = xgb_model.predict(X_test)
xgb_r2 = r2_score(y_test, y_pred_xgb)
xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))

print("XGBoost - R^2 Score:", xgb_r2)
print("XGBoost - RMSE:", xgb_rmse)
```

**Cell #150 [In ?] — Model results summary table**
```python
# Summarizing results
results = {
    "Model": ["Linear Regression", "Random Forest", "Gradient Boosting", "XGBoost"],
    "R^2 Score": [linear_r2, rf_r2, gb_r2, xgb_r2],
    "RMSE": [linear_rmse, rf_rmse, gb_rmse, xgb_rmse]
}

results_df = pd.DataFrame(results)
print(results_df)
```

**Cell #152 [In ?] — Model performance line plot**
```python
plt.figure(figsize=(10, 6))

# R² Scores
sns.lineplot(x="Model", y="R^2 Score", data=results_df, marker='o', label="R² Score", color='blue')

# RMSE
sns.lineplot(x="Model", y="RMSE", data=results_df, marker='o', label="RMSE", color='red')

plt.title("Model Performance Trends")
plt.ylabel("Metric Value")
plt.xlabel("Model")
plt.legend(title="Metric")
plt.grid()
plt.show()
```

**Cell #154 [In ?] — Model performance grouped bar chart (duplicate imports)**
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Prepare data for visualization
metrics = pd.melt(results_df, id_vars=["Model"], value_vars=["R^2 Score", "RMSE"], 
                  var_name="Metric", value_name="Value")

# Bar Plot
plt.figure(figsize=(12, 6))
sns.barplot(x="Model", y="Value", hue="Metric", data=metrics, palette="Set2")
plt.title("Comparison of Model Performance (R² Score and RMSE)")
plt.ylabel("Metric Value")
plt.xlabel("Model")
plt.xticks(rotation=30)
plt.legend(title="Metric")
plt.show()
```

**Cell #156 [In ?] — Linear Regression: predicted vs actual**
```python
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_linear, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color='red')
plt.title("Linear Regression: Predictions vs Actual Values")
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.show()
```

---

### 2d. Stored Model Output Metrics (from saved cell outputs)

The notebook was run to completion in a previous session and outputs are stored. Saved model metrics:

```
Linear Regression - R^2 Score: 0.972913900482795
Linear Regression - RMSE: 80.46134025870236

Random Forest - R^2 Score: 0.9989521895306274
Random Forest - RMSE: 15.825425444815657

Gradient Boosting - R^2 Score: 0.9986648003084868
Gradient Boosting - RMSE: 17.86434672194154

XGBoost - R^2 Score: 0.9914005789708599
XGBoost - RMSE: 45.33656727913853

Summary Table:
               Model  R^2 Score       RMSE
0  Linear Regression   0.972914  80.461340
1      Random Forest   0.998952  15.825425
2  Gradient Boosting   0.998665  17.864347
3            XGBoost   0.991401  45.336567
```

---

## 3. OTHER FILES

### `Dataset/BigBasket Products.csv`
- **Size:** 16,739,247 bytes (16.0 MB)
- **Columns:** `index, product, category, sub_category, brand, sale_price, market_price, type, rating, description`
- **First 3 data rows (truncated):**
```
1, Garlic Oil - Vegetarian Capsule 500 mg, Beauty & Hygiene, Hair Care, Sri Sri Ayurveda, 220, 220, Hair Oil & Serum, 4.1
2, Water Bottle - Orange, Kitchen Garden & Pets, Storage & Accessories, Mastercook, 180, 180, Water & Fridge Bottles, 2.3
3, Brass Angle Deep - Plain No.2, Cleaning & Household, Pooja Needs, Trm, 119, 250, Lamp & Lamp Oil, 3.4
```

### `Docs or Reports/BigBasket EDA Final Report end.pdf`
- **Size:** 1,833,372 bytes (1.74 MB)
- **Page count:** Not extractable without PDF library

### `EDA PROJECT.pptx`  *(in repo root — note: PDF is in subfolder, PPTX is loose in root)*
- **Size:** 733,068 bytes (716 KB)
- **Slide count:** 15 slides (confirmed via ZIP inspection of internal `ppt/slides/` entries)

---

## 4. CURRENT README.md — FULL CONTENT VERBATIM

```
# EDA-on-BigBasket-Dataset
This project analyzes BigBasket data to uncover trends in product categories, pricing, discounts, and customer ratings. Using EDA, To identifies actionable insights for pricing strategies, inventory management, and customer engagement to enhance business decisions.
```

*(295 bytes total — just a title and one sentence. No setup instructions, no folder structure, no dataset description, no model results.)*

---

## 5. MISSING STANDARD FILES CHECK

| File | Status |
|---|---|
| `README.md` | **EXISTS** — but only 2 lines, functionally empty |
| `.gitignore` | **MISSING** |
| `requirements.txt` | **MISSING** |
| `LICENSE` | **MISSING** |
| `environment.yml` | **MISSING** |
| `Pipfile` | **MISSING** |

---

## 6. CODE QUALITY OBSERVATIONS

### 6a. Hardcoded Absolute File Paths

**Critical relative path bug — not absolute, but still broken:**

```python
df=pd.read_csv('BigBasket Products.csv')   # Cell #2
```

The notebook is located at `Script or Codefile/EDA on BigBasket DataSet.ipynb`. Jupyter sets the working directory to the folder containing the notebook — `Script or Codefile/`. The CSV is at `Dataset/BigBasket Products.csv` (one level up, then into `Dataset/`). The path `'BigBasket Products.csv'` would look for the file in `Script or Codefile/BigBasket Products.csv`, which does not exist.

**The correct path should be:** `'../Dataset/BigBasket Products.csv'`

The notebook ran successfully (stored outputs exist), which means it was run either (a) from a local machine with a different directory layout, or (b) with the Jupyter working directory manually changed. Anyone who clones this repo and runs "Restart & Run All" will immediately hit `FileNotFoundError` on Cell #2.

### 6b. Unused Imports, Dead Code, Duplicate Cells

**`silhouette_score` imported but never called (Cell #1):**
```python
from sklearn.metrics import silhouette_score   # imported, never used
```

**Duplicate imports — `matplotlib.pyplot` and `seaborn` re-imported 5 extra times** in cells #112, #114, #116, #136, #154. These are completely redundant since both are already imported in Cell #1.

**15 empty code cells** at positions: 47, 48, 50, 107, 110, 113, 115, 134, 139, 141, 143, 145, 147, 153, 157. The five between model training blocks (139, 141, 143, 145, 147) are especially cluttering.

**Near-duplicate pair plots:** Cell #60 and Cell #73 produce essentially the same chart on the same 5 columns — the only difference is Cell #73 adds `diag_kind='kde'` and `markers='o'`. Both are titled "Pair Plot of Numerical Variables". One should be removed.

**Dead KMeans result (Cell #90):** Fits KMeans with 4 clusters and assigns to `df['Cluster']`, but this column is never used again. Cell #92 immediately re-fits with 3 clusters on PCA space. The 4-cluster result is dead code.

**Misleading variable name (Cell #66):**
```python
# Group by brand and calculate the mean discount percentage   <-- comment says brand
mean_discount_by_brand = df.groupby('category')['discount_percentage'].mean()   # <-- but groups by category
```
The variable name `mean_discount_by_brand` and the comment both say "brand" but the actual groupby uses `'category'`. This is a copy-paste error.

### 6c. Missing Critical Imports — WILL CAUSE NameError on Re-run

Three symbols are used in the model section without ever being imported:

| Used in Cell | Symbol | Missing import |
|---|---|---|
| Cell #142 | `LinearRegression` | `from sklearn.linear_model import LinearRegression` |
| Cell #146 | `GradientBoostingRegressor` | `from sklearn.ensemble import GradientBoostingRegressor` |
| Cell #148 | `XGBRegressor` | `from xgboost import XGBRegressor` |

Only `RandomForestRegressor` is imported correctly (Cell #138). The other three were likely defined in a session where these had been imported in a now-deleted cell. Running "Restart & Run All" will fail at Cell #142 with `NameError: name 'LinearRegression' is not defined`.

### 6d. Narrative Structure

The notebook follows a broadly sensible EDA pipeline:

```
Load → Null Analysis → Clean Nulls → Impute Rating → 
Feature Engineering (diff_in_prices, discount_percentage) → 
Outlier Detection (boxplots) → Univariate → Bivariate (Num-Num, Num-Cat, Cat-Cat) →
Multivariate → Extended EDA (PCA, t-SNE, KMeans) →
Feature-by-feature deep dives → Model Creation → Model Evaluation
```

**Issues with flow:**
- The "Feature deep-dives" section (cells 93–136) re-analyzes individual columns *after* the main EDA has concluded. Several plots in this section duplicate plots already made earlier (e.g., "Average Sale Price by Category" appears in both Cell #64 and Cell #106).
- The PCA/t-SNE/KMeans section precedes the feature deep-dives — dimensionality reduction normally follows EDA, not precedes a second round of EDA.
- The markdown cell at position #101 reads: **"product feature tell"** — this is an unfinished sentence, clearly a draft placeholder never completed.
- The markdown cell at position #57 is entirely blank (no text).

### 6e. Visualization Label Quality

Most plots have proper titles, x-labels, and y-labels. Specific gaps:

| Cell | Issue |
|---|---|
| Cell #44 | `df['sale_price'].hist(bins=50)` — no `plt.xlabel('Sale Price')`, no `plt.ylabel('Frequency')` |
| Cell #46 | `sns.countplot(y=...)` — no `plt.xlabel('Count')` |
| Cell #78 | `sns.violinplot(...)` — no `plt.xlabel('Category')`, no `plt.ylabel('Discount Percentage')` |
| Cell #130 | Y-axis label is `"market Price"` (lowercase 'P') — inconsistent capitalization |

### 6f. Deprecated Syntax

| Cell | Deprecated usage | Recommended replacement |
|---|---|---|
| Cell #26 | `df.rating.fillna(..., inplace=True)` | `df['rating'] = df['rating'].fillna(...)` — `inplace=True` deprecated pandas 2.0+ |
| Cell #51 | `sns.kdeplot(..., shade=True, ...)` | `fill=True` — `shade` renamed in seaborn 0.12+ |
| Cell #76 | `sns.violinplot(..., split=True, ...)` | `split` parameter removed in seaborn 0.13+; use `dodge='auto'` or hue-based split |

### 6g. Other Bugs and Anti-patterns

**Division by zero in `discount_percentage` (Cell #32):**
```python
df['discount_percentage'] = (df['diff_in_prices'] / df['market_price']) * 100
```
No guard against `market_price == 0`. Produces `inf` or `NaN` silently.

**`discount_percentage` can be negative:** No check for cases where `sale_price > market_price` (possible data error), which produces negative discount percentages. The `discount_bin` in Cell #76 uses `bins=[0, 10, 100]` — negative values fall outside these bins and become `NaN`, silently excluded from the violin plot.

**Index alignment risk (Cell #89):** `pca_result` was computed on the full `df` but `df_filtered` drops rows where `rating` is null. After `df.rating.fillna(...)` in Cell #26, no rows should have null ratings — but this dependency is fragile and will silently misalign if run in a different order.

### 6h. Random Forest / XGBoost — Presence Confirmation

**All four models are present in the code.**

| Model | Cell | Import Status | Stored Output |
|---|---|---|---|
| Linear Regression | #142 | **MISSING** — `NameError` on re-run | R² = 0.9729, RMSE = 80.46 |
| Random Forest | #144 | OK (imported Cell #138) | R² = **0.9990**, RMSE = **15.83** |
| Gradient Boosting | #146 | **MISSING** — `NameError` on re-run | R² = 0.9987, RMSE = 17.86 |
| XGBoost | #148 | **MISSING** — `NameError` on re-run | R² = 0.9914, RMSE = 45.34 |

**Train/test split:** 80/20 (`test_size=0.2, random_state=42`)  
**Features:** `market_price`, `category`, `sub_category`, `brand`, `type`, `rating`, `discount_percentage`  
**Target:** `sale_price`  
**Preprocessing:** `StandardScaler` on numerics + `OneHotEncoder(handle_unknown='ignore')` on categoricals, via `ColumnTransformer` + `Pipeline`  
**Evaluation metrics:** R² and RMSE, printed and stored in outputs  

**Note on suspiciously high R²:** R² > 0.999 for tree models may indicate data leakage. `market_price` and `sale_price` are near-identical for most products (many products have zero discount), so the model essentially learns `sale_price ≈ market_price`. This is technically correct but not especially insightful — a portfolio reviewer may flag it.

---

## 7. GIT HISTORY (`git log --oneline -20`)

```
ec32953 Update project description in README.md
58ad466 Delete Docs and Reports directory
c7db66e Rename BigBasket EDA Final Report end.pdf to Docs or Reports/BigBasket EDA Final Report end.pdf
f6fc5ac Delete Docs and Reports/BigBasket%20EDA%20Final%20Report[1].docx
3d3b642 Add files via upload
260d34e Create placeholder
cbeb774 Delete Docs and Presentations
488065a Create Docs and Presentations
cc158b0 Delete EDA on BigBasket DataSet.ipynb
7853e25 Delete BigBasket Products.csv
ba8f68c Delete Script or Codefile  directory
582092c Add files via upload
4665d3b Create placeholder
0bba4a6 Delete Dataset  directory
0112c7b Add files via upload
f38e94d Add placeholder file to Dataset
ffa4864 Create Dataset
bd9ed6b Add files via upload
057df2f Initial commit
```

**Observation:** The git history shows a heavily manual, UI-driven workflow (repeated delete-then-re-upload cycles). There are no incremental, descriptive commit messages that document analytical decisions. The history records folder structure experiments but no meaningful checkpoints of the analysis itself.

---

## SUMMARY — TOP ISSUES FOR PORTFOLIO READINESS

| Priority | Issue | Impact |
|---|---|---|
| **CRITICAL** | `pd.read_csv('BigBasket Products.csv')` wrong path — CSV is at `../Dataset/`, notebook is in `Script or Codefile/` | First cell of data load fails for anyone who clones the repo |
| **CRITICAL** | `LinearRegression`, `GradientBoostingRegressor`, `XGBRegressor` never imported — 3 of 4 models fail on re-run | Model section appears broken to any reviewer who runs "Restart & Run All" |
| **HIGH** | README.md is only 2 lines — no setup, no folder structure, no results, no how-to-run | First thing a recruiter reads; currently unhelpful |
| **HIGH** | No `.gitignore` — `.ipynb_checkpoints/`, `__pycache__/`, OS files can get committed | Standard hygiene |
| **HIGH** | No `requirements.txt` — no way to reproduce the environment | Anyone cloning can't install dependencies |
| **HIGH** | No `LICENSE` | No legal clarity for portfolio use |
| **MEDIUM** | `EDA PROJECT.pptx` is in repo root while PDF is in `Docs or Reports/` — inconsistent organization | Untidy structure |
| **MEDIUM** | 15 empty code cells, near-duplicate pair plots, dead KMeans result | Notebook looks unfinished |
| **MEDIUM** | Misleading variable `mean_discount_by_brand` groups by `category` | Logic bug, confusing to reader |
| **MEDIUM** | `silhouette_score` imported but never used; KMeans cluster count hardcoded without elbow analysis | Incomplete advanced analysis |
| **MEDIUM** | Incomplete markdown cell "product feature tell" | Looks careless |
| **LOW** | Deprecated `inplace=True`, `shade=True`, `split=True` | Warnings on newer library versions |
| **LOW** | `matplotlib`/`seaborn` duplicate imports in 5 cells | Clutter |
| **LOW** | Missing axis labels on cells #44, #46, #78 | Minor visualization quality |
| **LOW** | R² > 0.999 for tree models — possible data leakage via `market_price` feature | Analytical credibility question |
