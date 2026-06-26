#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np #mathematical calculations
import pandas as pd # cleaning exploring and manupulation of data
import matplotlib.pyplot as plt # it is used for data visualisation which is a ML lib
import seaborn as sns #data viz library used to show attractive statistical graphs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


# In[ ]:


# Path is relative to the notebook location (Script or Codefile/)
df = pd.read_csv('../Dataset/BigBasket Products.csv')


# In[ ]:


df


# In[ ]:


df.info()


# In[ ]:


df.isnull().sum()


# In[ ]:


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


# From the visualisation we can say that only product, brand, rating and description c0lumns have null values

# In[ ]:


df[(df['product'].isnull())] 


# In[ ]:


df = df[~(df['product'].isnull())] 


# In[ ]:


df[(df['product'].isnull())] 


# In[ ]:


df[(df['brand'].isnull())] 


# In[ ]:


df = df[~(df['brand'].isnull())] 


# In[ ]:


df[(df['brand'].isnull())] 


# In[ ]:


# percentage of null values

df.isnull().sum()*100/len(df)


# In[ ]:


df[df[['rating', 'description',]].isnull().all(axis=1)]


# In[ ]:


df = df[~df[['rating', 'description',]].isnull().all(axis=1)]


# In[ ]:


df[df[['rating', 'description',]].isnull().all(axis=1)]


# In[ ]:


# percentage of null values

df.isnull().sum()*100/len(df)


# In[ ]:


df[(df['description'].isnull())] 


# In[ ]:


df = df[~(df['description'].isnull())] 


# In[ ]:


df[(df['description'].isnull())] 


# In[ ]:


# percentage of null values

df.isnull().sum()*100/len(df)


# In[ ]:


df[(df['rating'].isnull())] 


# In[ ]:


df.info()


# In[ ]:


df.rating.skew()


# In[ ]:


df['rating'] = df['rating'].fillna(df['rating'].mean())
df.isnull().sum()


# In[ ]:


df.describe()


# # Creation of a new feature "Price difference" 

# In[ ]:


df['diff_in_prices'] = df['market_price'] - df['sale_price']


# In[ ]:


df


# # Creation of a new feature "Discount Percentage"

# In[ ]:


df['discount_percentage'] = (df['diff_in_prices'] / df['market_price']) * 100


# In[ ]:


df


# In[ ]:


df.info()


# # Outliers Detection

# In[ ]:


df.describe()


# In[ ]:


df['sale_price'].quantile(0.95)


# In[ ]:


df['sale_price'].quantile(0.99)


# # Boxploting All the Numerical features to see outliers

# In[ ]:


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


# Even though there are outliers, it is not necessary to impute or remove them

# # Univariate Analysis 

# In[ ]:



numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

print("Numerical Columns:")
print(numerical_columns)


# In[ ]:


df['sale_price'].hist(bins=50)
plt.title('Distribution of Sale Price')
plt.xlabel('Sale Price')
plt.ylabel('Frequency')
plt.show()


# In[ ]:


df['category'].value_counts()


# In[ ]:


# Countplot for Category
plt.figure(figsize=(12, 8))
sns.countplot(y=df['category'], order=df['category'].value_counts().index)
plt.title('Category Selling Distribution')
plt.xlabel('Count')
plt.show()


# In[ ]:


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


# In[ ]:


# Create a density plot
plt.figure(figsize=(10, 6))
sns.kdeplot(df['rating'], fill=True, color='blue')
plt.title('Density Plot of Ratings')
plt.xlabel('Rating')
plt.ylabel('Density')
plt.grid(True)
plt.show()


# In[ ]:


# Create a violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(x='rating', data=df, inner='quartile', color='blue')
plt.title('Violin Plot of Ratings')
plt.xlabel('Rating')
plt.grid(True)
plt.show()


# # Bivariate Analysis

# Bivariate analysis helps you understand the relationship between two variables.
# there are four types of bivariate analysis
# 
# 
# 1.Numerical to Numerical 
# 2.Categorical to Categorical 
# 3.Numerical to Categrical 
# 4.Categorical to Numerical

# In Numerical to Numerical, Firstly we will see the relationship between Sale price and Market Price

# In[ ]:


# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['market_price'], df['sale_price'], alpha=0.5)
plt.title('Scatter Plot of Sale Price vs. Market Price')
plt.xlabel('Market Price')
plt.ylabel('Sale Price')
plt.grid(True)
plt.show()


# 

# In[ ]:


# Select numerical columns
numerical_columns = ['sale_price', 'market_price', 'rating', 'diff_in_prices', 'discount_percentage']

# Create a correlation matrix
correlation_matrix = df[numerical_columns].corr()

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Variables')
plt.show()


# This Correlation matrix helps us to understand the correlation between numerical variables.

# In[ ]:


# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['discount_percentage'], df['rating'], alpha=0.5)
plt.title('Scatter Plot of Rating vs. Discount Percentage')
plt.xlabel('Discount Percentage')
plt.ylabel('Rating')
plt.grid(True)
plt.show()


# # Numerical to Categorical

# In[ ]:


# Create a box plot
plt.figure(figsize=(15, 6))
sns.boxplot(x='category', y='sale_price', data=df)
plt.title('Box Plot of Sale Price by Category')
plt.xlabel('Category')
plt.ylabel('Sale Price')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[ ]:


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


# In[ ]:


# Create a box plot
plt.figure(figsize=(12, 6))
sns.boxplot(x='category', y='discount_percentage', data=df)
plt.title('Box Plot of Discount Percentage by Category')
plt.xlabel('Category')
plt.ylabel('Discount Percentage')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[ ]:


# Group by brand and calculate the mean discount percentage
mean_discount_by_category = df.groupby('category')['discount_percentage'].mean().reset_index()

# Create a bar plot
plt.figure(figsize=(16, 6))
sns.barplot(x='category', y='discount_percentage', data=mean_discount_by_category)
plt.title('Average Discount Percentage by category')
plt.xlabel('category Name')
plt.ylabel('Average Discount Percentage')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[ ]:


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


# In[ ]:


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


# # Categorical-Categorical Relationships

# Count Plot: Category by Sub Category
# This plot helps you understand the distribution of categories within subcategories.

# In[ ]:


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


# # Multivariate analysis

# In[ ]:


# Select numerical columns
numerical_columns = ['sale_price', 'market_price', 'rating', 'diff_in_prices', 'discount_percentage']

# Create a pair plot
sns.pairplot(df[numerical_columns], diag_kind='kde', markers='o')
plt.suptitle('Pair Plot of Numerical Variables', y=1.02)
plt.show()


# In[ ]:


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


# In[ ]:


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


# In[ ]:



# Discretize the discount_percentage into two bins
df['discount_bin'] = pd.cut(df['discount_percentage'], bins=[0, 10, 100], labels=['Low', 'High'])

# Create a violin plot
plt.figure(figsize=(12, 6))
sns.violinplot(x='category', y='sale_price', hue='discount_bin', data=df, inner='quartile')
plt.title('Violin Plot of Sale Price by Category and Discount Percentage')
plt.xlabel('Category')
plt.ylabel('Sale Price')
plt.xticks(rotation=45)
plt.legend(title='Discount Bin')
plt.grid(True)
plt.show()


# In[ ]:


plt.figure(figsize=(10, 6))
plt.scatter(df['market_price'], df['sale_price'], 
            s=df['discount_percentage']*10, c=df['rating'], cmap='viridis', alpha=0.7)
plt.colorbar(label="Rating")
plt.title("Bubble Chart: Market Price vs Sale Price with Discount Percentage and Rating", fontsize=16)
plt.xlabel("Market Price")
plt.ylabel("Sale Price")
plt.show()


# In[ ]:


plt.figure(figsize=(12, 6))
sns.violinplot(data=df, x='category', y='discount_percentage', palette='muted')
plt.xticks(rotation=45)
plt.title("Distribution of Discount Percentage Across Categories", fontsize=16)
plt.xlabel('Category')
plt.ylabel('Discount Percentage')
plt.show()


# In[ ]:


# Create a pivot table for stacked bar chart
stacked_data = df.groupby(['category', 'sub_category']).size().unstack()

# Plot the stacked bar chart
stacked_data.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab20')
plt.title("Stacked Bar Plot of Subcategories within Categories", fontsize=16)
plt.xlabel("Category")
plt.ylabel("Count")
plt.legend(title="Subcategories", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# In[ ]:


# Scatter plot with 'discount_percentage' as hue
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='market_price', y='sale_price', hue='discount_percentage', palette='coolwarm', size='rating', sizes=(20, 200))
plt.title("Market Price vs Sale Price with Discount Percentage and Rating", fontsize=16)
plt.xlabel("Market Price")
plt.ylabel("Sale Price")
plt.legend(title="Discount Percentage", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# # EXTENDED ADVANCED EDA VISUALISATIONS

# In[ ]:


sns.set(style="whitegrid")


# In[ ]:


# Scaling the data for PCA and t-SNE
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.select_dtypes(include=[np.number]))


# In[ ]:


# PCA Analysis
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)
pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
pca_df['category'] = df['category']


# In[ ]:


#Multivariate Scatter Plot with PCA
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='category', data=pca_df, palette='Set1', alpha=0.7)
plt.title('PCA Result: Multivariate Scatter Plot')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend()
plt.show()


# In[ ]:


# Explicitly set init and learning_rate to avoid FutureWarnings
tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='random', learning_rate=200.0)
tsne_result = tsne.fit_transform(scaled_features)

tsne_df = pd.DataFrame(data=tsne_result, columns=['TSNE1', 'TSNE2'])
tsne_df['category'] = df['category']


# In[ ]:


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


# In[ ]:


import plotly.express as px 


# In[ ]:


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


# In[ ]:


print(pca_df.columns)


# In[ ]:


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


# # Feature Engineering - feature selection and exploration

# In[ ]:


df.columns


# There are 15 features and those less related to price prediction will be dropped to refine our EDA and Modeling Focus:

# ## Feature: "Index"

# In[ ]:


df["index"].head()


# index is nothing but numbering or indexing for each item. So It is unnecessary column for further analysis

# ## Feature: "Product"

# In[ ]:


df["product"].head()


# product feature tell

# ## Feature: 'Category'

# In[ ]:


df["category"].head()


# The category feature typically represents the broad classification of a product.

# In[ ]:


# Revenue contribution by category
revenue_by_category = df.groupby('category')['sale_price'].sum()

# Pie chart
plt.figure(figsize=(20, 15))
revenue_by_category.plot(kind='pie', autopct='%1.1f%%', startangle=90, colormap='coolwarm')
plt.title('Revenue Contribution by Category')
plt.ylabel('')  # Hide y-axis label
plt.show()


# In[ ]:


# Calculate average sale price by category
avg_sale_price = df.groupby('category')['sale_price'].mean().sort_values()

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=avg_sale_price.values, y=avg_sale_price.index, palette='viridis')
plt.title('Average Sale Price by Category')
plt.xlabel('Average Sale Price')
plt.ylabel('Category')
plt.show()


# ## feature: 'Sub-category'

# In[ ]:


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


# ## Feature: 'brand'

# In[ ]:


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


# In[ ]:


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


# In[ ]:


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


# # Feature: 'Sale Price'

# The Sale_Price feature in this dataset represents the price at which a product is being sold.
# It is a numerical feature and is likely measured in the currency relevant to the dataset (e.g., INR if it's an Indian dataset).

# In[ ]:


print(df['sale_price'].describe())


# Descriptive Statistics from above code
# 
# central tendency and spread of Sale_Price:
# 
# Mean: The average sale price of products.
# 
# Median: The midpoint value, less influenced by extreme values (outliers).
# 
# Min/Max: The range of sale prices.
# 
# Standard Deviation: How much the sale prices vary from the mean

# In[ ]:


plt.figure(figsize=(10, 6))
sns.histplot(df['sale_price'], bins=30, kde=True, color='blue')
plt.title("Distribution of Sale Price", fontsize=16)
plt.xlabel("Sale Price")
plt.ylabel("Frequency")
plt.show()


#  the distribution of Sale_Price is Normal from above figure which is bell shape 

# In[ ]:


sns.scatterplot(data=df, x='market_price', y='sale_price', hue='rating', palette='coolwarm')
plt.title("Sale Price vs Market Price by Rating", fontsize=16)
plt.xlabel("Market Price")
plt.ylabel("Sale Price")
plt.show()


# Products with higher Market Price tend to have higher Sale Price, as expected.
# 
# The clustering of points near the diagonal suggests that, for many products, the Sale Price is close to the Market Price. 
# 
# Below the diagonal: These points indicate products with discounts, where the Sale Price is less than the Market Price.

# In[ ]:


sns.boxplot(data=df, y='sale_price', color='orange')
plt.title("Outliers in Sale Price", fontsize=16)
plt.ylabel("Sale Price")
plt.show()


# # Feature: 'Market Price'

# The Market_Price feature in this dataset represents the original listed price of a product before any discounts or reductions.
# It is a numerical feature that serves as a baseline to compare with the actual selling price (Sale_Price).

# In[ ]:


print(df['market_price'].describe())


# In[ ]:


plt.figure(figsize=(10, 6))
sns.histplot(df['market_price'], bins=30, kde=True, color='green')
plt.title("Distribution of Market Price", fontsize=16)
plt.xlabel("Market Price")
plt.ylabel("Frequency")
plt.show()


# In[ ]:


plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, y='market_price', x='sale_price', color='blue', alpha=0.6)
plt.title("Market Price vs Sale Price", fontsize=16)
plt.xlabel("Sale Price")
plt.ylabel("Market Price")
plt.show()


# # Feature: 'Type'

# The Type feature in your dataset likely refers to the type or classification of the product, describing its nature, use case, or how it is categorized within the inventory.
# It is a categorical variable, and its values could represent product characteristics such as:
# "Fresh"
# "Packaged"
# "Frozen"
# "Organic"
# Or any other meaningful product classification.

# In[ ]:


print(df['type'].value_counts())


# In[ ]:


# Top 10 product types by count
top_10_types = df['type'].value_counts().head(10)

# Display the result
print(top_10_types)


# In[ ]:


# Plot the top 10 product types
plt.figure(figsize=(10, 6))
sns.barplot(x=top_10_types.values, y=top_10_types.index, palette='Blues_r')
plt.title("Top 10 Product Types by Count", fontsize=16)
plt.xlabel("Count")
plt.ylabel("Product Type")
plt.show()


# # MODEL CREATION

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor


# In[ ]:


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


# In[ ]:



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


# In[ ]:



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


# In[ ]:


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


# In[ ]:


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


# # MODEL EVALUATION

# In[ ]:


# Summarizing results
results = {
    "Model": ["Linear Regression", "Random Forest", "Gradient Boosting", "XGBoost"],
    "R^2 Score": [linear_r2, rf_r2, gb_r2, xgb_r2],
    "RMSE": [linear_rmse, rf_rmse, gb_rmse, xgb_rmse]
}

results_df = pd.DataFrame(results)
print(results_df)


# # Model Performance Visualisations

# In[ ]:


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


# In[ ]:


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


# # Visualisation of Linear Regression

# In[ ]:


plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_linear, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color='red')
plt.title("Linear Regression: Predictions vs Actual Values")
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.show()

