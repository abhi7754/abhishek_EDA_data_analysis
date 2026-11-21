"""
Retail Customer Sales: Exploratory Data Analysis (Project 1)
Name : Abhishek Sharma
ID   : 2026M01410110

This is the same analysis as the notebook, written as one script.
It audits and cleans the data, runs the 25 test cases and the fairness check,
then saves the cleaned data and one PDF with all charts.

How to run: open the main project folder in VS Code, open this file and press Run.
Needs: pandas, numpy, matplotlib, seaborn, scipy (the next lines install them if missing)
"""
import importlib.util
import os
import subprocess
import sys

for package in ['pandas', 'numpy', 'matplotlib', 'seaborn', 'scipy']:
    if importlib.util.find_spec(package) is None:
        print('Installing', package, '...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '-q'])

import matplotlib
matplotlib.use('Agg')            # draw charts without opening windows
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def display(x):
    """Small helper so display() also works outside Jupyter."""
    print(x.to_string() if hasattr(x, 'to_string') else x)


# ======================================================================
# 0. Install the libraries (run this cell first)
# ======================================================================


# ======================================================================
# 1. Setup and First Look
# ======================================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')
plt.rcParams.update({'axes.titlesize': 13, 'axes.titleweight': 'bold', 'axes.labelsize': 11})
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: f'{x:,.2f}')

# One consistent colour scheme for the whole notebook
TEAL, AMBER, GREY, BRICK = '#0F766E', '#D98B16', '#9AA39F', '#B4432C'

# Find the main project folder. This works if VS Code starts inside the notebook folder or in the main folder.
import glob
found = [p for p in ['.', '..'] if os.path.isdir(p + '/02_Original_Dataset')]
found += [os.path.dirname(p) for p in glob.glob('*/02_Original_Dataset')]
if not found:
    raise FileNotFoundError('Open the main project folder in VS Code and keep the numbered folders as they are.')
ROOT = found[0]
RAW_FILE = ROOT + '/02_Original_Dataset/Abhishek_sharma_2026M01410110_Original_Dataset.csv'
CLEAN_FILE = ROOT + '/07_Cleaned_Dataset/Abhishek_sharma_2026M01410110_Cleaned_Dataset.csv'
CHARTS_FILE = ROOT + '/09_Charts/Abhishek_sharma_2026M01410110_Charts.pdf'

# Every chart is remembered here with a short caption. All charts are saved as one PDF near the end.
CHARTS = []
def keep_chart(caption):
    CHARTS.append((plt.gcf(), caption))

raw = pd.read_csv(RAW_FILE)   # untouched copy; I never edit this
print('Shape (rows, columns):', raw.shape)

print(raw.dtypes)

display(raw.head())
display(raw.tail())


# ======================================================================
# 2. Data Quality Audit
# ======================================================================


# ---- 2.1 Missing values ----

missing = pd.DataFrame({'missing_count': raw.isna().sum(),
                        'missing_%': (raw.isna().mean() * 100).round(2)})
missing = missing[missing.missing_count > 0].sort_values('missing_count', ascending=False)
display(missing)

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(missing.index[::-1], missing['missing_%'][::-1], color=TEAL)
for i, v in enumerate(missing['missing_%'][::-1]):
    ax.text(v + 0.05, i, f'{v:.1f}%', va='center', fontsize=9)
ax.set_xlabel('Missing values (% of rows)')
ax.set_title('Which columns have missing values?')
plt.tight_layout()
keep_chart('Missing values: which columns have gaps (as % of rows)')
plt.close()


# ---- 2.2 Duplicate rows ----

print('Exact duplicate rows            :', raw.duplicated().sum())
print('Repeated Transaction_ID values  :', raw.Transaction_ID.duplicated().sum())
print('Unique Transaction_IDs          :', raw.Transaction_ID.nunique())

# Transaction_ID should be unique. Some IDs repeat but the rows are not exact copies. Why?
dup_ids = raw[raw.Transaction_ID.duplicated(keep=False)]
not_exact = dup_ids[~dup_ids.duplicated(keep=False)]
print('Rows with repeated ID that are NOT exact copies:', len(not_exact))
display(not_exact.sort_values('Transaction_ID').head(4)[['Transaction_ID', 'Order_Date', 'Gender', 'Region', 'Sales']])


# ---- 2.3 Inconsistent category labels ----

print(raw.Gender.value_counts(), '\n')
print(raw.Region.value_counts())


# ---- 2.4 Impossible and out-of-range values ----

bad_age = raw[(raw.Age <= 0) | (raw.Age > 100)]
bad_disc = raw[(raw.Discount < 0) | (raw.Discount > 0.5)]
print('Impossible Age values (<=0 or >100):', len(bad_age), '->', sorted(bad_age.Age.tolist()))
print('Invalid Discount values (<0 or >0.5):', len(bad_disc), '->', sorted(bad_disc.Discount.tolist()))
print()
print('Age range of the remaining rows    :', raw.Age[(raw.Age > 0) & (raw.Age <= 100)].min(), 'to', raw.Age[(raw.Age > 0) & (raw.Age <= 100)].max())
print('Quantity range                     :', raw.Quantity.min(), 'to', raw.Quantity.max())
print('Rows with Unit_Price <= 0 or Sales <= 0:', ((raw.Unit_Price <= 0) | (raw.Sales <= 0)).sum())
print('Rows where Profit != Sales - Cost  :', ((raw.Profit - (raw.Sales - raw.Cost)).abs() > 0.02).sum())


# ---- 2.5 Extreme Income values ----

print(raw.Income.describe())
print('\nTop 10 incomes:')
print(raw.Income.nlargest(10).round(0).tolist())

# Income is a property of the customer, so it should not change between a customer's orders
inc_per_cust = raw.groupby('Customer_ID').Income.agg(['min', 'max', 'nunique'])
changing = inc_per_cust[inc_per_cust['nunique'] > 1]
print('\nCustomers whose Income differs between their own orders:', len(changing))
display(changing)


# ---- 2.6 Date format problems ----

def date_pattern(s):
    s = s.dropna().astype(str)
    return (s.str.replace(r'\d', '9', regex=True)
             .str.replace(r'[A-Za-z]{3}', 'Mon', regex=True))

print('Order_Date formats:')
print(date_pattern(raw.Order_Date).value_counts(), '\n')
print('Delivery_Date formats:')
print(date_pattern(raw.Delivery_Date).value_counts())

def parse_mixed_date(s):
    """Parse a column that mixes four date formats into one datetime column."""
    s = s.astype('string').str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype='datetime64[ns]')
    rules = [(r'^\d{4}-\d{2}-\d{2}$',   '%Y-%m-%d'),   # 2023-12-19
             (r'^\d{2}/\d{2}/\d{4}$',   '%d/%m/%Y'),   # 19/12/2023  (day first)
             (r'^\d{2}-\d{2}-\d{4}$',   '%m-%d-%Y'),   # 12-19-2023  (month first)
             (r'^\d{2}-[A-Za-z]{3}-\d{4}$', '%d-%b-%Y')] # 19-Dec-2023
    for pattern, fmt in rules:
        m = s.str.match(pattern).fillna(False).astype(bool)
        out[m] = pd.to_datetime(s[m], format=fmt, errors='coerce')
    return out

order_dt = parse_mixed_date(raw.Order_Date)
deliv_dt = parse_mixed_date(raw.Delivery_Date)

# Proof that my reading of the formats is right:
# 1) every date must parse and fall inside 2023-2024
print('Order_Date not parsed          :', order_dt.isna().sum())
print('Order_Date range               :', order_dt.min().date(), 'to', order_dt.max().date())
# 2) duplicate copies of the same transaction (written in different formats) must agree
chk = pd.DataFrame({'id': raw.Transaction_ID, 'd': order_dt})
chk = chk[chk.id.duplicated(keep=False)].groupby('id').d.nunique()
print('Duplicate IDs whose copies disagree on date:', (chk > 1).sum(), '(0 means our format rules are correct)')

days = (deliv_dt - order_dt).dt.days
print('\nDelivery_Days summary:')
print(days.describe())
print('Delivered BEFORE ordered (negative days):', (days < 0).sum())


# ======================================================================
# 3. Data Cleaning
# ======================================================================


# ---- 3.1 Standardise Gender and Region ----

df = raw.copy()
cleaning_log = []

gender_map = {'male': 'Male', 'm': 'Male', 'female': 'Female', 'f': 'Female'}
region_map = {'north': 'North', 'nort': 'North', 'south': 'South', 'sth': 'South',
              'east': 'East', 'west': 'West'}

df['Gender'] = df.Gender.str.strip().str.lower().map(gender_map)
df['Region'] = df.Region.str.strip().str.lower().map(region_map)

print('Unmapped Gender:', df.Gender.isna().sum(), '| Unmapped Region:', df.Region.isna().sum())
print(df.Gender.value_counts().to_dict())
print(df.Region.value_counts().to_dict())
cleaning_log.append(('Gender / Region', '8 and 12 spellings', 'Mapped to 2 and 4 standard labels', 'Same value written differently; mapping loses no information'))


# ---- 3.2 Convert dates to proper datetime ----

df['Order_Date'] = parse_mixed_date(raw.Order_Date)
df['Delivery_Date'] = parse_mixed_date(raw.Delivery_Date)
print(df[['Order_Date', 'Delivery_Date']].dtypes)

# Delivery before order is impossible. I do not guess the right date, so I set it to missing and flag it.
bad_delivery = df.Delivery_Date < df.Order_Date
df['Delivery_Date_Fixed'] = bad_delivery
df.loc[bad_delivery, 'Delivery_Date'] = pd.NaT
print('Delivery dates earlier than order date, set to missing:', bad_delivery.sum())
cleaning_log.append(('Order_Date / Delivery_Date', '4 mixed formats; 15 deliveries before order', 'Parsed to datetime; 15 bad delivery dates set to missing and flagged', 'Real delivery date is unknown, so inventing one would be worse than leaving it blank'))


# ---- 3.3 Remove duplicates ----

before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f'Rows before: {before:,}  |  rows after: {len(df):,}  |  removed: {before - len(df)}')
print('Transaction_ID is now unique:', df.Transaction_ID.is_unique)
cleaning_log.append(('Duplicate rows', '84 repeated rows (61 exact, 23 formatting-only)', 'Dropped, kept first copy', 'A transaction must appear once; double counting would inflate Sales and Profit'))


# ---- 3.4 Age: impossible values and missing values ----

age_valid = df.Age.between(18, 100)
print('Rows with impossible age:', ((df.Age.notna()) & ~age_valid).sum(), '| missing age:', df.Age.isna().sum())

df.loc[~age_valid, 'Age'] = np.nan                     # impossible -> treat as missing
df['Age_Imputed'] = df.Age.isna()                      # flag every row I fill
age_from_customer = df.groupby('Customer_ID').Age.transform('median')
df['Age'] = df.Age.fillna(age_from_customer)           # 1st choice: same customer
still_missing = df.Age.isna().sum()
df['Age'] = df.Age.fillna(df.Age.median())             # last resort: overall median
print('Filled from same customer:', df.Age_Imputed.sum() - still_missing, '| filled with overall median:', still_missing)
print('Age range now:', df.Age.min(), 'to', df.Age.max())
cleaning_log.append(('Age', '6 impossible (-12 to 999) + 142 missing', 'Impossible -> missing; filled from the same customer, flagged in Age_Imputed', 'Age is constant per customer, so this is exact, not a guess'))


# ---- 3.5 Income: extreme errors and missing values ----

cust_min_income = df.groupby('Customer_ID').Income.transform('min')
income_error = df.Income > 3 * cust_min_income          # far above the same customer's own lowest value
print('Income values far above the same customer\'s other orders:', income_error.sum())
display(df.loc[income_error, ['Customer_ID', 'Income']].assign(Normal_Income=cust_min_income[income_error]))

df.loc[income_error, 'Income'] = np.nan
df['Income_Imputed'] = df.Income.isna()
income_from_customer = df.groupby('Customer_ID').Income.transform('median')
df['Income'] = df.Income.fillna(income_from_customer)
left = df.Income.isna().sum()
seg_median = df.groupby('Customer_Segment').Income.transform('median')
df['Income'] = df.Income.fillna(seg_median)
print('Missing/erroneous incomes:', df.Income_Imputed.sum(), '| recovered from same customer:', df.Income_Imputed.sum() - left, '| filled with segment median:', left)
cleaning_log.append(('Income', '9 extreme errors (up to 17.5 lakh) + 253 missing', 'Errors and gaps filled from the same customer; 6 leftovers with segment median; flagged in Income_Imputed', 'Income is constant per customer; median (not mean) is used because income is right-skewed'))


# ---- 3.6 Discount: invalid and missing values ----

implied_discount = (1 - df.Sales / (df.Quantity * df.Unit_Price)).round(2)
good = df.Discount.between(0, 0.5)
print('Formula check on valid rows: max difference =', (implied_discount[good] - df.Discount[good]).abs().max().round(3))

bad_disc = df.Discount.notna() & ~good
print('Invalid discounts:', bad_disc.sum(), '| missing discounts:', df.Discount.isna().sum())
df.loc[bad_disc, 'Discount'] = np.nan
df['Discount_Imputed'] = df.Discount.isna()
df['Discount'] = df.Discount.fillna(implied_discount.clip(0, 0.5))
print('Discount range now:', df.Discount.min(), 'to', df.Discount.max())
cleaning_log.append(('Discount', '6 invalid (-0.2 to 2.0) + 71 missing', 'Recomputed from Sales / (Quantity x Unit_Price); flagged in Discount_Imputed', 'Formula matches recorded discount on valid rows, so the rebuilt value is reliable'))


# ---- 3.7 Rating and Satisfaction: keep the gaps honest ----

df['Rating_Given'] = df.Customer_Rating.notna()
df['Satisfaction_Level'] = df.Satisfaction_Level.fillna('Not Recorded')
print('Orders without a rating:', (~df.Rating_Given).sum(), f'({(~df.Rating_Given).mean():.1%})')
print(df.Satisfaction_Level.value_counts().to_dict())
cleaning_log.append(('Customer_Rating', '428 missing (6%)', 'Left missing, flag Rating_Given added', 'A rating is the customer\'s opinion; inventing it would create fake satisfaction'))
cleaning_log.append(('Satisfaction_Level', '179 missing', 'Labelled "Not Recorded"', 'Keeps the rows and keeps the gap visible instead of hiding it'))


# ---- 3.8 Check the cleaned data and show the cleaning log ----

# Final validation checks
print('Duplicates left          :', df.duplicated().sum())
print('Transaction_ID unique    :', df.Transaction_ID.is_unique)
print('Gender values            :', sorted(df.Gender.unique()))
print('Region values            :', sorted(df.Region.unique()))
print('Age range                :', df.Age.min(), '-', df.Age.max())
print('Discount range           :', df.Discount.min(), '-', df.Discount.max())
print('Max Income               :', round(df.Income.max()))
print('Sales formula max error  :', (df.Sales - df.Quantity * df.Unit_Price * (1 - df.Discount)).abs().max().round(2))
print('Rows x columns           :', df.shape)
print('Missing values left      :')
print(df.isna().sum()[lambda s: s > 0].to_dict())

log = pd.DataFrame(cleaning_log, columns=['Column', 'Problem found', 'Action taken', 'Why'])
pd.set_option('display.max_colwidth', 80)
display(log)


# ======================================================================
# 4. Descriptive Statistics and Univariate Analysis
# ======================================================================


# ---- 4.1 Descriptive statistics of the key numerical variables ----

num_cols = ['Age', 'Income', 'Quantity', 'Unit_Price', 'Discount', 'Sales', 'Cost', 'Profit']

def full_describe(s):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    return pd.Series({'mean': s.mean(), 'median': s.median(), 'mode': s.mode().iloc[0],
                      'std': s.std(), 'variance': s.var(), 'min': s.min(),
                      'P5': s.quantile(0.05), 'Q1 (25%)': q1, 'Q3 (75%)': q3, 'P95': s.quantile(0.95),
                      'max': s.max(), 'IQR': q3 - q1, 'skewness': s.skew()})

stats = df[num_cols].apply(full_describe).T
display(stats[['mean', 'median', 'mode', 'std', 'variance']])
print()
display(stats[['min', 'P5', 'Q1 (25%)', 'Q3 (75%)', 'P95', 'max', 'IQR', 'skewness']])


# ---- 4.2 Distribution shape of Age, Income, Sales and Profit ----

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, col in zip(axes.ravel(), ['Age', 'Income', 'Sales', 'Profit']):
    sns.histplot(df[col], bins=40, kde=True, color=TEAL, ax=ax)
    ax.axvline(df[col].mean(), color=BRICK, ls='--', lw=1.5, label=f'Mean {df[col].mean():,.0f}')
    ax.axvline(df[col].median(), color=AMBER, ls='-', lw=1.5, label=f'Median {df[col].median():,.0f}')
    ax.set_title(f'{col}: skewness = {df[col].skew():.2f}')
    ax.set_xlabel(col + (' (years)' if col == 'Age' else ' (Rs.)'))
    ax.set_ylabel('Number of transactions')
    ax.legend()
plt.tight_layout()
keep_chart('Shape of Age, Income, Sales and Profit. The mean (red) sits right of the median (amber) when a few large values pull it up.')
plt.close()

print(df[['Age', 'Income', 'Sales', 'Profit']].skew().round(2).to_dict())


# ---- 4.3 Categorical variables: frequency and proportion ----

for col in ['Region', 'Product_Category', 'Customer_Segment']:
    t = pd.DataFrame({'count': df[col].value_counts(), 'share_%': (df[col].value_counts(normalize=True) * 100).round(1)})
    print(col); display(t); print()

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, col in zip(axes, ['Region', 'Product_Category', 'Customer_Segment']):
    order = df[col].value_counts().index
    sns.countplot(data=df, x=col, order=order, color=TEAL, ax=ax)
    for p in ax.patches:
        ax.annotate(f'{p.get_height() / len(df):.0%}', (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom', fontsize=9)
    ax.set_title(f'Transactions by {col.replace("_", " ")}')
    ax.set_xlabel('')
    ax.set_ylabel('Number of transactions')
    ax.tick_params(axis='x', rotation=20)
plt.tight_layout()
keep_chart('Number of transactions by region, product category and customer segment')
plt.close()


# ======================================================================
# 5. Outlier Detection
# ======================================================================

def outlier_report(s):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    z = (s - s.mean()) / s.std()
    return pd.Series({'lower fence': low, 'upper fence': high,
                      'IQR outliers': int(((s < low) | (s > high)).sum()),
                      'IQR outliers %': round(((s < low) | (s > high)).mean() * 100, 1),
                      'Z-score outliers (|z|>3)': int((z.abs() > 3).sum())})

display(df[['Sales', 'Profit', 'Age', 'Income']].apply(outlier_report).T)

fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
for ax, col in zip(axes, ['Sales', 'Profit', 'Age', 'Income']):
    sns.boxplot(y=df[col], color='#BFE0DC', flierprops=dict(marker='o', markerfacecolor=AMBER, markersize=3, alpha=0.6), ax=ax)
    ax.set_title(f'{col}: spread and outliers')
    ax.set_ylabel(col + (' (years)' if col == 'Age' else ' (Rs.)'))
plt.tight_layout()
keep_chart('Boxplots of Sales, Profit, Age and Income. The dots are outliers by the IQR rule.')
plt.close()


# ---- Are these outliers genuine or errors? Checking the evidence ----

sales_high = df.Sales > df.Sales.quantile(0.75) + 1.5 * (df.Sales.quantile(0.75) - df.Sales.quantile(0.25))
print('Sales outliers by category (% of that category\'s orders):')
print((df[sales_high].Product_Category.value_counts() / df.Product_Category.value_counts() * 100).round(1).dropna().to_dict())
print('Average Quantity: outlier orders =', df[sales_high].Quantity.mean().round(2), '| all orders =', df.Quantity.mean().round(2))
print('Average Unit_Price: outlier orders =', df[sales_high].Unit_Price.mean().round(0), '| all orders =', df.Unit_Price.mean().round(0))
print('Sales formula still holds for outlier rows (max error):', (df[sales_high].Sales - df[sales_high].Quantity * df[sales_high].Unit_Price * (1 - df[sales_high].Discount)).abs().max().round(2))
print()
print('Age outliers (ages):', sorted(df.Age[(df.Age < 29 - 1.5 * 14) | (df.Age > 43 + 1.5 * 14)].unique().tolist()))
inc_hi = df.Income > df.Income.quantile(0.75) + 1.5 * (df.Income.quantile(0.75) - df.Income.quantile(0.25))
print('Income outliers:', inc_hi.sum(), '| largest income now:', round(df.Income.max()), '| Income outliers that are flagged as imputed:', df[inc_hi].Income_Imputed.sum())


# ======================================================================
# 6. Feature Engineering and Business Questions
# ======================================================================


# ---- 6.1 New calculated columns ----

# 1) Profit_Margin: how many rupees of profit per rupee of sales
df['Profit_Margin'] = df.Profit / df.Sales

# 2) Delivery_Days: days between order and delivery (missing where the date is missing or was invalid)
df['Delivery_Days'] = (df.Delivery_Date - df.Order_Date).dt.days

# 3) Revenue_per_Customer: total sales of the customer across all their orders
df['Revenue_per_Customer'] = df.groupby('Customer_ID').Sales.transform('sum')

# 4) Orders_per_Customer: how many orders that customer placed
df['Orders_per_Customer'] = df.groupby('Customer_ID').Transaction_ID.transform('count')

# 5) Time parts, for trend analysis
df['Order_Year'] = df.Order_Date.dt.year
df['Order_Month'] = df.Order_Date.dt.to_period('M').astype(str)

# 6) Discount_Band: group the discount into readable ranges
df['Discount_Band'] = pd.cut(df.Discount, bins=[-0.001, 0.10, 0.20, 0.30, 0.50],
                             labels=['0-10%', '10-20%', '20-30%', '30-50%'])

display(df[['Transaction_ID', 'Sales', 'Profit', 'Profit_Margin', 'Delivery_Days', 'Revenue_per_Customer', 'Orders_per_Customer', 'Discount_Band']].head())
print(df[['Profit_Margin', 'Delivery_Days']].describe().round(3))


# ---- 6.2 Correlation heat map ----

from matplotlib.colors import LinearSegmentedColormap

corr_cols = ['Age', 'Income', 'Quantity', 'Unit_Price', 'Discount', 'Sales', 'Cost', 'Profit', 'Customer_Rating', 'Delivery_Days']
corr = df[corr_cols].corr()                      # Pearson correlation for every pair of columns

# Amber means a negative link, teal means a positive link, white means no link
teal_amber = LinearSegmentedColormap.from_list('teal_amber', [AMBER, '#FBF7EF', TEAL])

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap=teal_amber, vmin=-1, vmax=1, linewidths=0.5,
            square=True, cbar_kws={'label': 'Correlation'}, annot_kws={'size': 9})
plt.title('Correlation heat map of the numeric columns')
plt.tight_layout()
keep_chart('Correlation heat map. Sales, Cost and Profit move together; Unit_Price drives Sales; customer details show almost no link.')
plt.close()

# Spearman (rank based) is less affected by a few very large orders, so I check it too
print(df[corr_cols].corr(method='spearman').loc[['Sales'], ['Quantity', 'Unit_Price', 'Discount']].round(2))


# ---- 6.3 Business questions using groupby ----

q1 = (df.groupby('Region')
        .agg(Orders=('Sales', 'size'), Total_Sales=('Sales', 'sum'), Avg_Order_Value=('Sales', 'mean'), Median_Order=('Sales', 'median'))
        .sort_values('Avg_Order_Value', ascending=False))
display(q1.round(0))

fig, ax = plt.subplots(figsize=(7, 4.5))
colors = [AMBER if r == q1.index[0] else GREY for r in q1.index]
ax.bar(q1.index, q1.Avg_Order_Value, color=colors)
for i, v in enumerate(q1.Avg_Order_Value):
    ax.text(i, v + 80, f'Rs. {v:,.0f}', ha='center', fontsize=10)
ax.set_title('Average order value by region')
ax.set_ylabel('Average order value (Rs.)')
ax.set_ylim(0, q1.Avg_Order_Value.max() * 1.15)
plt.tight_layout()
keep_chart('Average order value by region. East is the highest and South the lowest.')
plt.close()

q2 = (df.groupby('Product_Category')
        .agg(Orders=('Sales', 'size'), Total_Sales=('Sales', 'sum'), Total_Profit=('Profit', 'sum'), Avg_Order_Value=('Sales', 'mean'))
        .assign(Profit_Margin_pct=lambda t: t.Total_Profit / t.Total_Sales * 100,
                Sales_Share_pct=lambda t: t.Total_Sales / t.Total_Sales.sum() * 100)
        .sort_values('Total_Sales', ascending=False))
display(q2.round(1))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].bar(q2.index, q2.Total_Sales / 1e6, color=[TEAL if i < 2 else GREY for i in range(len(q2))])
axes[0].set_title('Total sales by product category')
axes[0].set_ylabel('Total sales (Rs. million)')
axes[0].tick_params(axis='x', rotation=20)
low_margin = q2.Profit_Margin_pct.round(1) == q2.Profit_Margin_pct.round(1).min()
axes[1].bar(q2.index, q2.Profit_Margin_pct, color=[AMBER if low_margin[i] else GREY for i in q2.index])
axes[1].set_title('Profit margin by product category')
axes[1].set_ylabel('Profit margin (%)')
axes[1].set_ylim(0, 40)
axes[1].tick_params(axis='x', rotation=20)
plt.tight_layout()
keep_chart('Total sales and profit margin by product category. Electronics and Furniture bring most of the sales; margins are almost equal.')
plt.close()

q3 = (df.groupby('Customer_Segment')
        .agg(Orders=('Sales', 'size'), Customers=('Customer_ID', 'nunique'), Total_Sales=('Sales', 'sum'),
             Total_Profit=('Profit', 'sum'), Avg_Order_Value=('Sales', 'mean'))
        .assign(Sales_Share_pct=lambda t: t.Total_Sales / t.Total_Sales.sum() * 100))
display(q3.round(1))

q4 = (df.groupby('Delivery_Mode')
        .agg(Orders=('Sales', 'size'), Avg_Delivery_Days=('Delivery_Days', 'mean'),
             Median_Delivery_Days=('Delivery_Days', 'median'), Avg_Rating=('Customer_Rating', 'mean'))
        .reindex(['Same-Day', 'Express', 'Standard']))
display(q4.round(2))

q5 = (df.groupby('Discount_Band')
        .agg(Orders=('Sales', 'size'), Profit_Margin_pct=('Profit', 'sum'), Sales=('Sales', 'sum'),
             Return_Rate_pct=('Returned', lambda s: (s == 'Yes').mean() * 100), Avg_Rating=('Customer_Rating', 'mean')))
q5['Profit_Margin_pct'] = q5.Profit_Margin_pct / q5.Sales * 100
display(q5.drop(columns='Sales').round(2))

q6 = (df.groupby('Returned')
        .agg(Orders=('Sales', 'size'), Avg_Rating=('Customer_Rating', 'mean'),
             Low_Satisfaction_pct=('Satisfaction_Level', lambda s: (s == 'Low').mean() * 100)))
display(q6.round(2))
print('Overall return rate: %.1f%%' % ((df.Returned == 'Yes').mean() * 100))
print(df.groupby('Product_Category').Returned.apply(lambda s: (s == 'Yes').mean() * 100).round(1).to_dict())

yearly = df.groupby('Order_Year').agg(Orders=('Sales', 'size'), Sales=('Sales', 'sum'), Profit=('Profit', 'sum'))
display(yearly.round(0))

monthly = df.groupby('Order_Month').Sales.sum() / 1e6
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(monthly.index, monthly.values, marker='o', color=TEAL, lw=2)
ax.axhline(monthly.mean(), color=GREY, ls='--', label=f'Monthly average (Rs. {monthly.mean():.1f} M)')
ax.set_title('Monthly sales, 2023-2024')
ax.set_ylabel('Total sales (Rs. million)')
ax.set_xlabel('Month')
ax.set_xticks(range(0, len(monthly), 2))
ax.set_xticklabels(monthly.index[::2], rotation=45)
ax.set_ylim(0, monthly.max() * 1.2)
ax.legend()
plt.tight_layout()
keep_chart('Monthly sales, 2023 to 2024. Sales move in a narrow band with no clear season.')
plt.close()

cust = df.groupby('Customer_ID').agg(Orders=('Sales', 'size'), Revenue=('Sales', 'sum')).sort_values('Revenue', ascending=False)
top20 = cust.head(int(len(cust) * 0.2)).Revenue.sum() / cust.Revenue.sum()
print(f'Customers: {len(cust):,} | average orders per customer: {cust.Orders.mean():.1f} | average revenue per customer: Rs. {cust.Revenue.mean():,.0f}')
print(f'Top 20% of customers bring {top20:.0%} of total sales')
display(cust.head(5).round(0))


# ======================================================================
# 7. Save Cleaned Data and Summary of Findings
# ======================================================================

# Keep the helper flag columns so every cleaning decision stays traceable
df_out = df.copy()
df_out['Order_Date'] = df_out.Order_Date.dt.strftime('%Y-%m-%d')
df_out['Delivery_Date'] = df_out.Delivery_Date.dt.strftime('%Y-%m-%d')
df_out.to_csv(CLEAN_FILE, index=False)
print('Saved cleaned dataset:', df_out.shape[0], 'rows x', df_out.shape[1], 'columns')


# ---- Summary: most important findings ----


# ======================================================================
# 8. Test Cases (Checking that my code and my cleaned data are correct)
# ======================================================================

import sys
import unittest

CORE_COLS = ['Transaction_ID', 'Customer_ID', 'Order_Date', 'Age', 'Gender', 'Income', 'Region', 'Customer_Segment',
             'Product_Category', 'Quantity', 'Unit_Price', 'Discount', 'Sales', 'Cost', 'Profit', 'Satisfaction_Level', 'Returned']

class TestCleanedRetailData(unittest.TestCase):

    # ---------- A. pandas ----------
    def test_A1_pandas_row_count_after_removing_duplicates(self):
        self.assertEqual(len(df), 7000)

    def test_A2_pandas_raw_data_is_untouched(self):
        self.assertEqual(raw.shape, (7084, 24))

    def test_A3_pandas_transaction_id_is_unique(self):
        self.assertTrue(df.Transaction_ID.is_unique)

    def test_A4_pandas_no_duplicate_rows(self):
        self.assertEqual(df.duplicated().sum(), 0)

    def test_A5_pandas_gender_labels_standardised(self):
        self.assertEqual(set(df.Gender), {'Male', 'Female'})

    def test_A6_pandas_region_labels_standardised(self):
        self.assertEqual(set(df.Region), {'North', 'South', 'East', 'West'})

    def test_A7_pandas_date_columns_are_datetime(self):
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df.Order_Date))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df.Delivery_Date))

    def test_A8_pandas_no_missing_values_in_core_columns(self):
        self.assertEqual(int(df[CORE_COLS].isna().sum().sum()), 0)

    def test_A9_pandas_profit_equals_sales_minus_cost(self):
        pd.testing.assert_series_equal((df.Sales - df.Cost).round(2), df.Profit.round(2), check_names=False)

    # ---------- B. numpy ----------
    def test_B1_numpy_age_in_valid_range(self):
        ages = df.Age.to_numpy()
        self.assertTrue(np.all((ages >= 18) & (ages <= 100)))

    def test_B2_numpy_discount_between_0_and_half(self):
        d = df.Discount.to_numpy()
        self.assertTrue(np.all((d >= 0) & (d <= 0.5)))

    def test_B3_numpy_sales_formula_holds(self):
        expected = (df.Quantity * df.Unit_Price * (1 - df.Discount)).to_numpy()
        np.testing.assert_allclose(df.Sales.to_numpy(), expected, rtol=0, atol=0.02)

    def test_B4_numpy_all_numbers_are_finite(self):
        self.assertTrue(np.isfinite(df[num_cols].to_numpy(dtype=float)).all())

    def test_B5_numpy_income_errors_are_gone(self):
        self.assertLess(df.Income.max(), 500_000)          # raw maximum was 17.5 lakh
        self.assertGreaterEqual(df.Income.min(), 15_000)

    # ---------- C. datetime ----------
    def test_C1_parse_mixed_date_reads_all_four_formats(self):
        samples = pd.Series(['2023-12-19', '19/12/2023', '12-19-2023', '19-Dec-2023'])
        parsed = parse_mixed_date(samples)
        self.assertTrue((parsed == pd.Timestamp(2023, 12, 19)).all())

    def test_C2_order_dates_inside_2023_2024(self):
        self.assertGreaterEqual(df.Order_Date.min(), pd.Timestamp('2023-01-01'))
        self.assertLessEqual(df.Order_Date.max(), pd.Timestamp('2024-12-31'))

    def test_C3_delivery_days_never_negative(self):
        self.assertGreaterEqual(df.Delivery_Days.min(), 0)

    # ---------- D. my own functions, tiny data with a known answer ----------
    def test_D1_full_describe_on_known_data(self):
        r = full_describe(pd.Series([1, 2, 3, 4, 5]))
        self.assertEqual(r['mean'], 3)
        self.assertEqual(r['median'], 3)
        self.assertEqual(r['IQR'], 2)

    def test_D2_outlier_report_finds_the_one_outlier(self):
        r = outlier_report(pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 100]))
        self.assertEqual(r['IQR outliers'], 1)

    def test_D3_outlier_report_finds_none_in_even_data(self):
        r = outlier_report(pd.Series([10, 11, 12, 13, 14, 15]))
        self.assertEqual(r['IQR outliers'], 0)

    # ---------- E. feature engineering ----------
    def test_E1_profit_margin_matches_manual_calculation(self):
        np.testing.assert_allclose(df.Profit_Margin, df.Profit / df.Sales)
        self.assertTrue(df.Profit_Margin.between(0, 1).all())

    def test_E2_delivery_days_matches_date_difference(self):
        row = df[df.Delivery_Date.notna()].iloc[0]
        self.assertEqual(row.Delivery_Days, (row.Delivery_Date - row.Order_Date).days)

    def test_E3_revenue_per_customer_matches_groupby_total(self):
        cust = df.Customer_ID.iloc[0]
        self.assertAlmostEqual(df[df.Customer_ID == cust].Revenue_per_Customer.iloc[0], df[df.Customer_ID == cust].Sales.sum(), places=2)

    # ---------- F. matplotlib / seaborn output and file I/O ----------
    def test_F1_charts_were_created(self):
        self.assertGreaterEqual(len(CHARTS), 7)
        for fig, caption in CHARTS:
            self.assertGreater(len(fig.axes), 0, msg='a chart is empty')
            self.assertGreater(len(caption), 10)

    def test_F2_cleaned_csv_reloads_correctly(self):
        back = pd.read_csv(CLEAN_FILE)
        self.assertEqual(back.shape, df.shape)
        self.assertEqual(set(back.Transaction_ID), set(df.Transaction_ID))

suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestCleanedRetailData)
result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
assert result.wasSuccessful(), 'Some tests failed, check the output above'


# ======================================================================
# 9. Fairness Check
# ======================================================================

from scipy import stats

fair = df.copy()          # work on a copy so the cleaned data stays as it is
fair['Age_Group'] = pd.cut(fair.Age, [17, 29, 44, 59, 100], labels=['18-29', '30-44', '45-59', '60+'])
fair['High_Discount'] = fair.Discount >= 0.30          # 30% or more counts as a big discount
fair['Is_Returned'] = fair.Returned == 'Yes'

def check_groups(col):
    """Compare the groups of one column and test if the gaps are real."""
    g = fair.groupby(col, observed=True)
    table = pd.DataFrame({
        'Orders': g.size(),
        'Avg_Discount_pct': g.Discount.mean() * 100,
        'High_Discount_pct': g.High_Discount.mean() * 100,
        'Avg_Rating': g.Customer_Rating.mean(),
        'Return_Rate_pct': g.Is_Returned.mean() * 100,
        'Avg_Delivery_Days': g.Delivery_Days.mean(),
    })
    parts = [x for _, x in fair.groupby(col, observed=True)]

    def numeric_test(metric):
        values = [x[metric].dropna() for x in parts]
        if len(values) == 2:
            return stats.ttest_ind(*values, equal_var=False).pvalue    # two groups: Welch t-test
        return stats.f_oneway(*values).pvalue                          # 3+ groups: ANOVA

    p = {
        'Avg_Discount_pct': numeric_test('Discount'),
        'Avg_Rating': numeric_test('Customer_Rating'),
        'Avg_Delivery_Days': numeric_test('Delivery_Days'),
        'Return_Rate_pct': stats.chi2_contingency(pd.crosstab(fair[col], fair.Is_Returned))[1],
        'High_Discount_pct': stats.chi2_contingency(pd.crosstab(fair[col], fair.High_Discount))[1],
    }
    not_returned = 100 - table.Return_Rate_pct
    ratio = not_returned.min() / not_returned.max()                    # four-fifths rule
    return table, p, ratio

all_p = {}
for col in ['Gender', 'Age_Group', 'Region', 'Customer_Segment']:
    table, p, ratio = check_groups(col)
    all_p[col] = p
    print('====', col, '====')
    display(table.round(2))
    print('p-values:', {k: round(float(v), 3) for k, v in p.items()})
    print('Four-fifths ratio (not returned):', round(ratio, 3), '-> passes' if ratio >= 0.8 else '-> FAILS')
    print()

n_tests = sum(len(p) for p in all_p.values())
small = [(c, m, round(float(v), 3)) for c, p in all_p.items() for m, v in p.items() if v < 0.05]
print('Tests run:', n_tests, '| p < 0.05:', small)
print('Expected by pure chance at the 5% level:', round(n_tests * 0.05, 1), '| Bonferroni limit:', round(0.05 / n_tests, 4))

old = fair[fair.Age_Group == '60+']
rate = old.Is_Returned.mean()
margin = 1.96 * np.sqrt(rate * (1 - rate) / len(old))
print('Age 60+: orders =', len(old), '| return rate = %.1f%% (95%% range %.1f%% to %.1f%%)' % (rate * 100, (rate - margin) * 100, (rate + margin) * 100))

rows = []
for col in ['Gender', 'Age_Group', 'Region', 'Customer_Segment']:
    g = fair.groupby(col, observed=True).agg(ret=('Is_Returned', 'mean'), disc=('Discount', 'mean'))
    for name, r in g.iterrows():
        rows.append((col.replace('_', ' ') + ': ' + str(name), r.ret * 100, r.disc * 100))
plot = pd.DataFrame(rows, columns=['Group', 'Return rate (%)', 'Average discount (%)'])

fig, axes = plt.subplots(1, 2, figsize=(11, 6.5), sharey=True)
for ax, col, overall in zip(axes, ['Return rate (%)', 'Average discount (%)'],
                            [fair.Is_Returned.mean() * 100, fair.Discount.mean() * 100]):
    ax.barh(plot.Group, plot[col], color=GREY)
    ax.axvline(overall, color=BRICK, ls='--', label=f'Overall {overall:.1f}%')
    ax.set_title(col)
    ax.set_xlim(0, plot[col].max() * 1.25)
    for i, v in enumerate(plot[col]):
        ax.text(v + 0.4, i, f'{v:.1f}', va='center', fontsize=8)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.06))
axes[0].invert_yaxis()
plt.suptitle('Fairness check: return rate and discount for every group', fontweight='bold')
plt.tight_layout()
keep_chart('Return rate and average discount for every customer group. The dashed line is the overall value.')
plt.close()


# ---- Save all charts in one PDF ----

from matplotlib.backends.backend_pdf import PdfPages

def save_charts_pdf(path):
    """Write every collected chart into one PDF, one chart per page, with its caption."""
    with PdfPages(path) as pdf:
        cover = plt.figure(figsize=(8.27, 4))
        cover.text(0.5, 0.62, 'Charts: Retail Customer Sales EDA', ha='center', fontsize=20, fontweight='bold', color=TEAL)
        cover.text(0.5, 0.45, 'Abhishek Sharma  |  ID 2026M01410110', ha='center', fontsize=12)
        cover.text(0.5, 0.33, str(len(CHARTS)) + ' charts, one per page', ha='center', fontsize=10, color=GREY)
        pdf.savefig(cover)
        plt.close(cover)
        for number, (fig, caption) in enumerate(CHARTS, 1):
            fig.text(0.5, -0.03, 'Chart ' + str(number) + '. ' + caption, ha='center', va='top',
                     fontsize=9, style='italic', color='#555555', wrap=True)
            pdf.savefig(fig, bbox_inches='tight')

save_charts_pdf(CHARTS_FILE)
print('Saved', len(CHARTS), 'charts in one PDF:', CHARTS_FILE)


# ======================================================================
# 10. Final Conclusion (in simple words)
# ======================================================================
