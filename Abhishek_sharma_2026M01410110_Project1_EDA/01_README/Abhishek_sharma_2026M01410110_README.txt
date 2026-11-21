RETAIL CUSTOMER SALES: EXPLORATORY DATA ANALYSIS (PROJECT 1)
Data Analyst Training Program, Mini Project 1

Name : Abhishek Sharma
ID   : 2026M01410110
GitHub: https://github.com/abhi7754/abhishek_EDA_data_analysis

WHAT THIS PROJECT IS
A retail company gave me two years of sales data (2023-2024, four regions of
India). The file was messy on purpose. I checked its quality, cleaned it with
reasons, tested my cleaning, and wrote down what it tells a manager.

Short result: 7,084 raw rows became 7,000 clean transactions. Sales about
Rs. 7.7 crore at a 33% margin. Electronics and Furniture bring about 83% of
sales. Same-Day and Express delivery are not faster than Standard. 12.6% of
orders are returned and those customers rate much lower. A fairness check found
no real difference between groups of customers.

FOLDERS (read in this order, one file in each)
01_README               This file
02_Original_Dataset     The data exactly as I received it (never edited)
03_Data_Dictionary      Meaning of every column and what I changed
04_EDA_Notebook         My full work with output (open this in VS Code)
05_Source_Code          The same work as one Python script
06_Cleaning_Log         Each problem, what I did, why, and rows affected
07_Cleaned_Dataset      7,000 clean rows (the notebook and script save it here)
08_Demo_Ranking         Rankings of regions, states, categories, products, segments, customers
09_Charts               All 9 charts in one PDF (the notebook and script save it here)
10_Fairness_Report      Do gender, age, region or segment get treated differently?
11_Execution_Output     Output of every notebook cell, readable without running it
12_Source_Code_PDF      The script printed with line numbers
13_EDA_Summary          One-page summary
14_Project_Report       The full report (4 pages)

HOW TO RUN IN VS CODE
1. Open the MAIN folder in VS Code (File > Open Folder). Keep the numbered
   folders as they are, because the code finds the data by folder name.
2. Notebook: open 04_EDA_Notebook, choose a Python kernel (top right) and press
   Run All. The first cell installs any missing library.
   Script: open 05_Source_Code and press Run. It also installs missing libraries,
   runs the 25 tests and the fairness check, and prints "OK" for the tests.
3. Needs Python 3.10 or later with internet only for the first install.
   Libraries: pandas, numpy, matplotlib, seaborn, scipy.

MY MAIN CLEANING DECISIONS (full list in 06_Cleaning_Log)
- Labels were standardised first and duplicates removed after, so look-alike
  copies were caught too.
- Dates came in 4 formats. I checked my reading of them by comparing duplicate
  copies of the same order.
- Age and Income do not change for a customer, so wrong or missing values were
  taken from the same customer's other orders.
- A wrong or missing Discount was rebuilt from
  Sales = Quantity x Unit_Price x (1 - Discount).
- Missing Customer_Rating was left blank, not invented.

HONEST LIMITS
The dataset is simulated and its patterns are very even, so small gaps (for
example between regions) may be chance. A statistical test is needed to
confirm them. The fairness check is a first look, not a proof.
