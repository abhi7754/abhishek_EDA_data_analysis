 Retail Customer Sales: Exploratory Data Analysis

Name: Abhishek Sharma  |  ID: 2026M01410110
Data Analyst Training Program, Mini Project 1**

I cleaned and analysed a deliberately messy retail sales file (7,084 rows, 2023-2024, four regions of India), tested my cleaning with 25 test cases, and checked the data for fairness across customer groups.

 Result in short
- 7,084 raw rows became **7,000 clean transactions**.
- Sales are about **Rs. 7.7 crore** at a **33% margin**. Electronics and Furniture bring about 83% of sales.
- Same-Day and Express delivery are **not faster** than Standard (about 5.7 days each).
- 12.6% of orders are returned, and those customers rate much lower (2.1 vs 3.2).
- No real difference was found between gender, age group, region or customer segment.

Folders (read in this order, one file in each)
| Folder | What it has |
|---|---|
| `01_README` | Full project guide |
| `02_Original_Dataset` | The data exactly as received |
| `03_Data_Dictionary` | Meaning of every column |
| `04_EDA_Notebook` | The complete analysis (open this first) |
| `05_Source_Code` | The same work as one Python script |
| `06_Cleaning_Log` | Every cleaning decision and why |
| `07_Cleaned_Dataset` | The 7,000 clean rows |
| `08_Demo_Ranking` | Rankings of regions, states, products, customers |
| `09_Charts` | All 9 charts in one PDF |
| `10_Fairness_Report` | Fairness check |
| `11_Execution_Output` | Notebook output as PDF |
| `12_Source_Code_PDF` | The script as PDF |
| `13_EDA_Summary` | One-page summary |
| `14_Project_Report` | Full report |

 How to run
1. Download or clone this repository and keep the numbered folders as they are.
2. Open the folder in VS Code.
3. Open the notebook in `04_EDA_Notebook` and press **Run All**, or run the script in `05_Source_Code`. Missing libraries (pandas, numpy, matplotlib, seaborn, scipy) are installed automatically.

The dataset is simulated and its patterns are very even, so small gaps should not be over-read.
