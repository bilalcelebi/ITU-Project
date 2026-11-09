import pandas as pd

def execute_conversion(data, dtypes):

    for i, row in dtypes.iterrows():
        if row['Variable'] in data.columns.tolist():
            data[row['Variable']] = data[row['Variable']].astype(row['Type'])
            print(f"{row['Variable']} casted to {row['Type']}")


def execute_drop(data, columns):

    for i, row in columns.iterrows():
        if row['Drop'] == True and row['Column'] in data.columns.tolist():
            data.drop(columns = [row['Column']], inplace = True)


def execute_filling(data, filling):

    for i, row in filling.iterrows():

        num_cols = data.select_dtypes(include = ['number']).columns.tolist()
        cat_cols = data.select_dtypes(include = ['object', 'category']).columns.tolist()

        strategy = row['Null Strategy']
        col = row['Variable']

        if strategy == 'Median' and col in num_cols:
            data[col] = data[col].fillna(data[col].median())
        elif strategy == 'Mean' and col in num_cols:
            data[col] = data[col].fillna(data[col].mean())
        elif strategy == 'Unkown' and col in cat_cols:
            data[col] = data[col].fillna("Unknowm")
        else:
            pass


def execute_outliers(data, outliers):
    for col in outliers:
        if col in data.select_dtypes(include = ['number']).columns.tolist():
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            finded = data[(data[col] < lower) | (data[col] > upper)]
            data.drop(finded.index.tolist(), inplace = True)

#Custom Hardcoded
def feature_engineering(df):

    df['Balance_to_Salary_Ratio'] = df['balance'] / df['estimated_salary']
    df['CreditScore_to_Age'] = df['credit_score'] / df['age']
    df['Is_Zero_Balance'] = (df['balance'] == 0).astype(int)
    df['Tenure_to_Age'] = df['tenure'] / df['age']
    df['Products_Per_Tenure'] = df['products_number'] / (df['tenure'] + 1)
    df['Age_Group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 100], labels=['Genc', 'Yetiskin', 'Orta_Yas', 'Yaşli'])
    df['Credit_Score_Group'] = pd.cut(df['credit_score'], bins=[0, 580, 670, 740, 850], labels=['Riskli', 'Orta', 'Iyi', 'Mukemmel'])
    df['Product_to_Age'] = df['products_number'] / df['age']