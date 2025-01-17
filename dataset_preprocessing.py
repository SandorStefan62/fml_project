import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Threshold for deleting entire row if too many NaNs present
threshold = 200

df = pd.read_csv('dataset/dataset.csv')

print(df)

# ================================
# Data Cleaning
# ================================

#Counts every NaN value for each column
nan_counts = df.isna().sum()
print("Missing values per column:")
print(nan_counts)

#Drops columns that have too many NaNs
columns_to_drop = nan_counts[nan_counts > threshold].index
print("\nColumns to drop (missing > {}) :".format(threshold))
print(columns_to_drop)

df.drop(columns=columns_to_drop, inplace=True)

print("\nShape of data after dropping columns:", df.shape)
print("Remaining columns:", df.columns.tolist())

#Filling in data in Age column based on Sex and Pclass columns
df['Age'] = df.groupby(['Sex','Pclass'])['Age'].transform(lambda grp: grp.fillna(grp.median()))

print(df)

#Dropping PassengerId, Name and Ticket columns because they serve no purpose
df.drop(['PassengerId', 'Name', 'Ticket'], axis = 1, inplace = True)
print("Columns after dropping unnecessary ones:", df.columns.tolist())

# ================================
# Basic Feature Engineering
# ================================

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df.drop(['SibSp', 'Parch'], axis = 1, inplace = True)
print(df)

# ================================
# Feature transformations
# ================================

#Label encoding
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2, 'NaN': 0})

# ================================
# Normalization
# ================================

numeric_cols = df.select_dtypes(include=[float, int]).columns.tolist()
numeric_cols.remove('Survived')  # don't scale the target

scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("\n--- After Scaling Numeric Columns ---")
print(df)

# ================================
# Splitting the dataset
# ================================

y = df['Survived']
X = df.drop('Survived', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2,   # 20% for testing
    random_state=42, # for reproducibility
    stratify=y       # keeps the proportion of Survived classes consistent
)

# ================================
# Saving split dataset
# ================================

X_train.to_csv('dataset/X_train.csv', index=False)
X_test.to_csv('dataset/X_test.csv', index=False)

y_train.to_csv('dataset/y_train.csv', index=False)
y_test.to_csv('dataset/y_test.csv', index=False)
