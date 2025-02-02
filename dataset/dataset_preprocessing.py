import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from torch.utils.data import Dataset, DataLoader

"""
Titanic - Machine Learning from Disaster 

Data Dictionary:

Variable | Definition                                  | Key
survival |	Survival                                   |	0 = No, 1 = Yes
pclass   |	Ticket class                               |	1 = 1st, 2 = 2nd, 3 = 3rd
sex      |	Sex	                                       |  
Age      |	Age in years                               |	
sibsp    |	# of siblings / spouses aboard the Titanic |
parch    |	# of parents / children aboard the Titanic |	
ticket   |	Ticket number	                           |
fare     |	Passenger fare	                           |
cabin    |	Cabin number	                           |
embarked |	Port of Embarkation	                       | C = Cherbourg, Q = Queenstown, S = Southampton
"""

class TitanicDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
def preprocess_data(data, for_rf = False):
    """
    Preprocess the dataset by handling missing values,
    encoding categorical variables and scaling numerical features.
    """

    df = data.copy()

    # Select usable features, removed "Name", "Ticket", and "Cabin" because we considered they were of no use for the classification
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize']

    # Handling missing features
    numeric_features = ['Age', 'Fare']
    categorical_features = ['Embarked']

    # Imputing numeric features with median
    numeric_imputer = SimpleImputer(strategy = 'median')
    df[numeric_features] = numeric_imputer.fit_transform(df[numeric_features])

    # Imputing categorical features with most frequent value
    categorical_imputer = SimpleImputer(strategy = 'most_frequent')
    df[categorical_features] = categorical_imputer.fit_transform(df[categorical_features])

    # Encoding categorical variables 

    label_encoder = LabelEncoder()
    df['Embarked'] = label_encoder.fit_transform(df['Embarked'])
    df['Sex'] = label_encoder.fit_transform(df['Sex'])

    # Scale numeric features
    if for_rf == False:
        scaler = StandardScaler()
        df[numeric_features] = scaler.fit_transform(df[numeric_features])

    # Capping fare using logarithmic transformation
    if for_rf == False:
        df['Fare'] = np.log1p(df['Fare'])

    # Feature engineering
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df.drop(['SibSp', 'Parch'], axis = 1, inplace = True)

    # Creating X (features) and y (target)
    X = df[features].values
    y = df['Survived'].values

    return X, y

def get_data_loaders(filepath, batch_size = 32, test_size = 0.2, random_state = 42, for_rf = False):
    """
    Load Data from CSV and return PyTorch DataLoaders
    """

    data = pd.read_csv(filepath)

    X, y = preprocess_data(data, for_rf)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = test_size, random_state = random_state
    )

    train_dataset = TitanicDataset(X_train, y_train)
    test_dataset = TitanicDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size)

    return train_loader, test_loader, X_train.shape[1]

if __name__ == "__main__":
    # Testing the preprocessing
    train_loader, test_loader, input_dim = get_data_loaders('dataset.csv')
    print("Input dimensions: ", input_dim)
    print("Number of batches in train loader: ", len(train_loader))
    print("Number of batches in test loader: ", len(test_loader))