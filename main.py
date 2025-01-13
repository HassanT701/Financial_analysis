import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler


# Load the dataset
file_path = '/Users/hassantalha/Desktop/VS-Code/Financial-Analysis/data/FinancialMarketData-EWS.csv'
data = pd.read_csv(file_path)

# Clean column names and convert 'Date' to datetime
data.columns = data.columns.str.replace(' ', '_')
data['Date'] = pd.to_datetime(data['Date'])

# Feature Engineering: Rolling statistics
rolling_window = 30
features_to_engineer = ['XAU_BGNL', 'BDIY', 'VIX', 'DXY', 'CRY']  # Key financial indicators
for feature in features_to_engineer:
    data[f'{feature}_rolling_mean_{rolling_window}'] = data[feature].rolling(window=rolling_window, min_periods=1).mean()
    data[f'{feature}_rolling_std_{rolling_window}'] = data[feature].rolling(window=rolling_window, min_periods=1).std()

# Add temporal features from the 'Date' column
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day_of_Week'] = data['Date'].dt.dayofweek

# Correlation Analysis: Find features most correlated with the target variable (Y)
correlation_matrix = data.corr()
correlation_with_target = correlation_matrix['Y'].sort_values(ascending=False)

# Visualize the top 10 features most correlated with Y
top_features = correlation_with_target.head(10).index
plt.figure(figsize=(10, 6))
sns.barplot(x=correlation_with_target[top_features], y=top_features, palette='coolwarm')
plt.title('Top Features Correlated with Target (Y)')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Features')
#plt.show()

# Train-Test Split: Use chronological order to split data
train_size = int(len(data) * 0.8)  # 80% training, 20% testing
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

# Output dataset shapes
print(f"Train Data Shape: {train_data.shape}")
print(f"Test Data Shape: {test_data.shape}")


# Separate features and target variable
X_train = train_data.drop(columns=['Date', 'Y'])  # Exclude Date and Target
y_train = train_data['Y']
X_test = test_data.drop(columns=['Date', 'Y'])
y_test = test_data['Y']

# Check for missing values
print("Missing values in train data:", X_train.isnull().sum().sum())
print("Missing values in test data:", X_test.isnull().sum().sum())

# Handle missing values using imputation
imputer = SimpleImputer(strategy='mean')  # Use mean imputation (can adjust to median or other strategies)
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compute class weights to handle imbalance
class_weights = compute_class_weight('balanced', classes=[0, 1], y=y_train)
class_weights_dict = {0: class_weights[0], 1: class_weights[1]}

# Initialize Random Forest Classifier
rf_model = RandomForestClassifier(
    n_estimators=100,       # Number of trees
    max_depth=10,           # Limit tree depth for regularization
    class_weight=class_weights_dict,  # Handle class imbalance
    random_state=42,        # For reproducibility
    n_jobs=-1               # Use all processors for parallel training
)

# Train the model
rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf_model.predict(X_test_scaled)
y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC-AUC Score: {roc_auc:.4f}")
