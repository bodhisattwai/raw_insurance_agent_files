
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

url = "https://raw.githubusercontent.com/bodhisattwai/insurance_featuredata/refs/heads/main/updated_customer_data.csv"
df = pd.read_csv(url)

!pip install scikit-learn==1.7.0

def convert_income(income_str):
    income_str = str(income_str).replace('$', '').replace('k', '000').replace('+', '')
    if '-' in income_str:
        low, high = map(int, income_str.split('-'))
        return (low + high) / 2
    else:
        return int(income_str)

df['Annual_Income_Value'] = df['Annual Income Range'].apply(convert_income)

df['Is_Smoker'] = df['Lifestyle Factors'].str.contains('Smoker', case=False, na=False).astype(int)
df['Has_Chronic_Illness'] = df['Lifestyle Factors'].str.contains('chronic illness', case=False, na=False).astype(int)
df['Travels_Frequently'] = df['Lifestyle Factors'].str.contains('travels frequently', case=False, na=False).astype(int)
df['Is_Active_Lifestyle'] = df['Lifestyle Factors'].str.contains('active lifestyle', case=False, na=False).astype(int)

df.drop(columns=['Persona Name', "Annual Income Range", "Lifestyle Factors"], inplace=True)

df_encoded = pd.get_dummies(df, drop_first=True)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_encoded)

df.info()

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd

def find_best_model_score(df, features_list, k):
    df_model = df[features_list].copy()
    df_encoded = pd.get_dummies(df_model, drop_first=True)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_encoded)
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    labels = kmeans.fit_predict(df_scaled)
    score = silhouette_score(df_scaled, labels)
    return score

features_1 = ['Age', 'Annual_Income_Value', 'Family Size', 'Key Financial Goals']
features_2 = ['Age', 'Annual_Income_Value', 'Family Size', 'Income Source']
features_3 = ['Age', 'Annual_Income_Value', 'Family Size', 'Key Financial Goals', 'Is_Smoker', 'Has_Chronic_Illness']

for k in [3, 4, 5, 6]:
    score1 = find_best_model_score(df, features_1, k)
    print(f"  - Score for 'Hybrid' (Goals): {score1:.3f}")
    score2 = find_best_model_score(df, features_2, k)
    print(f"  - Score for 'Profession': {score2:.3f}")
    score3 = find_best_model_score(df, features_3, k)
    print(f"  - Score for 'Lifestyle': {score3:.3f}")

winning_features = ['Age', 'Annual_Income_Value', 'Family Size', 'Key Financial Goals']
winning_k = 4

df_winner = df[winning_features].copy()
df_winner_encoded = pd.get_dummies(df_winner, drop_first=True)
scaler = StandardScaler()
df_winner_scaled = scaler.fit_transform(df_winner_encoded)

kmeans_winner = KMeans(n_clusters=winning_k, init='k-means++', random_state=42, n_init=10)
final_labels = kmeans_winner.fit_predict(df_winner_scaled)

df['Final_Segment'] = final_labels

final_winning_profile = df.groupby('Final_Segment').agg({
    'Age': 'mean',
    'Annual_Income_Value': 'mean',
    'Family Size': 'mean',
    'Income Source': lambda x: x.mode()[0],
    'Key Financial Goals': lambda x: x.mode()[0],
    'Relationship Status': lambda x: x.mode()[0]
})

final_winning_profile = final_winning_profile.round(0)
print(final_winning_profile)

with open('kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans_winner, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

from google.colab import files
files.download('kmeans_model.pkl')
files.download('scaler.pkl')
