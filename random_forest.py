import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


file_path = '/Users/maneloutajar/Documents/Internship/results_docking/results.xlsx'
df = pd.read_excel(file_path)
print("Aperçu du fichier du docking :")
print(df.head())

#Vérification et suppression des valeurs manquantes
print("\nVérification des valeurs manquantes :")
print(df.isnull().sum())

#Solution choisie en cas de valeurs manquantes: suppression de toutes les lignes contenant au moins une valeur manquante
df = df.dropna()

print("\nExcel après suppression des lignes comportants valeurs manquantes :")
print(df.head())

#On charge les protéines concernées
proteins_dir = '/Users/maneloutajar/Documents/Internship/proteins'
protein_files = [f for f in os.listdir(proteins_dir) if f.endswith('.pdb')]
known_proteins = [os.path.splitext(f)[0] for f in protein_files]

#Création de la colonne Experimental_Activity_Y/N (pour savoir si un médicament est jugé comme efficace ou non sur une protéine)
'''Les chiffres ont été choisis de sorte à avoir des positifs mais nous pouvons resserrer l'étau. '''
def assign_activity(row):
    if (
        row['affinity_score'] < -7
        and row['hydrogen_bonds'] >= 2
        and row['hydrophobic_contacts'] >= 3
        and row['Protein_Name'] in known_proteins
    ):
        return "Yes"
    else:
        return "No"

df['Experimental_Activity_Y/N'] = df.apply(assign_activity, axis=1)

print("\nAperçu après ajout de la colonne Experimental_Activity_Y/N :")
print(df[['Protein_Name','Ligand_Name','Experimental_Activity_Y/N']].head())

#Encodage des variables catégorielles, pour n'avoir que des nombres.
le_protein = LabelEncoder()
le_ligand = LabelEncoder()
df['Protein_Name'] = le_protein.fit_transform(df['Protein_Name'])
df['Ligand_Name'] = le_ligand.fit_transform(df['Ligand_Name'])

#Création du label d'activité de manière binaire 0/1
df['Activity_Label'] = df['Experimental_Activity_Y/N'].map({'Yes': 1, 'No': 0})

#Préparation des features 
features = [
    'affinity_score', 
    'hydrogen_bonds', 
    'hydrophobic_contacts', 
    'Protein_Name',
    'Ligand_Name',
    'residues',
]

X = df[features]
y = df['Activity_Label']

#Division train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Entraînement du modèle Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

#Prédictions et évaluation 
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nPrécision: {accuracy:.3f}")
print("\nRapport de classification:")
print(classification_report(y_test, y_pred))

#Importance des features
importance = pd.DataFrame({
    'feature': features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nImportance des features:")
print(importance)

#Graphique d'importance
plt.figure(figsize=(8, 5))
plt.barh(importance['feature'], importance['importance'])
plt.xlabel('Importance')
plt.title('Importance des Features')
plt.tight_layout()
plt.show()

#Sauvegarde dans un nouvel Excel pour ne pas écraser l'ancien
output_path = '/Users/maneloutajar/Documents/Internship/results_docking/results_with_activity.xlsx'
df.to_excel(output_path, index=False)
print(f"\nFichier sauvegardé : {output_path}")
