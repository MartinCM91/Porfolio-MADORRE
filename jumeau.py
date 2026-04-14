"""
Interface graphique de jumeau numérique thermique prédictif pour Raspberry Pi
---------------------------------------------------------------------------
 OBJECTIF: Créer un jumeau numérique qui prédit T2 (température résistance/processeur)
           à partir des paramètres mesurés (T1, T3, puissances, états)
           
IA PRÉDICTIVE: Utilise l'apprentissage automatique pour prédire T2 en temps réel
                 et suggérer les meilleurs paramètres pour atteindre une consigne

 FONCTIONNALITÉS:
- Contrôle PWM résistance chauffante et ventilateur
- Lecture 3 capteurs NTC via MCP3008 (T1, T2, T3)
- Jumeau numérique prédictif avec réseau de neurones
- Graphique temps réel: températures réelles vs prédictions IA
- Optimisation automatique des paramètres
- Sauvegarde continue des données pour apprentissage

🔧 MODIFICATIONS:
- Enregistrement données: toutes les 10s (au lieu de 0.3s)
- Sauvegarde sécurité: toutes les 10 minutes
- Ventilateur automatisation: 66-100% (au lieu de 10-100%)
"""

# =============== IMPORTS NÉCESSAIRES ===============
import tkinter as tk
from tkinter import ttk
import time
import threading
import csv
import datetime
import random
import matplotlib.pyplot as plt
import os
import math
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox
import glob

# =============== CONFIGURATION RASPBERRY PI ===============
try:
    import RPi.GPIO as GPIO
    import spidev
    RPI = True
    print(" Mode Raspberry Pi détecté")
except ImportError:
    RPI = False
    print(" Mode simulation PC")

if RPI:
    # Configuration broches GPIO
    VENT_IN1 = 16    # PWM ventilateur
    VENT_IN2 = 26    # Direction ventilateur (fixe LOW)
    HEAT_IN3 = 5     # PWM résistance chauffante
    HEAT_IN4 = 6     # Direction résistance (fixe LOW)

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(VENT_IN1, GPIO.OUT)
    GPIO.setup(VENT_IN2, GPIO.OUT)
    GPIO.setup(HEAT_IN3, GPIO.OUT)
    GPIO.setup(HEAT_IN4, GPIO.OUT)

    # Initialisation PWM
    pwm_vent = GPIO.PWM(VENT_IN1, 100)
    pwm_chauffe = GPIO.PWM(HEAT_IN3, 100)
    pwm_vent.start(0)
    pwm_chauffe.start(0)

    # Configuration SPI pour MCP3008
    spi = spidev.SpiDev()
    spi.open(0, 0)
    spi.max_speed_hz = 1350000

    def lire_adc(channel):
        """Lecture ADC via MCP3008 sur canal spécifié"""
        adc = spi.xfer2([1, (8 + channel) << 4, 0])
        data = ((adc[1] & 3) << 8) + adc[2]
        return data

    def adc_to_temperature(adc_val):
        """Conversion ADC → Température pour capteurs NTC"""
        tension = adc_val * (3.3 / 1023.0)
        temperature = 8.5 + (tension * 12.2) + (tension**2 * 0.15)
        return round(max(5, min(80, temperature)), 2)

# =============== JUMEAU NUMÉRIQUE PRÉDICTIF ===============
class JumeauNumeriquePredict:
    """
    JUMEAU NUMÉRIQUE PRÉDICTIF pour la température T2
    OBJECTIF: Prédire T2 (température résistance/processeur) en fonction de:
    - T1 (température entrée)
    - T3 (température sortie) 
    - Puissance chauffage (0-100%)
    - Puissance ventilateur (0-100%)
    - États ON/OFF des équipements
    ALGORITHMES: RandomForest, GradientBoosting, Réseau de neurones
     APPRENTISSAGE: Sur données historiques CSV
     PRÉDICTION: Temps réel avec comparaison vs capteur physique
    """
    def __init__(self):
        # Modèles et données d'entraînement
        self.modele_prediction = None
        self.scaler = StandardScaler()
        self.donnees_entrainement = pd.DataFrame()
        self.feature_names = []
        self.meilleur_modele_nom = ""
        
        # Configuration fichiers
        self.dossier_donnees = "."
        self.fichier_csv_specifique = None
        
        # Historiques pour comparaison temps réel
        self.historique_predictions = []  # T2 prédit par l'IA
        self.historique_reel = []         # T2 mesuré par capteur
        self.temps_predictions = []       # Timestamps
        
        # État et performance
        self.modele_actif = False
        self.precision_courante = 0.0
        self.erreur_moyenne = 0.0
        
        # Variables pour prédiction temporelle
        self.derniers_t1 = 25.0
        self.derniers_t3 = 25.0

    def detecter_format_csv(self, fichier_csv):
        """
         DÉTECTION AUTOMATIQUE DU FORMAT CSV
        Analyse la structure pour adapter le traitement
        """
        try:
            # Test avec différents délimiteurs
            for delimiter in [';', ',', '\t']:
                try:
                    df_test = pd.read_csv(fichier_csv, delimiter=delimiter, nrows=5)
                    if len(df_test.columns) >= 6:  # Au minimum 6 colonnes attendues
                        print(f" Délimiteur détecté: '{delimiter}'")
                        print(f" Colonnes: {list(df_test.columns)}")
                        return delimiter, df_test.columns.tolist()
                except:
                    continue
            
            return ';', []  # Défaut
            
        except Exception as e:
            print(f" Erreur détection format: {e}")
            return ';', []

    def charger_donnees_csv(self, fichier_specifique=None):
        """
         CHARGEMENT DONNÉES CSV avec gestion du format 
        """
        try:
            # Utiliser le fichier spécifique si fourni, sinon celui configuré
            fichier_a_charger = fichier_specifique or self.fichier_csv_specifique
            
            if not fichier_a_charger:
                print(" Aucun fichier CSV spécifié")
                return False
            
            print(f"\n === CHARGEMENT DONNÉES CSV ===")
            print(f" Fichier: {os.path.basename(fichier_a_charger)}")
            
            # Détection du format
            delimiter, colonnes = self.detecter_format_csv(fichier_a_charger)
            
            # Chargement complet du fichier
            df = pd.read_csv(fichier_a_charger, delimiter=delimiter)
            print(f" Lignes chargées: {len(df)}")
            
            # === GESTION DU FORMAT SPÉCIAL (toutes colonnes dans une seule) ===
            if len(df.columns) == 1 and ';' in str(df.iloc[0, 0]):
                print(" Format spécial détecté - Parsing manuel...")
                
                # Récupérer le nom de la première colonne qui contient tout
                col_name = df.columns[0]
                
                # Parser chaque ligne
                data_rows = []
                for idx, row in df.iterrows():
                    if idx == 0:  # Skip header row if it's repeated
                        continue
                    try:
                        # Séparer les valeurs
                        values = str(row[col_name]).split(';')
                        if len(values) >= 9:  # S'assurer qu'on a assez de colonnes
                            data_rows.append(values)
                    except:
                        continue
                
                # Créer le DataFrame avec les bonnes colonnes
                colonnes_attendues = ['name', 'date', 'time', 'Consigne', 'T1', 'T2', 'T3', 
                                     'Puissance Vent', 'Puissance Chauffe']
                df = pd.DataFrame(data_rows, columns=colonnes_attendues)
                print(f" Données parsées: {len(df)} lignes")
            
            # === MAPPING DES COLONNES ===
            # Vérifier si c'est le format de
            colonnes_c = ['t_rear', 't_middle', 't_front', 'pwm_fan', 'pwm_heat']
            format_c = all(col in df.columns for col in colonnes_c)
            
            if format_c:
                print(" Format  détecté - Mapping des colonnes...")
                df_mapped = pd.DataFrame()
                df_mapped['T1'] = pd.to_numeric(df['t_front'], errors='coerce')
                df_mapped['T2'] = pd.to_numeric(df['t_middle'], errors='coerce')
                df_mapped['T3'] = pd.to_numeric(df['t_rear'], errors='coerce')
                df_mapped['Puissance Vent'] = pd.to_numeric(df['pwm_fan'], errors='coerce')
                df_mapped['Puissance Chauffe'] = pd.to_numeric(df['pwm_heat'], errors='coerce')
                
                # Ajout des états ON/OFF basés sur les puissances
                df_mapped['Ventilateur ON'] = (df_mapped['Puissance Vent'] > 0).astype(int)
                df_mapped['Chauffage ON'] = (df_mapped['Puissance Chauffe'] > 0).astype(int)
                
                # Ajout d'une consigne par défaut si manquante
                df_mapped['Consigne'] = 30.0
                
                df = df_mapped
            else:
                # Format standard ou format avec parsing manuel
                print(" Conversion des colonnes numériques...")
                
                # Conversion des colonnes numériques
                colonnes_numeriques = ['T1', 'T2', 'T3', 'Puissance Vent', 'Puissance Chauffe', 'Consigne']
                for col in colonnes_numeriques:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Ajout des états ON/OFF si manquants
                if 'Ventilateur ON' not in df.columns:
                    df['Ventilateur ON'] = (df['Puissance Vent'] > 0).astype(int)
                if 'Chauffage ON' not in df.columns:
                    df['Chauffage ON'] = (df['Puissance Chauffe'] > 0).astype(int)
            
            # Ajout du temps si manquant
            if 'Temps' not in df.columns:
                df['Temps'] = range(len(df))
            
            # Nettoyage des données
            print(" Nettoyage des données...")
            df = df.dropna(subset=['T1', 'T2', 'T3', 'Puissance Vent', 'Puissance Chauffe'])
            
            if len(df) < 50:
                print(f" Pas assez de données après nettoyage: {len(df)} lignes")
                return False
            
            # Sauvegarde des données
            self.donnees_entrainement = df
            
            # Affichage statistiques
            print(f"\n === STATISTIQUES DONNÉES ===")
            print(f"Nombre total de points: {len(df)}")
            print(f"  T1: {df['T1'].min():.1f} - {df['T1'].max():.1f}°C (moy: {df['T1'].mean():.1f})")
            print(f" T2: {df['T2'].min():.1f} - {df['T2'].max():.1f}°C (moy: {df['T2'].mean():.1f})")
            print(f"  T3: {df['T3'].min():.1f} - {df['T3'].max():.1f}°C (moy: {df['T3'].mean():.1f})")
            print(f" Puissance Vent: {df['Puissance Vent'].min():.0f} - {df['Puissance Vent'].max():.0f}%")
            print(f" Puissance Chauffe: {df['Puissance Chauffe'].min():.0f} - {df['Puissance Chauffe'].max():.0f}%")
            
            return True
            
        except Exception as e:
            print(f" Erreur chargement CSV: {e}")
            import traceback
            traceback.print_exc()
            return False

    def valider_donnees_csv(self, df):
        """
         VALIDATION DES DONNÉES CSV
        Vérifie la cohérence et la qualité des données
        """
        try:
            print(" === VALIDATION DES DONNÉES ===")
            
            # Vérification des plages de valeurs
            problemes = []
            
            if df['T1'].min() < -10 or df['T1'].max() > 100:
                problemes.append(f"T1 hors plage réaliste: {df['T1'].min():.1f} - {df['T1'].max():.1f}°C")
            
            if df['T2'].min() < -10 or df['T2'].max() > 100:
                problemes.append(f"T2 hors plage réaliste: {df['T2'].min():.1f} - {df['T2'].max():.1f}°C")
            
            if df['T3'].min() < -10 or df['T3'].max() > 100:
                problemes.append(f"T3 hors plage réaliste: {df['T3'].min():.1f} - {df['T3'].max():.1f}°C")
            
            if df['Puissance Chauffe'].min() < 0 or df['Puissance Chauffe'].max() > 100:
                problemes.append(f"Puissance Chauffe hors plage: {df['Puissance Chauffe'].min():.1f} - {df['Puissance Chauffe'].max():.1f}%")
            
            if df['Puissance Vent'].min() < 0 or df['Puissance Vent'].max() > 100:
                problemes.append(f"Puissance Vent hors plage: {df['Puissance Vent'].min():.1f} - {df['Puissance Vent'].max():.1f}%")
            
            # Vérification cohérence états vs puissances
            incohérences_chauffe = ((df['Chauffage ON'] == 1) & (df['Puissance Chauffe'] == 0)).sum()
            incohérences_vent = ((df['Ventilateur ON'] == 1) & (df['Puissance Vent'] == 0)).sum()
            
            if incohérences_chauffe > 0:
                problemes.append(f"Incohérences chauffage: {incohérences_chauffe} lignes (ON mais puissance=0)")
            
            if incohérences_vent > 0:
                problemes.append(f"Incohérences ventilateur: {incohérences_vent} lignes (ON mais puissance=0)")
            
            # Affichage des problèmes
            if problemes:
                print(" PROBLÈMES DÉTECTÉS:")
                for prob in problemes:
                    print(f"   - {prob}")
            else:
                print(" Données validées avec succès")
            
            # Statistiques de qualité
            valeurs_manquantes = df.isnull().sum().sum()
            print(f" Valeurs manquantes: {valeurs_manquantes}")
            print(f" Taux de remplissage: {((1 - valeurs_manquantes/(len(df)*len(df.columns)))*100):.1f}%")
            
            return len(problemes) == 0
            
        except Exception as e:
            print(f" Erreur validation: {e}")
            return False
        
    def preparer_features_prediction(self, df):
        """
         PRÉPARATION DES FEATURES pour prédiction T2
        
        INPUT FEATURES:
        - T1, T3: Températures entrée/sortie
        - Puissance_Chauffe, Puissance_Vent: 0-100%
        - Chauffage_ON, Ventilateur_ON: États 0/1
        - Features avancées: gradients, ratios, historique
        
        OUTPUT TARGET: T2 (température à prédire)
        """
        try:
            # Features de base (mesures directes)
            features_base = [
                'T1', 'T3',                    # Températures entrée/sortie
                'Puissance Chauffe', 'Puissance Vent',  # Puissances 0-100%
                'Chauffage ON', 'Ventilateur ON'        # États binaires
            ]
            
            df_work = df.copy()
            
            #  FEATURES AVANCÉES (améliorent la prédiction)
            df_work['Gradient_T1_T3'] = df_work['T1'] - df_work['T3']  # Différence thermique
            df_work['Puissance_Totale'] = df_work['Puissance Chauffe'] + df_work['Puissance Vent']
            df_work['Ratio_Chauffe_Vent'] = np.where(
                df_work['Puissance Vent'] > 0,
                df_work['Puissance Chauffe'] / df_work['Puissance Vent'],
                df_work['Puissance Chauffe']
            )
            df_work['T_Moyenne_Entree_Sortie'] = (df_work['T1'] + df_work['T3']) / 2
            
            #  FEATURES TEMPORELLES (contexte historique)
            features_temporelles = []
            if len(df_work) > 10:
                df_work['T1_Precedent'] = df_work['T1'].shift(1)
                df_work['T3_Precedent'] = df_work['T3'].shift(1)
                df_work['Delta_T1'] = df_work['T1'].diff()
                df_work['Delta_T3'] = df_work['T3'].diff()
                features_temporelles = ['T1_Precedent', 'T3_Precedent', 'Delta_T1', 'Delta_T3']
            
            # Combiner toutes les features
            features_finales = features_base + [
                'Gradient_T1_T3', 'Puissance_Totale', 
                'Ratio_Chauffe_Vent', 'T_Moyenne_Entree_Sortie'
            ] + features_temporelles
            
            # Nettoyer NaN et valeurs manquantes
            df_work = df_work.dropna(subset=features_finales + ['T2'])
            
            if len(df_work) < 50:
                print(" Pas assez de données après nettoyage features")
                return None, None
            
            X = df_work[features_finales]  # Features d'entrée
            y = df_work['T2']              # Target (température à prédire)
            
            print(f" Features préparées: {len(X):,} échantillons, {len(features_finales)} variables")
            print(f" T2 cible - Min: {y.min():.1f}°C, Max: {y.max():.1f}°C, Moy: {y.mean():.1f}°C")
            
            self.feature_names = features_finales
            return X, y
            
        except Exception as e:
            print(f" Erreur préparation features: {e}")
            return None, None
    
    def entrainer_modele_comportemental(self):
        """
         ENTRAÎNEMENT DU JUMEAU NUMÉRIQUE
        
        PROCESSUS:
        1. Préparation des features à partir des données historiques
        2. Test de 3 algorithmes (RandomForest, GradientBoosting, Neural Network)
        3. Sélection du meilleur modèle (R² le plus élevé)
        4. Sauvegarde du modèle entraîné
        """
        try:
            if len(self.donnees_entrainement) < 100:
                print(f" Pas assez de données: {len(self.donnees_entrainement)} (minimum 100)")
                return False
            
            print(" === ENTRAÎNEMENT JUMEAU NUMÉRIQUE PRÉDICTIF ===")
            
            # Préparation des données
            X, y = self.preparer_features_prediction(self.donnees_entrainement)
            if X is None:
                return False
            
            # Division train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Normalisation pour les réseaux de neurones
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            #  COMPÉTITION DES ALGORITHMES
            modeles_candidats = {
                'RandomForest': RandomForestRegressor(
                    n_estimators=100, max_depth=15, random_state=42
                ),
                'GradientBoosting': GradientBoostingRegressor(
                    n_estimators=100, max_depth=8, learning_rate=0.1, random_state=42
                ),
                'NeuralNetwork': MLPRegressor(
                    hidden_layer_sizes=(100, 50, 25), max_iter=1000, 
                    random_state=42, early_stopping=True
                )
            }
            
            meilleur_modele = None
            meilleure_precision = -999
            
            print(" Test des algorithmes...")
            
            for nom, modele in modeles_candidats.items():
                try:
                    # Entraînement
                    if nom == 'NeuralNetwork':
                        modele.fit(X_train_scaled, y_train)
                        y_pred = modele.predict(X_test_scaled)
                    else:
                        modele.fit(X_train, y_train)
                        y_pred = modele.predict(X_test)
                    
                    # Évaluation
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    print(f" {nom:15s}: MAE={mae:.3f}°C, R²={r2:.3f}")
                    
                    # Sélection du meilleur (R² max)
                    if r2 > meilleure_precision:
                        meilleure_precision = r2
                        meilleur_modele = modele
                        self.erreur_moyenne = mae
                        self.precision_courante = r2
                        self.meilleur_modele_nom = nom
                        
                except Exception as e:
                    print(f" Erreur {nom}: {e}")
            
            if meilleur_modele is not None:
                self.modele_prediction = meilleur_modele
                self.modele_actif = True
                
                print(f"\n GAGNANT: {self.meilleur_modele_nom}")
                print(f" Précision (R²): {self.precision_courante:.3f}")
                print(f" Erreur moyenne: {self.erreur_moyenne:.3f}°C")
                
                # Sauvegarde des modèles
                try:
                    joblib.dump(self.modele_prediction, 'jumeau_numerique_t2.pkl')
                    joblib.dump(self.scaler, 'scaler_jumeau.pkl')
                    print(" Jumeau numérique sauvegardé")
                except Exception as e:
                    print(f" Erreur sauvegarde: {e}")
                
                return True
            
            print(" Aucun modèle n'a pu être entraîné correctement")
            return False
            
        except Exception as e:
            print(f" Erreur entraînement jumeau: {e}")
            return False
    
    def predire_t2_temps_reel(self, t1, t3, puissance_chauffe, puissance_vent, chauffage_on, ventilateur_on):
        """
         PRÉDICTION T2 EN TEMPS RÉEL
        
        INPUT: Paramètres actuels du système physique
        OUTPUT: T2 prédit par le jumeau numérique
        """
        try:
            if not self.modele_actif or self.modele_prediction is None:
                return None
            
            #  Préparation des features en temps réel
            features_base = [
                t1, t3,                               # Températures actuelles
                puissance_chauffe, puissance_vent,    # Puissances actuelles
                1 if chauffage_on else 0,             # États binaires
                1 if ventilateur_on else 0
            ]
            
            # Features avancées (mêmes calculs qu'à l'entraînement)
            gradient_t1_t3 = t1 - t3
            puissance_totale = puissance_chauffe + puissance_vent
            ratio_chauffe_vent = puissance_chauffe / puissance_vent if puissance_vent > 0 else puissance_chauffe
            t_moyenne = (t1 + t3) / 2
            
            features_avancees = [gradient_t1_t3, puissance_totale, ratio_chauffe_vent, t_moyenne]
            
            # Features temporelles (utilise historique si disponible)
            features_temporelles = []
            if hasattr(self, 'feature_names') and 'T1_Precedent' in self.feature_names:
                features_temporelles = [
                    self.derniers_t1, self.derniers_t3,        # Valeurs précédentes
                    t1 - self.derniers_t1, t3 - self.derniers_t3  # Variations
                ]
            
            # Assemblage final
            if features_temporelles:
                features_completes = features_base + features_avancees + features_temporelles
            else:
                features_completes = features_base + features_avancees
            
            #  PRÉDICTION
            X_pred = np.array([features_completes])
            
            if self.meilleur_modele_nom == 'NeuralNetwork':
                X_pred_scaled = self.scaler.transform(X_pred)
                t2_predit = self.modele_prediction.predict(X_pred_scaled)[0]
            else:
                t2_predit = self.modele_prediction.predict(X_pred)[0]
            
            # Contraintes physiques réalistes
            t2_predit = max(10, min(100, t2_predit))
            
            # Mémorisation pour prochaine prédiction
            self.derniers_t1 = t1
            self.derniers_t3 = t3
            
            return round(t2_predit, 2)
            
        except Exception as e:
            print(f" Erreur prédiction temps réel: {e}")
            return None
    
    def ajouter_mesure_temps_reel(self, temps, t2_reel, t2_predit=None):
        """ Ajoute une mesure pour comparaison prédiction vs réalité"""
        self.historique_reel.append(t2_reel)
        self.temps_predictions.append(temps)
        
        if t2_predit is not None:
            self.historique_predictions.append(t2_predit)
        
        # Limitation mémoire (30 minutes max)
        max_historique = 180  # 30 min à 10s = 180 points
        if len(self.historique_reel) > max_historique:
            self.historique_reel = self.historique_reel[-max_historique:]
            self.temps_predictions = self.temps_predictions[-max_historique:]
            if self.historique_predictions:
                self.historique_predictions = self.historique_predictions[-max_historique:]
    
    def obtenir_donnees_comparaison(self):
        """ Retourne les données pour affichage graphique"""
        return (
            self.temps_predictions.copy(),
            self.historique_reel.copy(), 
            self.historique_predictions.copy()
        )
    
    def suggerer_parametres_optimaux(self, temperature_cible, t1_actuel, t3_actuel):
        """
         OPTIMISATION AUTOMATIQUE
        Utilise le jumeau numérique pour trouver les meilleurs paramètres
        pour atteindre une température T2 cible
        """
        if not self.modele_actif:
            return None
        
        meilleure_config = None
        meilleur_ecart = float('inf')
        
        #  Test systématique des combinaisons (avec ventilateur optimisé 66-100%)
        for chauffage_on in [0, 1]:
            for ventilateur_on in [0, 1]:
                for puissance_chauffe in [0, 25, 50, 75, 100]:
                    #  MODIFICATION: Ventilateur 66-100% au lieu de 0-100%
                    puissances_vent = [0] if ventilateur_on == 0 else [66, 75, 85, 100]
                    for puissance_vent in puissances_vent:
                        # Skip configurations impossibles
                        if chauffage_on == 0 and puissance_chauffe > 0:
                            continue
                        if ventilateur_on == 0 and puissance_vent > 0:
                            continue
                        
                        # Prédiction avec cette configuration
                        t2_predit = self.predire_t2_temps_reel(
                            t1_actuel, t3_actuel, puissance_chauffe, puissance_vent,
                            chauffage_on == 1, ventilateur_on == 1
                        )
                        
                        if t2_predit is not None:
                            ecart = abs(t2_predit - temperature_cible)
                            
                            if ecart < meilleur_ecart:
                                meilleur_ecart = ecart
                                meilleure_config = {
                                    'puissance_chauffe': puissance_chauffe,
                                    'puissance_vent': puissance_vent,
                                    'chauffage_on': chauffage_on == 1,
                                    'ventilateur_on': ventilateur_on == 1,
                                    'temp_predite': t2_predit,
                                    'ecart': ecart
                                }
        
        return meilleure_config
    
    def definir_dossier_donnees(self, dossier):
        """ Définit le dossier de recherche des données CSV"""
        self.dossier_donnees = dossier
        print(f" Dossier données: {dossier}")
    
    def reinitialiser_historique(self):
        """ Remet à zéro l'historique de comparaison"""
        self.historique_predictions = []
        self.historique_reel = []
        self.temps_predictions = []
        print(" Historique jumeau numérique réinitialisé")

# =============== INTERFACE GRAPHIQUE PRINCIPALE ===============
class ControleThermique:
    """
    INTERFACE GRAPHIQUE PRINCIPALE
    Intègre le jumeau numérique prédictif avec contrôle hardware
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title(" Jumeau Numérique Thermique Prédictif")
        self.root.geometry("1000x600")

        # Style interface
        style = ttk.Style()
        style.configure("TButton", font=("Arial", 10, "bold"))
        style.configure("TFrame", background="#f0f0f0")
        style.configure("TLabel", font=("Arial", 10))

        # Interface scrollable
        self.setup_scrollable_interface()
        
        # Variables de contrôle
        self.puissance_vent = tk.DoubleVar(value=0.0)
        self.puissance_chauffe = tk.DoubleVar(value=0.0)
        self.texte_puissance_vent = tk.StringVar(value="0 %")
        self.texte_puissance_chauffe = tk.StringVar(value="0 %")
        self.consigne = tk.DoubleVar(value=25.0)

        # États équipements
        self.ventilateur_etat = False
        self.chauffage_etat = False
        self.regulation = False
        
        # Variables temporelles et données
        self.debut_acquisition = time.time()
        self.tps = []
        self.temp_capteurs = [[], [], []]
        self.historique_etats = []
        
        # GESTION TEMPORELLE
        self.derniere_mesure = 0  # Dernière mesure enregistrée
        self.derniere_sauvegarde = 0  # Dernière sauvegarde de sécurité
        
        # Jumeau numérique
        self.jumeau_numerique = JumeauNumeriquePredict()
        self.ia_active = False
        self.suggestions_ia = None
        
        self.telemetry_data = {}
        self.stats_gaming = {}
        self.temp_labels_mobile = {}
        self.radar_points = {}
        self.mission_start_time = time.time()


        # Automatisation
        self.automatisation_active = False
        self.prochaine_automatisation = time.time() + 300  # 5 min
        
        # Sauvegarde
        self.fichier_session_actuel = None
        self.points_depuis_derniere_sauvegarde = 0
        
        # Création interface et démarrage
        self.create_widgets()
        self.creer_fichier_session()
        self.charger_config_ia()
        self.demarrer_acquisition()
        self.activer_toutes_interfaces()
    
        self.demarrer_acquisition()

    def setup_scrollable_interface(self):
        """ Configuration interface scrollable"""
        container = ttk.Frame(self.root)
        container.pack(fill='both', expand=True)

        canvas = tk.Canvas(container)
        scrollbar = ttk.Scrollbar(container, orient='vertical', command=canvas.yview)

        self.scrollable = ttk.Frame(canvas)
        self.scrollable.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Support molette souris
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

    def create_widgets(self):
        """ Création de tous les widgets de l'interface"""
        
        # === CRÉATION DU NOTEBOOK PRINCIPAL ===
        self.notebook = ttk.Notebook(self.scrollable)
        self.notebook.pack(fill="both", expand=True, pady=10)
        
        # === ONGLET 1: CONTRÔLE SYSTÈME ===
        self.onglet_controle = ttk.Frame(self.notebook)
        self.notebook.add(self.onglet_controle, text=" Contrôle Système")
        
        # === CONTRÔLES PUISSANCE ===
        f = ttk.Frame(self.onglet_controle)
        f.pack(pady=10)

        # Résistance chauffante
        ttk.Label(f, text="Résistance chauffante").grid(row=0, column=0)
        ttk.Checkbutton(f, text="ON/OFF", command=self.toggle_chauffage).grid(row=1, column=0)
        ttk.Scale(f, from_=0, to=100, variable=self.puissance_chauffe, 
                 orient='horizontal', command=self.update_pwm_chauffe).grid(row=2, column=0)
        ttk.Label(f, textvariable=self.texte_puissance_chauffe).grid(row=3, column=0)

        # Ventilateur
        ttk.Label(f, text="Ventilateur").grid(row=0, column=1)
        ttk.Checkbutton(f, text="ON/OFF", command=self.toggle_ventilateur).grid(row=1, column=1)
        ttk.Scale(f, from_=0, to=100, variable=self.puissance_vent, 
                 orient='horizontal', command=self.update_pwm_vent).grid(row=2, column=1)
        ttk.Label(f, textvariable=self.texte_puissance_vent).grid(row=3, column=1)

        # === AFFICHAGE TEMPÉRATURES ===
        temp_frame = ttk.Frame(self.onglet_controle)
        temp_frame.pack(pady=10)
        
        self.capteur1_label = ttk.Label(temp_frame, text=" T1 Entrée Ventillo : -- °C", 
                                       font=("Arial", 12, "bold"))
        self.capteur2_label = ttk.Label(temp_frame, text=" T2 Résistance : -- °C", 
                                       font=("Arial", 12, "bold"))
        self.capteur3_label = ttk.Label(temp_frame, text=" T3 Sortie : -- °C", 
                                       font=("Arial", 12, "bold"))
        self.capteur1_label.pack(pady=2)
        self.capteur2_label.pack(pady=2)
        self.capteur3_label.pack(pady=2)

        # === STATUS SYSTÈME ===
        status_frame = ttk.Frame(self.onglet_controle)
        status_frame.pack(pady=5)

        self.status_ventil_label = ttk.Label(status_frame, text="Ventilateur : OFF", 
                                           font=("Arial", 12), foreground='red')
        self.status_chauffage_label = ttk.Label(status_frame, text="Résistance : OFF", 
                                              font=("Arial", 12), foreground='red')
        self.status_regulation_label = ttk.Label(status_frame, text="Régulation : OFF", 
                                               font=("Arial", 12), foreground='red')
        self.status_sauvegarde_label = ttk.Label(status_frame, text="Sauvegarde : En attente", 
                                               font=("Arial", 12))
        
        self.status_ventil_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.status_chauffage_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.status_regulation_label.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        self.status_sauvegarde_label.grid(row=1, column=1, padx=10, pady=5, sticky="w")

        # ===  ANIMATIONS VENTILATEUR ET RÉSISTANCE ===
        self.creer_animations()

        # === GRAPHIQUE TEMPS RÉEL ===
        self.creer_graphique_controle()

        # === RÉGULATION AUTOMATIQUE ===
        control_frame = ttk.Frame(self.onglet_controle)
        control_frame.pack(pady=10)

        ttk.Label(control_frame, text="Température cible (°C):").grid(row=0, column=0, padx=5)
        ttk.Entry(control_frame, textvariable=self.consigne, width=5).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Activer régulation", 
                  command=self.toggle_regulation).grid(row=0, column=2, padx=10)

        # === AUTOMATISATION ===
        auto_frame = ttk.Frame(self.onglet_controle)
        auto_frame.pack(pady=10)

        ttk.Button(auto_frame, text=" Automatisation Jumeau Numérique", 
                  command=self.toggle_automatisation).grid(row=0, column=0, padx=10)
        self.status_auto_label = ttk.Label(auto_frame, text="Automatisation : OFF", 
                                         font=("Arial", 12), foreground='red')
        self.status_auto_label.grid(row=0, column=1, padx=20)

        self.timer_auto_label = ttk.Label(auto_frame, text="Prochaine action : --", font=("Arial", 10))
        self.timer_auto_label.grid(row=1, column=0, columnspan=2, pady=5)

        # === JUMEAU NUMÉRIQUE ===
        ia_frame = ttk.LabelFrame(self.onglet_controle, text="🤖 Jumeau Numérique Prédictif")
        ia_frame.pack(pady=10, fill="x")

        # Sélection fichiers
        ttk.Button(ia_frame, text="Choisir dossier données", 
                  command=self.choisir_dossier_donnees).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(ia_frame, text=" Choisir fichier CSV", 
                  command=self.choisir_fichier_csv).grid(row=0, column=1, padx=5, pady=5)
        
        self.dossier_label = ttk.Label(ia_frame, text=" Dossier: ./", font=("Arial", 9))
        self.dossier_label.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        # Actions IA
        ttk.Button(ia_frame, text=" Charger & Entraîner IA", 
                  command=self.charger_et_entrainer_jumeau).grid(row=2, column=0, padx=10, pady=5)
        ttk.Button(ia_frame, text=" Appliquer suggestions IA", 
                  command=self.appliquer_suggestions_ia).grid(row=2, column=1, padx=10, pady=5)
        ttk.Button(ia_frame, text=" Réinitialiser historique", 
                  command=self.reinitialiser_jumeau).grid(row=2, column=2, padx=10, pady=5)
        
        # Status IA
        self.status_ia_label = ttk.Label(ia_frame, text=" Jumeau : Non chargé", font=("Arial", 10))
        self.status_ia_label.grid(row=3, column=0, columnspan=3, pady=5)
        
        self.prediction_label = ttk.Label(ia_frame, text=" Prédiction T2 : --", font=("Arial", 10))
        self.prediction_label.grid(row=4, column=0, columnspan=3, pady=2)
        
        self.suggestion_label = ttk.Label(ia_frame, text=" Suggestion : --", font=("Arial", 10))
        self.suggestion_label.grid(row=5, column=0, columnspan=3, pady=2)

        # === BOUTONS PRINCIPAUX ===
        button_frame = ttk.Frame(self.onglet_controle)
        button_frame.pack(pady=15)
        
        ttk.Button(button_frame, text=" Sauvegarder données", 
                  command=self.sauvegarder).grid(row=0, column=0, padx=10)
        ttk.Button(button_frame, text=" Quitter", 
                  command=self.stop).grid(row=0, column=1, padx=10)
        
        # === ONGLET 2: ANALYSE COMPARATIVE ===
        self.create_analyse_tab()

    def creer_graphique_controle(self):
        """ Création du graphique de contrôle temps réel (onglet 1)"""
        graph_frame = ttk.LabelFrame(self.onglet_controle, text=" Graphique: Températures Réelles vs Prédictions IA")
        graph_frame.pack(pady=10, fill="both", expand=True)
        
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        
        # Couleurs distinctes pour chaque courbe
        colors = ['#e74c3c', '#3498db', '#27ae60', '#9b59b6']  # Rouge, Bleu, Vert, Violet
        markers = ['o', 's', '^', 'D']

        # Courbes des 3 capteurs physiques
        self.lines = []
        labels = [" T1 Entrée Ventillo", " T2 Résistance", " T3 Sortie"]
        for i in range(3):
            line = self.ax.plot([], [], 
                            label=labels[i],
                            color=colors[i],
                            linewidth=2.5,
                            marker=markers[i],
                            markersize=3,
                            markevery=10)[0]
            self.lines.append(line)

        # Courbe prédiction jumeau numérique T2
        self.line_prediction_ia = self.ax.plot([], [], 
                                            label=" T2 Prédit (Jumeau Numérique)", 
                                            color=colors[3],
                                            linestyle='--',
                                            linewidth=3,
                                            alpha=0.8)[0]

        # Configuration graphique
        self.ax.set_ylim(15, 50)
        self.ax.set_xlim(0, 60)
        self.ax.set_title(" Comparaison Capteur Physique vs Jumeau Numérique", fontweight='bold')
        self.ax.set_xlabel("Temps (s)")
        self.ax.set_ylabel("Température (°C)")
        self.ax.legend(loc='upper left')
        self.ax.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        self.fig.subplots_adjust(bottom=0.15, left=0.15)

        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(pady=5, fill="both", expand=True)

    def creer_animations(self):
        """ Création des animations ventilateur et résistance chauffante"""
        
        # Frame pour les animations
        animations_frame = ttk.Frame(self.onglet_controle)
        animations_frame.pack(pady=10)
        
        #  ANIMATION VENTILATEUR (à gauche)
        fan_frame = ttk.Frame(animations_frame)
        fan_frame.grid(row=0, column=0, padx=15)
        self.fan_canvas = tk.Canvas(fan_frame, width=120, height=120, bg='white', 
                                   highlightthickness=1, highlightbackground="gray")
        self.fan_canvas.pack(pady=5)
        ttk.Label(fan_frame, text="Ventilateur", font=("Arial", 10, "bold")).pack()
        
        #  ANIMATION RÉSISTANCE CHAUFFANTE (à droite)
        heater_frame = ttk.Frame(animations_frame)
        heater_frame.grid(row=0, column=1, padx=15)
        self.heater_canvas = tk.Canvas(heater_frame, width=200, height=120, bg='white', 
                                      highlightthickness=1, highlightbackground="gray")
        self.heater_canvas.pack(pady=5)
        ttk.Label(heater_frame, text="Résistance Chauffante", font=("Arial", 10, "bold")).pack()
        
        # Initialisation animations
        self.fan_rotation_angle = 0
        self.setup_fan_animation()
        self.setup_heater_animation()

    def setup_fan_animation(self):
        """ Configuration animation ventilateur"""
        try:
            # Essayer de charger l'image du ventilateur
            script_dir = os.path.dirname(os.path.abspath(__file__))
            img_path = os.path.join(script_dir, "ventillo.png")
            
            if os.path.exists(img_path):
                self.fan_img_orig = Image.open(img_path).resize((80, 80))
                self.fan_img = ImageTk.PhotoImage(self.fan_img_orig)
                self.fan_item = self.fan_canvas.create_image(60, 60, image=self.fan_img)
                print(f" Image ventilateur chargée: {img_path}")
                self.has_fan_image = True
            else:
                print(f" Image ventilateur introuvable: {img_path}")
                self.create_default_fan()
                self.has_fan_image = False
        except Exception as e:
            print(f" Erreur chargement image ventilateur: {e}")
            self.create_default_fan()
            self.has_fan_image = False

    def create_default_fan(self):
        """ Création ventilateur par défaut (dessin)"""
        # Corps du ventilateur
        self.fan_canvas.create_oval(20, 20, 100, 100, outline='black', width=2, 
                                   fill='lightgray', tags="fan_body")
        self.fan_canvas.create_oval(55, 55, 65, 65, outline='black', width=2, 
                                   fill='darkgray', tags="fan_center")
        # Pales (créées dynamiquement)
        self.draw_fan_blades()

    def draw_fan_blades(self):
        """ Dessin des pales du ventilateur"""
        self.fan_canvas.delete("fan_blade")
        center_x, center_y = 60, 60
        radius = 35
        
        for i in range(4):  # 4 pales
            angle = math.radians(i * 90 + self.fan_rotation_angle)
            
            # Créer une pale avec arc
            blade_width = 12
            angle1 = math.degrees(angle) - blade_width
            angle2 = math.degrees(angle) + blade_width
            
            self.fan_canvas.create_arc(
                center_x - radius, center_y - radius,
                center_x + radius, center_y + radius,
                start=angle1, extent=blade_width*2,
                style=tk.PIESLICE, fill='lightblue',
                outline='blue', width=1, tags="fan_blade"
            )

    def setup_heater_animation(self):
        """Configuration animation résistance chauffante"""
        try:
            # Essayer de charger les images de résistance
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.radiator_img_path = os.path.join(script_dir, "radiator.png")
            self.radiator_hot_img_path = os.path.join(script_dir, "radiator_hot.png")
            
            if os.path.exists(self.radiator_img_path) and os.path.exists(self.radiator_hot_img_path):
                self.radiator_img = ImageTk.PhotoImage(Image.open(self.radiator_img_path).resize((80, 80)))
                self.radiator_hot_img = ImageTk.PhotoImage(Image.open(self.radiator_hot_img_path).resize((80, 80)))
                self.radiator_image_id = self.heater_canvas.create_image(100, 60, image=self.radiator_img)
                print(" Images radiateur chargées avec succès")
            else:
                print(" Images radiateur introuvables - Création graphique par défaut")
                self.create_default_heater()
        except Exception as e:
            print(f" Erreur chargement images résistance: {e}")
            self.create_default_heater()

    def create_default_heater(self):
        """ Création résistance par défaut (dessin)"""
        # Corps du radiateur
        self.heater_canvas.create_rectangle(50, 20, 150, 100, outline='black', width=2, 
                                           fill='lightgray', tags="heater_body")
        
        # Éléments chauffants
        self.heater_elements = []
        for i in range(6):
            x = 60 + i * 15
            rect = self.heater_canvas.create_rectangle(x, 30, x+10, 90, outline='gray', 
                                                      width=1, fill='lightgray', tags="heater_element")
            self.heater_elements.append(("rect", rect))
        
        # Fil chauffant
        wire = self.heater_canvas.create_line(50, 110, 150, 110, fill='black', width=2, tags="heater_wire")
        self.heater_elements.append(("line", wire))
        
        self.radiator_image_id = None

    def rotate_fan(self):
        """ Animation rotation ventilateur - VERSION CORRIGÉE"""
        if self.ventilateur_etat:
            if getattr(self, 'has_fan_image', False):
                try:
                    # Rotation de l'image du ventilateur (méthode qui fonctionne)
                    angle = 15  # Angle de rotation par étape
                    self.fan_img_orig = self.fan_img_orig.rotate(angle)
                    self.fan_img = ImageTk.PhotoImage(self.fan_img_orig)
                    self.fan_canvas.itemconfig(self.fan_item, image=self.fan_img)
                except Exception as e:
                    print(f" Erreur rotation image: {e}")
                    # En cas d'erreur, passer au mode dessin
                    if not hasattr(self, 'fan_rotation_angle'):
                        self.fan_rotation_angle = 0
                        self.create_default_fan()
                        self.has_fan_image = False
            else:
                # Animation des pales dessinées
                self.fan_rotation_angle = (self.fan_rotation_angle + 15) % 360
                self.draw_fan_blades()

    def update_heater_animation(self):
        """ Animation résistance chauffante"""
        if hasattr(self, 'radiator_image_id') and self.radiator_image_id is not None:
            # Mettre à jour l'image selon l'état
            if self.chauffage_etat:
                self.heater_canvas.itemconfig(self.radiator_image_id, image=self.radiator_hot_img)
            else:
                self.heater_canvas.itemconfig(self.radiator_image_id, image=self.radiator_img)
        
        # Animation des éléments dessinés avec effet de lueur
        if hasattr(self, 'heater_elements'):
            if self.chauffage_etat:
                # Intensité selon puissance + effet pulsation
                intensity = 0.2 + (self.puissance_chauffe.get() / 100.0) * 0.7
                pulse = 0.1 * math.sin(time.time() * 3)
                self.update_heater_glow(min(1.0, intensity + pulse))
            else:
                self.update_heater_glow(0.2)  # Faible lueur quand éteint

    def update_heater_glow(self, opacity):
        """ Mise à jour effet de lueur résistance"""
        if hasattr(self, 'heater_elements'):
            for typ, element in self.heater_elements:
                # Couleur rouge-orange selon intensité
                r = int(255)
                g = int(68 + 120 * opacity)
                b = int(opacity * 50)
                color = f'#{r:02x}{g:02x}{b:02x}'

                if typ == "rect":
                    self.heater_canvas.itemconfig(element, fill=color, outline=color)
                elif typ == "line":
                    self.heater_canvas.itemconfig(element, fill=color)

    def create_analyse_tab(self):
        """ Création onglet d'analyse comparative détaillée"""
        
        # Onglet analyse comparative
        self.onglet_analyse = ttk.Frame(self.notebook)
        self.notebook.add(self.onglet_analyse, text="📈 Analyse Comparative")
        
        # === SECTION CHARGEMENT FICHIER ===
        frame_fichier = ttk.LabelFrame(self.onglet_analyse, text=" Chargement Données")
        frame_fichier.pack(fill="x", padx=10, pady=5)
        
        ttk.Button(frame_fichier, text=" Charger fichier CSV", 
                command=self.charger_fichier_analyse).pack(side="left", padx=5, pady=5)
        
        self.fichier_analyse_label = ttk.Label(frame_fichier, text="Aucun fichier chargé")
        self.fichier_analyse_label.pack(side="left", padx=10, pady=5)
        
        # === SECTION STATISTIQUES ===
        frame_stats = ttk.LabelFrame(self.onglet_analyse, text=" Statistiques de Performance")
        frame_stats.pack(fill="x", padx=10, pady=5)
        
        # Métriques de base
        stats_frame = ttk.Frame(frame_stats)
        stats_frame.pack(fill="x", padx=5, pady=5)
        
        self.mae_label = ttk.Label(stats_frame, text="MAE : --", font=("Arial", 11, "bold"))
        self.rmse_label = ttk.Label(stats_frame, text="RMSE : --", font=("Arial", 11, "bold"))
        self.r2_label = ttk.Label(stats_frame, text="R² : --", font=("Arial", 11, "bold"))
        self.mape_label = ttk.Label(stats_frame, text="MAPE : --", font=("Arial", 11, "bold"))
        
        self.mae_label.grid(row=0, column=0, padx=20, pady=5, sticky="w")
        self.rmse_label.grid(row=0, column=1, padx=20, pady=5, sticky="w")
        self.r2_label.grid(row=1, column=0, padx=20, pady=5, sticky="w")
        self.mape_label.grid(row=1, column=1, padx=20, pady=5, sticky="w")
        
        # Boutons d'analyse
        boutons_frame = ttk.Frame(frame_stats)
        boutons_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(boutons_frame, text=" Actualiser analyse", 
                command=self.actualiser_analyse).pack(side="left", padx=5)
        ttk.Button(boutons_frame, text=" Exporter analyse", 
                command=self.exporter_analyse).pack(side="left", padx=5)
        ttk.Button(boutons_frame, text=" Analyse détaillée", 
                command=self.analyse_detaillee).pack(side="left", padx=5)
        
        # === GRAPHIQUE PRINCIPAL ===
        self.creer_graphique_analyse()
        
        # === SECTION ANALYSE TEMPORELLE ===
        frame_temporelle = ttk.LabelFrame(self.onglet_analyse, text=" Analyse Temporelle")
        frame_temporelle.pack(fill="x", padx=10, pady=5)
        
        # Sélection période d'analyse
        periode_frame = ttk.Frame(frame_temporelle)
        periode_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(periode_frame, text="Période d'analyse:").pack(side="left", padx=5)
        
        self.periode_var = tk.StringVar(value="Complète")
        periode_combo = ttk.Combobox(periode_frame, textvariable=self.periode_var, 
                                    values=["Complète", "Dernière heure", "Dernières 30 min", 
                                        "Dernières 10 min", "Personnalisée"])
        periode_combo.pack(side="left", padx=5)
        periode_combo.bind("<<ComboboxSelected>>", self.changer_periode_analyse)
        
        # Affichage période active
        self.periode_info_label = ttk.Label(periode_frame, text="", font=("Arial", 9))
        self.periode_info_label.pack(side="left", padx=20)

    def creer_interface_nasa_style(self):
        """Interface style NASA avec monitoring avancé"""
        try:
            # Onglet Mission Control
            self.onglet_mission = ttk.Frame(self.notebook)
            self.notebook.add(self.onglet_mission, text=" Mission Control")
            
            # === ÉCRAN PRINCIPAL STYLE NASA ===
            main_frame = tk.Frame(self.onglet_mission, bg='#0a0a0a')
            main_frame.pack(fill="both", expand=True)
            
            # Header avec horloge mission
            header = tk.Frame(main_frame, bg='#1a1a1a', height=60)
            header.pack(fill="x", padx=5, pady=5)
            header.pack_propagate(False)
            
            # Mission Timer
            self.mission_time_label = tk.Label(header, text="MISSION TIME: 00:00:00", 
                                              font=("Courier", 16, "bold"), 
                                              fg='#00ff00', bg='#1a1a1a')
            self.mission_time_label.pack(side="left", padx=20, pady=15)
            
            # Status général
            self.mission_status = tk.Label(header, text="● NOMINAL", 
                                          font=("Courier", 14, "bold"), 
                                          fg='#00ff00', bg='#1a1a1a')
            self.mission_status.pack(side="right", padx=20, pady=15)
            
            # === PANNEAUX DE CONTRÔLE ===
            panels_frame = tk.Frame(main_frame, bg='#0a0a0a')
            panels_frame.pack(fill="both", expand=True, padx=5, pady=5)
            
            # Configuration grid
            panels_frame.grid_columnconfigure(0, weight=1)
            panels_frame.grid_columnconfigure(1, weight=2)
            panels_frame.grid_columnconfigure(2, weight=1)
            panels_frame.grid_rowconfigure(0, weight=1)
            
            # Panel gauche - Télémétrie
            self.creer_panel_telemetrie(panels_frame)
            
            # Panel centre - Graphique radar
            self.creer_radar_thermique(panels_frame)
            
            # Panel droit - Commandes
            self.creer_panel_commandes_nasa(panels_frame)
            
            print(" Interface NASA créée")
            
        except Exception as e:
            print(f" Erreur interface NASA: {e}")

    def creer_panel_telemetrie(self, parent):
        """ Panel télémétrie style NASA"""
        
        telemetry_frame = tk.Frame(parent, bg='#1a1a1a', relief="ridge", bd=2)
        telemetry_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Titre
        tk.Label(telemetry_frame, text="TELEMETRY DATA", 
                font=("Courier", 12, "bold"), fg='#00ffff', bg='#1a1a1a').pack(pady=10)
        
        # Données en temps réel
        data_points = [
            ("T1_INLET", "°C", "#ff4444"),
            ("T2_CORE", "°C", "#44ff44"), 
            ("T3_OUTLET", "°C", "#4444ff"),
            ("PWR_HEAT", "%", "#ffaa44"),
            ("PWR_FAN", "%", "#aa44ff"),
            ("AI_PREDICT", "°C", "#ffff44")
        ]
        
        for name, unit, color in data_points:
            frame = tk.Frame(telemetry_frame, bg='#1a1a1a')
            frame.pack(fill="x", padx=10, pady=2)
            
            tk.Label(frame, text=f"{name}:", font=("Courier", 10), 
                    fg='#cccccc', bg='#1a1a1a', width=12, anchor="w").pack(side="left")
            
            value_label = tk.Label(frame, text="---.--", font=("Courier", 10, "bold"), 
                                  fg=color, bg='#1a1a1a', width=8, anchor="e")
            value_label.pack(side="left")
            
            tk.Label(frame, text=unit, font=("Courier", 10), 
                    fg='#cccccc', bg='#1a1a1a', width=3, anchor="w").pack(side="left")
            
            self.telemetry_data[name] = value_label

    def creer_radar_thermique(self, parent):
        """ Radar thermique circulaire"""
        
        radar_frame = tk.Frame(parent, bg='#1a1a1a', relief="ridge", bd=2)
        radar_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        tk.Label(radar_frame, text="THERMAL RADAR", 
                font=("Courier", 12, "bold"), fg='#00ffff', bg='#1a1a1a').pack(pady=10)
        
        # Canvas radar
        self.radar_canvas = tk.Canvas(radar_frame, width=300, height=300, 
                                     bg='#0a0a0a', highlightthickness=0)
        self.radar_canvas.pack(padx=10, pady=10)
        
        # Dessiner grille radar
        self.dessiner_grille_radar()

    def creer_panel_commandes_nasa(self, parent):
        """ Panel commandes NASA"""
        
        cmd_frame = tk.Frame(parent, bg='#1a1a1a', relief="ridge", bd=2)
        cmd_frame.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        
        tk.Label(cmd_frame, text="COMMAND PANEL", 
                font=("Courier", 12, "bold"), fg='#00ffff', bg='#1a1a1a').pack(pady=10)
        
        # Boutons style NASA
        boutons = [
            (" IGNITION", self.toggle_chauffage, "#ff4444"),
            (" COOLING", self.toggle_ventilateur, "#4444ff"),
            (" AI CTRL", self.toggle_automatisation, "#44ff44"),
            (" ABORT", self.emergency_stop, "#ff0000")
        ]
        
        for text, command, color in boutons:
            btn = tk.Button(cmd_frame, text=text, command=command,
                           font=("Courier", 10, "bold"), 
                           bg=color, fg='white', relief="raised", bd=3)
            btn.pack(fill="x", padx=10, pady=5)

    def emergency_stop(self):
        """ Arrêt d'urgence style NASA"""
        self.chauffage_etat = False
        self.ventilateur_etat = False
        self.puissance_chauffe.set(0)
        self.puissance_vent.set(0)
        print(" EMERGENCY STOP ACTIVATED")
        self.ajouter_alarme("EMERGENCY STOP ACTIVATED", "ALARM")

    def dessiner_grille_radar(self):
        """ Dessine la grille du radar"""
        
        center_x, center_y = 150, 150
        
        # Cercles concentriques
        for radius in [50, 100, 150]:
            self.radar_canvas.create_oval(center_x-radius, center_y-radius,
                                         center_x+radius, center_y+radius,
                                         outline='#004400', width=1)
        
        # Lignes radiales
        import math
        for angle in range(0, 360, 30):
            rad = math.radians(angle)
            x = center_x + 150 * math.cos(rad)
            y = center_y + 150 * math.sin(rad)
            self.radar_canvas.create_line(center_x, center_y, x, y, 
                                         fill='#004400', width=1)

    def creer_interface_gaming(self):
        """ Interface Gaming RGB avec effets visuels"""
        try:
            self.onglet_gaming = ttk.Frame(self.notebook)
            self.notebook.add(self.onglet_gaming, text=" Gaming RGB")
            
            # Background noir
            gaming_frame = tk.Frame(self.onglet_gaming, bg='#0d1117')
            gaming_frame.pack(fill="both", expand=True)
            
            # === HEADER GAMING ===
            header_gaming = tk.Frame(gaming_frame, bg='#161b22', height=80)
            header_gaming.pack(fill="x", padx=10, pady=10)
            header_gaming.pack_propagate(False)
            
            # Logo/Titre avec effet glow
            title_gaming = tk.Label(header_gaming, text=" THERMAL COMMANDER ", 
                                   font=("Impact", 20, "bold"), 
                                   fg='#ff6b35', bg='#161b22')
            title_gaming.pack(pady=20)
            
            # === PANELS RGB ===
            panels_gaming = tk.Frame(gaming_frame, bg='#0d1117')
            panels_gaming.pack(fill="both", expand=True, padx=10, pady=10)
            
            # Configuration grid
            panels_gaming.grid_columnconfigure(0, weight=1)
            panels_gaming.grid_columnconfigure(1, weight=2)
            panels_gaming.grid_columnconfigure(2, weight=1)
            panels_gaming.grid_rowconfigure(0, weight=1)
            
            # Panel gauche - Stats
            self.creer_panel_stats_gaming(panels_gaming)
            
            print(" Interface Gaming créée")
            
        except Exception as e:
            print(f" Erreur interface Gaming: {e}")

    def creer_panel_stats_gaming(self, parent):
        """ Panel stats style gaming"""
        
        stats_frame = tk.Frame(parent, bg='#21262d', relief="solid", bd=2)
        stats_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Titre avec bordure RGB
        title_frame = tk.Frame(stats_frame, bg='#ff6b35', height=40)
        title_frame.pack(fill="x")
        title_frame.pack_propagate(False)
        
        tk.Label(title_frame, text="⚡ CORE STATS", 
                font=("Arial Black", 14, "bold"), 
                fg='white', bg='#ff6b35').pack(pady=8)
        
        # Stats avec barres de progression
        stats_list = [
            (" TEMP", 0, 60, "#ff4757"),
            (" HEAT", 0, 100, "#ff3838"),
            (" COOL", 0, 100, "#3742fa"),
            (" AI", 0, 100, "#2ed573"),
            (" PWR", 0, 100, "#ffa502")
        ]
        
        for name, min_val, max_val, color in stats_list:
            self.creer_barre_stat_gaming(stats_frame, name, min_val, max_val, color)

    def creer_barre_stat_gaming(self, parent, name, min_val, max_val, color):
        """ Barre de stat style gaming"""
        
        stat_frame = tk.Frame(parent, bg='#21262d')
        stat_frame.pack(fill="x", padx=15, pady=8)
        
        # Label
        tk.Label(stat_frame, text=name, font=("Arial", 10, "bold"), 
                fg='white', bg='#21262d').pack(anchor="w")
        
        # Frame barre
        barre_frame = tk.Frame(stat_frame, bg='#0d1117', height=20, relief="sunken", bd=1)
        barre_frame.pack(fill="x", pady=2)
        barre_frame.pack_propagate(False)
        
        # Barre de progression
        barre_canvas = tk.Canvas(barre_frame, height=18, bg='#0d1117', highlightthickness=0)
        barre_canvas.pack(fill="both", expand=True)
        
        # Rectangle de fond
        barre_canvas.create_rectangle(0, 0, 200, 18, fill='#161b22', outline='')
        
        # Rectangle de progression (sera mis à jour)
        barre_id = barre_canvas.create_rectangle(0, 0, 0, 18, fill=color, outline='')
        
        # Texte valeur
        text_id = barre_canvas.create_text(100, 9, text="0%", 
                                          font=("Arial", 8, "bold"), fill='white')
        
        self.stats_gaming[name] = {
            'canvas': barre_canvas,
            'barre': barre_id,
            'text': text_id,
            'min': min_val,
            'max': max_val,
            'color': color
        }

    def activer_toutes_interfaces(self):
        """Active toutes les nouvelles interfaces"""
        
        try:
            self.creer_interface_nasa_style()
        except Exception as e:
            print(f" Erreur interface NASA: {e}")
        
        try:
            self.creer_interface_gaming()
        except Exception as e:
            print(f" Erreur interface Gaming: {e}")

    def creer_graphique_analyse(self):
        """ Création graphique d'analyse comparative"""
        graph_frame = ttk.LabelFrame(self.onglet_analyse, text="Comparaison Températures Réelles vs Prédites")
        graph_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Figure avec sous-graphiques
        self.fig_analyse, (self.ax_comparaison, self.ax_erreur) = plt.subplots(2, 1, figsize=(12, 8))
        
        # === GRAPHIQUE PRINCIPAL : COMPARAISON ===
        self.line_reelle_analyse = self.ax_comparaison.plot([], [], 
                                                        label=" Température Réelle", 
                                                        color='#2E86C1', linewidth=2)[0]
        self.line_predite_analyse = self.ax_comparaison.plot([], [], 
                                                            label=" Température Prédite", 
                                                            color='#E74C3C', linewidth=2, 
                                                            linestyle='--', alpha=0.8)[0]
        
        self.ax_comparaison.set_title("Comparaison des Températures Réelles et Prédites", 
                                    fontweight='bold', fontsize=14)
        self.ax_comparaison.set_ylabel("Température (°C)")
        self.ax_comparaison.legend(loc='upper left')
        self.ax_comparaison.grid(True, linestyle='--', alpha=0.7)
        
        # === GRAPHIQUE ERREUR ===
        self.line_erreur = self.ax_erreur.plot([], [], 
                                            label= "Erreur Absolue", 
                                            color='#E67E22', linewidth=2)[0]
        self.ax_erreur.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        self.ax_erreur.set_title("Erreur de Prédiction dans le Temps", fontweight='bold')
        self.ax_erreur.set_xlabel("Temps (s)")
        self.ax_erreur.set_ylabel("Erreur (°C)")
        self.ax_erreur.legend(loc='upper left')
        self.ax_erreur.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        self.canvas_analyse = FigureCanvasTkAgg(self.fig_analyse, master=graph_frame)
        self.canvas_analyse.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

    # =============== MÉTHODES JUMEAU NUMÉRIQUE ===============
    
    def choisir_fichier_csv(self):
        """ Sélection fichier CSV spécifique avec prévisualisation"""
        fichier = filedialog.askopenfilename(
            title="Choisir le fichier CSV de données d'entraînement",
            filetypes=[
                ("Fichiers CSV", "*.csv"), 
                ("Fichiers de session", "session_thermique_*.csv"),
                ("Tous les fichiers", "*.*")
            ],
            initialdir=self.jumeau_numerique.dossier_donnees
        )
        
        if fichier:
            try:
                # Configuration du jumeau numérique avec le fichier spécifique
                self.jumeau_numerique.fichier_csv_specifique = fichier
                nom = os.path.basename(fichier)
                self.dossier_label.config(text=f" Fichier: {nom}")
                
                print(f"\n === FICHIER SÉLECTIONNÉ ===")
                print(f" Fichier: {nom}")
                
                # Test de chargement direct
                if self.jumeau_numerique.charger_donnees_csv(fichier):
                    self.dossier_label.config(
                        text=f" {nom} ({len(self.jumeau_numerique.donnees_entrainement)} points)",
                        foreground='green'
                    )
                    
                    # Proposition d'entraînement automatique
                    if messagebox.askyesno("Entraînement IA", 
                                            "Fichier chargé avec succès!\n\n"
                                            "Voulez-vous entraîner le jumeau numérique maintenant?"):
                        self.charger_et_entrainer_jumeau()
                else:
                    self.dossier_label.config(text=f" Erreur chargement {nom}", foreground='red')
                    
            except Exception as e:
                error_msg = f"Erreur lors du chargement du fichier:\n{str(e)}"
                print(f" {error_msg}")
                messagebox.showerror("Erreur", error_msg)
                self.dossier_label.config(text=" Erreur fichier", foreground='red')

    def choisir_dossier_donnees(self):
        """ Sélection dossier de données"""
        dossier = filedialog.askdirectory(
            title="Choisir le dossier contenant les données CSV",
            initialdir=self.jumeau_numerique.dossier_donnees
        )
        
        if dossier:
            self.jumeau_numerique.definir_dossier_donnees(dossier)
            nom = dossier if len(dossier) < 50 else "..." + dossier[-47:]
            self.dossier_label.config(text=f" Dossier: {nom}")

    def charger_et_entrainer_jumeau(self):
        """ Chargement et entraînement du jumeau numérique avec feedback détaillé"""
        try:
            # Mise à jour interface
            self.status_ia_label.config(text=" Jumeau : Chargement...", foreground='orange')
            self.root.update()
            
            # Étape 1: Chargement des données
            print("\n === ENTRAÎNEMENT JUMEAU NUMÉRIQUE ===")
            
            if not hasattr(self.jumeau_numerique, 'donnees_entrainement') or len(self.jumeau_numerique.donnees_entrainement) == 0:
                if not self.jumeau_numerique.charger_donnees_csv():
                    self.status_ia_label.config(text=" Jumeau : Échec chargement", foreground='red')
                    return False
            
            # Étape 2: Validation des données
            if self.jumeau_numerique.valider_donnees_csv(self.jumeau_numerique.donnees_entrainement):
                print(" Validation des données réussie")
            else:
                print(" Données validées avec des avertissements")
            
            # Étape 3: Entraînement
            self.status_ia_label.config(text=" Jumeau : Entraînement...", foreground='orange')
            self.root.update()
            
            if self.jumeau_numerique.entrainer_modele_comportemental():
                self.ia_active = True
                algo = self.jumeau_numerique.meilleur_modele_nom
                precision = self.jumeau_numerique.precision_courante
                erreur = self.jumeau_numerique.erreur_moyenne
                
                # Affichage succès avec détails
                self.status_ia_label.config(
                    text=f" Jumeau {algo} : Actif (R²={precision:.3f}, MAE={erreur:.2f}°C)", 
                    foreground='green'
                )
                
                # Message de confirmation
                messagebox.showinfo("Succès", 
                                     f"Jumeau numérique entraîné avec succès!\n\n"
                                     f" Algorithme: {algo}\n"
                                     f" Précision (R²): {precision:.3f}\n"
                                     f" Erreur moyenne: {erreur:.2f}°C\n"
                                     f" Données: {len(self.jumeau_numerique.donnees_entrainement):,} points\n\n"
                                     f"Le jumeau numérique est maintenant prêt à prédire T2!")
                
                print(" Jumeau numérique prêt pour les prédictions!")
                return True
            else:
                self.status_ia_label.config(text=" Jumeau : Échec entraînement", foreground='red')
                messagebox.showerror("Erreur", "Échec de l'entraînement du jumeau numérique")
                return False
            
        except Exception as e:
            self.status_ia_label.config(text=" Jumeau : Erreur", foreground='red')
            error_msg = f"Erreur lors de l'entraînement:\n{str(e)}"
            print(f" {error_msg}")
            messagebox.showerror("Erreur", error_msg)
            return False

    def appliquer_suggestions_ia(self):
        """ Application des suggestions IA - VERSION CORRIGÉE"""
        
        if not self.ia_active:
            messagebox.showwarning("Attention", 
                                  "Le jumeau numérique n'est pas encore entraîné !\n\n"
                                  "1. Chargez un fichier CSV de données\n"
                                  "2. Cliquez sur 'Charger & Entraîner IA'")
            return
        
        #  GÉNÉRATION DES SUGGESTIONS À LA DEMANDE
        try:
            # Obtenir les températures actuelles
            t1, t2, t3 = self.lire_capteurs()
            
            # Demander la température cible à l'utilisateur
            consigne_actuelle = self.consigne.get()
            
            # Fenêtre de dialogue pour la température cible
            import tkinter.simpledialog as simpledialog
            temperature_cible = simpledialog.askfloat(
                "Optimisation IA",
                f"Température T2 cible ?\n\n"
                f" Températures actuelles:\n"
                f"   T1 (entrée): {t1}°C\n"
                f"   T2 (résistance): {t2}°C\n" 
                f"   T3 (sortie): {t3}°C\n\n"
                f"Consigne actuelle: {consigne_actuelle}°C",
                initialvalue=consigne_actuelle,
                minvalue=15.0,
                maxvalue=80.0
            )
            
            if temperature_cible is None:
                return  # Utilisateur a annulé
            
            #  GÉNÉRATION DES SUGGESTIONS
            self.suggestion_label.config(text=" Calcul suggestions IA...", foreground='orange')
            self.root.update()
            
            suggestions = self.jumeau_numerique.suggerer_parametres_optimaux(
                temperature_cible, t1, t3
            )
            
            if suggestions:
                self.suggestions_ia = suggestions
                sugg = suggestions
                
                #  AFFICHAGE DES SUGGESTIONS
                message_suggestions = (
                    f" Suggestions IA pour atteindre {temperature_cible}°C:\n\n"
                    f" Résistance: {'ON' if sugg['chauffage_on'] else 'OFF'} "
                    f"({sugg['puissance_chauffe']}%)\n"
                    f" Ventilateur: {'ON' if sugg['ventilateur_on'] else 'OFF'} "
                    f"({sugg['puissance_vent']}%)\n\n"
                    f" Température T2 prédite: {sugg['temp_predite']:.1f}°C\n"
                    f" Écart prévu: {sugg['ecart']:.2f}°C\n\n"
                    f"Voulez-vous appliquer ces paramètres ?"
                )
                
                if messagebox.askyesno("Suggestions IA", message_suggestions):
                    #  APPLICATION EFFECTIVE
                    self.appliquer_parametres_ia(suggestions)
                else:
                    self.suggestion_label.config(
                        text=f" Suggestion disponible: Chauffe={sugg['puissance_chauffe']}%, "
                             f"Vent={sugg['puissance_vent']}% → T2≈{sugg['temp_predite']:.1f}°C",
                        foreground='blue'
                    )
            else:
                messagebox.showerror("Erreur IA", 
                                   "Impossible de générer des suggestions.\n"
                                   "Vérifiez que le jumeau numérique est bien entraîné.")
                self.suggestion_label.config(text=" Échec génération suggestions", foreground='red')
                
        except Exception as e:
            print(f" Erreur suggestions IA: {e}")
            messagebox.showerror("Erreur", f"Erreur lors de la génération des suggestions:\n{e}")

    def appliquer_parametres_ia(self, suggestions):
        """ Application effective des paramètres suggérés par l'IA"""
        try:
            sugg = suggestions
            
            print(f"\n === APPLICATION SUGGESTIONS IA ===")
            print(f" Chauffage: {'ON' if sugg['chauffage_on'] else 'OFF'} ({sugg['puissance_chauffe']}%)")
            print(f" Ventilateur: {'ON' if sugg['ventilateur_on'] else 'OFF'} ({sugg['puissance_vent']}%)")
            print(f" T2 prédit: {sugg['temp_predite']:.1f}°C")
            
            # Application des puissances
            self.puissance_chauffe.set(sugg['puissance_chauffe'])
            self.puissance_vent.set(sugg['puissance_vent'])
            
            # Application des états ON/OFF
            if sugg['chauffage_on'] != self.chauffage_etat:
                self.toggle_chauffage()
                print("f Chauffage basculé: {self.chauffage_etat}")
            
            if sugg['ventilateur_on'] != self.ventilateur_etat:
                self.toggle_ventilateur()
                print(f" Ventilateur basculé: {self.ventilateur_etat}")
            
            # Mise à jour PWM
            self.update_pwm_chauffe()
            self.update_pwm_vent()
            
            # Affichage confirmation
            self.suggestion_label.config(
                text=f" IA appliquée ! Chauffe={sugg['puissance_chauffe']}%, "
                     f"Vent={sugg['puissance_vent']}% → T2≈{sugg['temp_predite']:.1f}°C",
                foreground='green'
            )
            
            # Notification réussie
            messagebox.showinfo("Succès", 
                               f"Paramètres IA appliqués avec succès !\n\n"
                               f" Surveillez l'évolution de T2 vers {sugg['temp_predite']:.1f}°C")
            
            print(" Suggestions IA appliquées avec succès !")
            
        except Exception as e:
            print(f" Erreur application IA: {e}")
            messagebox.showerror("Erreur", f"Impossible d'appliquer les suggestions:\n{e}")

    def generer_suggestions_manuelles(self):
        """ Génération manuelle de suggestions sans application"""
        
        if not self.ia_active:
            messagebox.showwarning("Attention", "Jumeau numérique non entraîné !")
            return
        
        try:
            # Température actuelle
            t1, t2, t3 = self.lire_capteurs()
            consigne = self.consigne.get()
            
            # Génération suggestions
            self.suggestions_ia = self.jumeau_numerique.suggerer_parametres_optimaux(
                consigne, t1, t3
            )
            
            if self.suggestions_ia:
                sugg = self.suggestions_ia
                self.suggestion_label.config(
                    text=f" Suggestions: Chauffe={sugg['puissance_chauffe']}%, "
                         f"Vent={sugg['puissance_vent']}% → T2≈{sugg['temp_predite']:.1f}°C",
                    foreground='blue'
                )
                print(f"Suggestions générées: {sugg}")
            else:
                self.suggestion_label.config(text=" Échec génération", foreground='red')
                
        except Exception as e:
            print(f" Erreur génération: {e}")

    def reinitialiser_jumeau(self):
        """ Réinitialisation historique jumeau numérique"""
        if hasattr(self, 'jumeau_numerique'):
            self.jumeau_numerique.reinitialiser_historique()
            self.prediction_label.config(text=" Prédiction T2 : --")
            self.suggestion_label.config(text=" Suggestion : --")
            print(" Historique jumeau numérique réinitialisé")

    def charger_config_ia(self, fichier_config="config_ia.txt"):
        """ Chargement configuration IA"""
        try:
            if os.path.exists(fichier_config):
                with open(fichier_config, 'r') as f:
                    for ligne in f:
                        ligne = ligne.strip()
                        if ligne.startswith("dossier_donnees="):
                            dossier = ligne.split("=", 1)[1]
                            if os.path.exists(dossier):
                                self.jumeau_numerique.definir_dossier_donnees(dossier)
                                print(f" Config chargée: {dossier}")
        except Exception as e:
            print(f" Erreur config: {e}")

    def sauvegarder_config_ia(self, fichier_config="config_ia.txt"):
        """ Sauvegarde configuration IA"""
        try:
            with open(fichier_config, 'w') as f:
                f.write(f"dossier_donnees={self.jumeau_numerique.dossier_donnees}\n")
            print("Configuration IA sauvegardée")
        except Exception as e:
            print(f" Erreur sauvegarde config: {e}")

    # =============== MÉTHODES ANALYSE ===============
    
    def charger_fichier_analyse(self):
        """ Chargement fichier CSV pour analyse"""
        fichier = filedialog.askopenfilename(
            title="Choisir le fichier CSV d'analyse",
            filetypes=[("Fichiers CSV", "*.csv"), ("Tous les fichiers", "*.*")],
            initialdir="."
        )
        
        if fichier:
            try:
                # Utiliser la même méthode de chargement que le jumeau numérique
                delimiter, colonnes = self.jumeau_numerique.detecter_format_csv(fichier)
                
                # Chargement avec gestion du format spécial
                df = pd.read_csv(fichier, delimiter=delimiter)
                
                # Gestion du format avec toutes les colonnes dans une seule
                if len(df.columns) == 1 and ';' in str(df.iloc[0, 0]):
                    col_name = df.columns[0]
                    data_rows = []
                    for idx, row in df.iterrows():
                        if idx == 0:
                            continue
                        try:
                            values = str(row[col_name]).split(';')
                            if len(values) >= 9:
                                data_rows.append(values)
                        except:
                            continue
                    
                    colonnes_attendues = ['name', 'date', 'time', 'Consigne', 'T1', 'T2', 'T3', 
                                         'Puissance Vent', 'Puissance Chauffe']
                    df = pd.DataFrame(data_rows, columns=colonnes_attendues)
                
                # Conversion colonnes numériques
                colonnes_numeriques = ['T1', 'T2', 'T3', 'Puissance Vent', 'Puissance Chauffe', 'Consigne']
                for col in colonnes_numeriques:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Ajout colonnes manquantes
                if 'Temps' not in df.columns:
                    df['Temps'] = range(len(df))
                if 'Ventilateur ON' not in df.columns:
                    df['Ventilateur ON'] = (df['Puissance Vent'] > 0).astype(int)
                if 'Chauffage ON' not in df.columns:
                    df['Chauffage ON'] = (df['Puissance Chauffe'] > 0).astype(int)
                
                # Nettoyage données
                self.donnees_analyse = df.dropna(subset=['T1', 'T2', 'T3'])
                
                # Génération prédictions si jumeau actif
                if hasattr(self, 'jumeau_numerique') and self.jumeau_numerique.modele_actif:
                    self.generer_predictions_analyse()
                
                # Mise à jour interface
                nom_fichier = os.path.basename(fichier)
                self.fichier_analyse_label.config(
                    text=f" {nom_fichier} ({len(self.donnees_analyse)} points)"
                )
                
                # Analyse automatique
                self.actualiser_analyse()
                
                print(f" Analyse chargée: {len(self.donnees_analyse)} points")
                
            except Exception as e:
                messagebox.showerror("Erreur", f"Impossible de charger le fichier:\n{e}")
                print(f" Erreur chargement analyse: {e}")

    def generer_predictions_analyse(self):
        """ Génération prédictions pour données d'analyse"""
        try:
            predictions = []
            
            for _, row in self.donnees_analyse.iterrows():
                pred = self.jumeau_numerique.predire_t2_temps_reel(
                    row['T1'], row['T3'], 
                    row['Puissance Chauffe'], row['Puissance Vent'],
                    row['Chauffage ON'] if 'Chauffage ON' in row else row['Puissance Chauffe'] > 0,
                    row['Ventilateur ON'] if 'Ventilateur ON' in row else row['Puissance Vent'] > 0
                )
                predictions.append(pred if pred is not None else row['T2'])
            
            self.donnees_analyse['T2_Predit'] = predictions
            print(f" {len(predictions)} prédictions générées")
            
        except Exception as e:
            print(f" Erreur génération prédictions: {e}")

    def calculer_metriques_performance(self, donnees=None):
        """ Calcul métriques de performance"""
        try:
            if donnees is None:
                if not hasattr(self, 'donnees_analyse') or 'T2_Predit' not in self.donnees_analyse.columns:
                    return None
                donnees = self.donnees_analyse
            
            y_true = donnees['T2'].values
            y_pred = donnees['T2_Predit'].values
            
            # Filtrage valeurs valides
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            
            if len(y_true) == 0:
                return None
            
            # Calcul métriques
            mae = np.mean(np.abs(y_true - y_pred))
            rmse = np.sqrt(np.mean((y_true - y_pred)**2))
            r2 = 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            return {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'mape': mape,
                'n_points': len(y_true)
            }
            
        except Exception as e:
            print(f" Erreur calcul métriques: {e}")
            return None

    def actualiser_analyse(self):
        """ Actualisation analyse et graphiques"""
        try:
            if not hasattr(self, 'donnees_analyse'):
                return
            
            # Application filtre période
            donnees_filtrees = self.filtrer_donnees_periode()
            
            if donnees_filtrees is None or len(donnees_filtrees) == 0:
                return
            
            # Calcul métriques
            metriques = self.calculer_metriques_performance(donnees_filtrees)
            
            if metriques:
                # Mise à jour affichage métriques
                self.mae_label.config(text=f"MAE : {metriques['mae']:.3f}°C")
                self.rmse_label.config(text=f"RMSE : {metriques['rmse']:.3f}°C")
                self.r2_label.config(text=f"R² : {metriques['r2']:.3f}")
                self.mape_label.config(text=f"MAPE : {metriques['mape']:.2f}%")
                
                # Couleurs selon performance
                couleur_r2 = '#27AE60' if metriques['r2'] > 0.8 else '#F39C12' if metriques['r2'] > 0.6 else '#E74C3C'
                self.r2_label.config(foreground=couleur_r2)
            
            # Mise à jour graphiques
            self.mettre_a_jour_graphiques_analyse(donnees_filtrees)
            
        except Exception as e:
            print(f" Erreur actualisation analyse: {e}")

    def filtrer_donnees_periode(self):
        """ Filtrage données selon période sélectionnée"""
        try:
            if not hasattr(self, 'donnees_analyse'):
                return None
            
            donnees = self.donnees_analyse.copy()
            periode = self.periode_var.get()
            
            if periode == "Complète":
                self.periode_info_label.config(text=f"({len(donnees)} points)")
                return donnees
            
            # Calcul temps limite
            temps_max = donnees['Temps'].max()
            
            if periode == "Dernière heure":
                temps_limite = temps_max - 3600
            elif periode == "Dernières 30 min":
                temps_limite = temps_max - 1800
            elif periode == "Dernières 10 min":
                temps_limite = temps_max - 600
            else:
                return donnees
            
            donnees_filtrees = donnees[donnees['Temps'] >= temps_limite]
            self.periode_info_label.config(text=f"({len(donnees_filtrees)} points)")
            
            return donnees_filtrees
            
        except Exception as e:
            print(f" Erreur filtrage période: {e}")
            return self.donnees_analyse if hasattr(self, 'donnees_analyse') else None

    def mettre_a_jour_graphiques_analyse(self, donnees):
        """ Mise à jour graphiques d'analyse"""
        try:
            if 'T2_Predit' not in donnees.columns:
                return
            
            temps = donnees['Temps'].values
            t2_reel = donnees['T2'].values
            t2_predit = donnees['T2_Predit'].values
            
            # Graphique principal : comparaison
            self.line_reelle_analyse.set_data(temps, t2_reel)
            self.line_predite_analyse.set_data(temps, t2_predit)
            
            # Ajustement axes comparaison
            if len(temps) > 0:
                self.ax_comparaison.set_xlim(temps.min(), temps.max())
                temp_min = min(t2_reel.min(), t2_predit.min()) - 2
                temp_max = max(t2_reel.max(), t2_predit.max()) + 2
                self.ax_comparaison.set_ylim(temp_min, temp_max)
            
            # Graphique erreur
            erreur = np.abs(t2_reel - t2_predit)
            self.line_erreur.set_data(temps, erreur)
            
            # Ajustement axes erreur
            if len(temps) > 0:
                self.ax_erreur.set_xlim(temps.min(), temps.max())
                self.ax_erreur.set_ylim(0, erreur.max() * 1.1)
            
            # Redessiner
            self.fig_analyse.canvas.draw_idle()
            
        except Exception as e:
            print(f" Erreur mise à jour graphiques: {e}")

    def changer_periode_analyse(self, event=None):
        """⏱️ Changement période d'analyse"""
        self.actualiser_analyse()

    def analyse_detaillee(self):
        """ Fenêtre d'analyse détaillée"""
        try:
            if not hasattr(self, 'donnees_analyse') or 'T2_Predit' not in self.donnees_analyse.columns:
                messagebox.showwarning("Attention", "Aucune donnée d'analyse disponible")
                return
            
            # Nouvelle fenêtre
            fenetre_detail = tk.Toplevel(self.root)
            fenetre_detail.title(" Analyse Détaillée - Jumeau Numérique")
            fenetre_detail.geometry("1000x700")
            
            # Calcul statistiques avancées
            donnees = self.donnees_analyse
            y_true = donnees['T2'].values
            y_pred = donnees['T2_Predit'].values
            
            # Création graphiques multiples
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. Scatter plot prédiction vs réalité
            ax1.scatter(y_true, y_pred, alpha=0.6, c='#3498DB')
            ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            ax1.set_xlabel('Température Réelle (°C)')
            ax1.set_ylabel('Température Prédite (°C)')
            ax1.set_title('Prédiction vs Réalité')
            ax1.grid(True, alpha=0.3)
            
            # 2. Histogramme des erreurs
            erreurs = y_pred - y_true
            ax2.hist(erreurs, bins=30, alpha=0.7, color='#E74C3C')
            ax2.set_xlabel('Erreur de Prédiction (°C)')
            ax2.set_ylabel('Fréquence')
            ax2.set_title('Distribution des Erreurs')
            ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            ax2.grid(True, alpha=0.3)
            
            # 3. Erreur vs temps
            ax3.plot(donnees['Temps'], np.abs(erreurs), color='#F39C12', linewidth=1)
            ax3.set_xlabel('Temps (s)')
            ax3.set_ylabel('Erreur Absolue (°C)')
            ax3.set_title('Évolution de l\'Erreur dans le Temps')
            ax3.grid(True, alpha=0.3)
            
            # 4. Résidus vs prédictions
            ax4.scatter(y_pred, erreurs, alpha=0.6, c='#9B59B6')
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax4.set_xlabel('Température Prédite (°C)')
            ax4.set_ylabel('Résidus (°C)')
            ax4.set_title('Résidus vs Prédictions')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Intégration dans fenêtre
            canvas_detail = FigureCanvasTkAgg(fig, master=fenetre_detail)
            canvas_detail.get_tk_widget().pack(fill="both", expand=True)
            
            print(" Analyse détaillée affichée")
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur analyse détaillée:\n{e}")

    def exporter_analyse(self):
        """ Export analyse vers fichier"""
        try:
            if not hasattr(self, 'donnees_analyse'):
                messagebox.showwarning("Attention", "Aucune donnée à exporter")
                return
            
            # Sélection fichier de sauvegarde
            fichier = filedialog.asksaveasfilename(
                title="Exporter analyse",
                defaultextension=".csv",
                filetypes=[("Fichiers CSV", "*.csv"), ("Tous les fichiers", "*.*")]
            )
            
            if fichier:
                # Export données avec métriques
                donnees_export = self.donnees_analyse.copy()
                
                if 'T2_Predit' in donnees_export.columns:
                    donnees_export['Erreur_Absolue'] = np.abs(donnees_export['T2'] - donnees_export['T2_Predit'])
                    donnees_export['Erreur_Relative'] = donnees_export['Erreur_Absolue'] / donnees_export['T2'] * 100
                
                donnees_export.to_csv(fichier, sep=';', index=False)
                
                messagebox.showinfo("Succès", f"Analyse exportée vers:\n{fichier}")
                print(f" Analyse exportée: {fichier}")
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur export:\n{e}")

    # =============== MÉTHODES ACQUISITION ET CONTRÔLE ===============

    def demarrer_acquisition(self):
        """ Démarrage de l'acquisition et du thread principal"""
        self.running = True
        self.thread = threading.Thread(target=self.update_loop)
        self.thread.daemon = True
        self.thread.start()

    def lire_capteurs(self):
        """ Lecture des capteurs (physiques ou simulation)"""
        if RPI:
            try:
                adc_vals = [lire_adc(ch) for ch in (0, 1, 2)]
                return [adc_to_temperature(val) for val in adc_vals]
            except Exception as e:
                print(f"❌ Erreur lecture ADC: {e}")
                return [25.0, 25.0, 25.0]
        else:
            #  SIMULATION PC avec comportement thermique réaliste
            temps_ecoule = time.time() - self.debut_acquisition
            
            # Températures de base avec variation naturelle
            t1_base = 26 + math.sin(temps_ecoule / 50) * 0.3
            t2_base = 25 + math.sin(temps_ecoule / 45 + 1) * 0.3
            t3_base = 24 + math.sin(temps_ecoule / 40 + 2) * 0.3
            
            # Initialisation historique températures
            if not hasattr(self, 'last_temps'):
                self.last_temps = [t1_base, t2_base, t3_base]
            
            #  Effet chauffage (impact principal sur T2)
            if self.chauffage_etat:
                factor = self.puissance_chauffe.get() / 100.0
                t1_base += 2 * factor    # Effet modéré sur entrée
                t2_base += 8 * factor    # Effet fort sur résistance
                t3_base += 1 * factor    # Effet faible sur sortie
            
            #  Effet ventilateur (refroidissement)
            if self.ventilateur_etat:
                factor = self.puissance_vent.get() / 100.0
                t1_base -= 0.5 * factor  # Effet faible sur entrée
                t2_base -= 4 * factor    # Effet fort sur résistance
                t3_base -= 3 * factor    # Effet modéré sur sortie
            
            # Inertie thermique réaliste
            inertie = 0.98
            new_temps = [
                self.last_temps[0] * inertie + t1_base * (1-inertie),
                self.last_temps[1] * inertie + t2_base * (1-inertie),
                self.last_temps[2] * inertie + t3_base * (1-inertie)
            ]
            
            # Bruit de mesure réaliste
            result = [
                round(max(15, min(80, new_temps[0] + random.uniform(-0.1, 0.1))), 2),
                round(max(15, min(80, new_temps[1] + random.uniform(-0.1, 0.1))), 2),
                round(max(15, min(80, new_temps[2] + random.uniform(-0.1, 0.1))), 2)
            ]
            
            self.last_temps = new_temps
            return result

    def update_predictions_jumeau_numerique(self, t1, t2, t3, temps_actuel):
        """ Mise à jour des prédictions du jumeau numérique"""
        if not self.jumeau_numerique.modele_actif:
            return
        
        try:
            # Prédiction T2 en temps réel
            t2_predit = self.jumeau_numerique.predire_t2_temps_reel(
                t1, t3,
                self.puissance_chauffe.get(),
                self.puissance_vent.get(),
                self.chauffage_etat,
                self.ventilateur_etat
            )
            
            if t2_predit is not None:
                # Enregistrement pour comparaison
                self.jumeau_numerique.ajouter_mesure_temps_reel(temps_actuel, t2, t2_predit)
                
                # Calcul précision
                ecart = abs(t2_predit - t2)
                
                # Affichage prédiction
                self.prediction_label.config(
                    text=f" Jumeau Numérique T2 : {t2_predit}°C | Réel: {t2}°C"
                )
                
                # Affichage précision avec couleur
                if ecart < 0.5:
                    couleur, precision = '#27ae60', "Excellent"
                elif ecart < 1.5:
                    couleur, precision = '#f39c12', " Bon" 
                else:
                    couleur, precision = '#e74c3c', " À calibrer"
                
                self.suggestion_label.config(
                    text=f" Précision Jumeau : {precision} (écart: {ecart:.1f}°C)",
                    foreground=couleur
                )
                
        except Exception as e:
            print(f" Erreur jumeau numérique: {e}")

    def update_graphique_avec_jumeau(self):
        """ Mise à jour du graphique avec prédictions IA"""
        visible_time = 60
        
        if len(self.tps) > 0:
            # Calcul fenêtre d'affichage
            if len(self.tps) <= visible_time:
                x_min = 0
                x_max = max(visible_time, self.tps[-1] + 5)
            else:
                x_min = self.tps[-1] - visible_time
                x_max = self.tps[-1] + 5
            
            # Mise à jour courbes capteurs physiques
            for i, line in enumerate(self.lines):
                line.set_data(self.tps, self.temp_capteurs[i])
            
            # Mise à jour courbe prédiction jumeau numérique
            if self.jumeau_numerique.modele_actif:
                temps_pred, t2_reel_hist, t2_pred_hist = self.jumeau_numerique.obtenir_donnees_comparaison()
                if temps_pred and t2_pred_hist:
                    self.line_prediction_ia.set_data(temps_pred, t2_pred_hist)
                else:
                    self.line_prediction_ia.set_data([], [])
            else:
                self.line_prediction_ia.set_data([], [])
            
            # Ajustement limites
            self.ax.set_xlim(x_min, x_max)
            
            # Échelle Y dynamique incluant prédictions
            all_temps = []
            for temps in self.temp_capteurs:
                if temps:
                    all_temps.extend(temps)
            
            if self.jumeau_numerique.modele_actif:
                _, _, t2_pred_hist = self.jumeau_numerique.obtenir_donnees_comparaison()
                if t2_pred_hist:
                    all_temps.extend(t2_pred_hist)
            
            if all_temps:
                min_temp = min(all_temps) - 2
                max_temp = max(all_temps) + 2
                self.ax.set_ylim(min(15, min_temp), max(50, max_temp))
            
            self.fig.canvas.draw_idle()

    def update_loop(self):
        """ Boucle principale d'acquisition et de traitement - CORRIGÉE BETA TEST"""
        cycle_count = 0
        
        while self.running:
            try:
                # Temps relatif depuis début acquisition
                now = round(time.time() - self.debut_acquisition, 1)
                t1, t2, t3 = self.lire_capteurs()
                
                self.mettre_a_jour_interfaces_modernes(t1, t2, t3, now)

                #  CORRECTION 1 - Mise à jour graphique temps réel (TOUJOURS à 10Hz)
                self.temp_capteurs[0].append(t1)
                self.temp_capteurs[1].append(t2)  #  LIGNE MANQUANTE AJOUTÉE !
                self.temp_capteurs[2].append(t3)
                self.tps.append(now)
                
                # Limitation points graphique (10 minutes à 10Hz = 6000 points)
                if len(self.tps) > 6000:
                    self.tps = self.tps[-6000:]
                    for i in range(3):
                        self.temp_capteurs[i] = self.temp_capteurs[i][-6000:]
                
                #  Mise à jour affichage températures (toujours)
                self.capteur1_label.config(text=f" T1 Entrée Ventillo: {t1} °C")
                self.capteur2_label.config(text=f" T2 Résistance : {t2} °C")
                self.capteur3_label.config(text=f" T3 Sortie : {t3} °C")
                
                #  CORRECTION 2 - Animations toujours actives
                self.rotate_fan()
                self.update_heater_animation()

                #  ENREGISTREMENT DONNÉES CSV : TOUTES LES 5 SECONDES (séparé du graphique)
                if not hasattr(self, 'derniere_mesure'):
                    self.derniere_mesure = 0
                    
                if now - self.derniere_mesure >= 5.0:
                    self.derniere_mesure = now
                    
                    # Sauvegarde état actuel pour CSV
                    etat_actuel = {
                        'temps': now, 't1': t1, 't2': t2, 't3': t3, 'consigne': self.consigne.get(),
                        'puissance_chauffe': self.puissance_chauffe.get(),
                        'puissance_vent': self.puissance_vent.get(),
                        'ventilateur_on': 1 if self.ventilateur_etat else 0,
                        'chauffage_on': 1 if self.chauffage_etat else 0,
                        'regulation_active': 1 if self.regulation else 0
                    }
                    self.historique_etats.append(etat_actuel)
                    
                    # Limitation mémoire (24H à 5s = 17280 points)
                    if len(self.historique_etats) > 17280:
                        self.historique_etats = self.historique_etats[-17280:]
                    
                    print(f" Mesure CSV enregistrée: T1={t1}°C, T2={t2}°C, T3={t3}°C (point #{len(self.historique_etats)})")
                
                #  Mise à jour jumeau numérique (toujours pour prédictions temps réel)
                if self.ia_active:
                    self.update_predictions_jumeau_numerique(t1, t2, t3, now)
                
                # Status IA
                if self.ia_active:
                    self.status_ia_label.config(text=" Jumeau : Actif", foreground='green')
                else:
                    self.status_ia_label.config(text=" Jumeau : Non chargé", foreground='red')

                #  Suggestions IA pour optimisation (toutes les 30 secondes)
                if self.ia_active and self.regulation and cycle_count % 300 == 0:  # 30s à 10Hz
                    self.suggestions_ia = self.jumeau_numerique.suggerer_parametres_optimaux(
                        self.consigne.get(), t1, t3
                    )
                    
                    if self.suggestions_ia:
                        sugg = self.suggestions_ia
                        print(f" IA suggère: Chauffe={sugg['puissance_chauffe']}%, "
                            f"Vent={sugg['puissance_vent']}% → T2={sugg['temp_predite']:.1f}°C")

                #  Régulation automatique (hystérésis)
                if self.regulation:
                    consigne = self.consigne.get()
                    if t2 > consigne + 2 and not self.ventilateur_etat:
                        self.ventilateur_etat = True
                        self.status_ventil_label.config(text="Ventilateur : ON (régulation)", foreground='green')
                    elif t2 < consigne - 2 and self.ventilateur_etat:
                        self.ventilateur_etat = False
                        self.status_ventil_label.config(text="Ventilateur : OFF (régulation)", foreground='red')
                    
                    if RPI:
                        pwm_vent.ChangeDutyCycle(self.puissance_vent.get() if self.ventilateur_etat else 0)
                        GPIO.output(VENT_IN2, GPIO.LOW)
                
                #  Gestion automatisation
                if self.automatisation_active:
                    temps_restant = self.prochaine_automatisation - time.time()
                    if temps_restant <= 0:
                        self.executer_automatisation()
                    else:
                        minutes = int(temps_restant // 60)
                        secondes = int(temps_restant % 60)
                        self.timer_auto_label.config(
                            text=f"Prochaine action dans : {minutes:02d}:{secondes:02d}"
                        )
                else:
                    self.timer_auto_label.config(text="Automatisation désactivée")
                
                # CORRECTION 3 - Mise à jour graphique (TOUJOURS)
                self.update_graphique_avec_jumeau()
                
                #  SAUVEGARDE SÉCURITÉ : TOUTES LES 10 MINUTES
                if not hasattr(self, 'derniere_sauvegarde'):
                    self.derniere_sauvegarde = 0
                    
                if now - self.derniere_sauvegarde >= 600:  # 10 minutes
                    self.derniere_sauvegarde = now
                    self.sauvegarde_incrementale()
                    print(f" Sauvegarde sécurité: {len(self.historique_etats)} points")
                
                cycle_count += 1
                time.sleep(0.1)  # 10 Hz
                
            except Exception as e:
                print(f" Erreur update_loop: {e}")
                time.sleep(1)

    def mettre_a_jour_interfaces_modernes(self, t1, t2, t3, temps):
        """ Mise à jour toutes les interfaces modernes"""
        
        try:
            # Interface NASA
            if hasattr(self, 'telemetry_data') and self.telemetry_data:
                self.telemetry_data.get('T1_INLET', tk.Label()).config(text=f"{t1:.2f}")
                self.telemetry_data.get('T2_CORE', tk.Label()).config(text=f"{t2:.2f}")
                self.telemetry_data.get('T3_OUTLET', tk.Label()).config(text=f"{t3:.2f}")
                self.telemetry_data.get('PWR_HEAT', tk.Label()).config(text=f"{self.puissance_chauffe.get():.1f}")
                self.telemetry_data.get('PWR_FAN', tk.Label()).config(text=f"{self.puissance_vent.get():.1f}")
                
                # Mission timer
                if hasattr(self, 'mission_time_label'):
                    mission_time = int(temps)
                    hours = mission_time // 3600
                    minutes = (mission_time % 3600) // 60
                    seconds = mission_time % 60
                    self.mission_time_label.config(text=f"MISSION TIME: {hours:02d}:{minutes:02d}:{seconds:02d}")
                
                # Radar
                if hasattr(self, 'radar_canvas'):
                    self.mettre_a_jour_radar(t1, t2, t3)
            
            # Interface Gaming
            if hasattr(self, 'stats_gaming') and self.stats_gaming:
                self.mettre_a_jour_stats_gaming(t1, t2, t3)
                
        except Exception as e:
            print(f" Erreur mise à jour interfaces: {e}")

    def mettre_a_jour_radar(self, t1, t2, t3):
        """ Mise à jour radar thermique"""
        
        try:
            # Effacer anciens points
            for point_id in self.radar_points.values():
                self.radar_canvas.delete(point_id)
            
            center_x, center_y = 150, 150
            
            # Placer les températures sur le radar
            positions = [(0, -120), (120, 0), (0, 120)]  # T1 haut, T2 droite, T3 bas
            temperatures = [t1, t2, t3]
            colors = ['#ff4444', '#44ff44', '#4444ff']
            labels = ['T1', 'T2', 'T3']
            
            for i, ((dx, dy), temp, color, label) in enumerate(zip(positions, temperatures, colors, labels)):
                x = center_x + dx
                y = center_y + dy
                
                # Point pulsant selon température
                size = 5 + (temp - 20) * 0.5
                size = max(3, min(15, size))
                
                point_id = self.radar_canvas.create_oval(x-size, y-size, x+size, y+size,
                                                        fill=color, outline='white', width=2)
                
                # Label
                text_id = self.radar_canvas.create_text(x, y-20, text=f"{label}\n{temp:.1f}°C",
                                                       fill=color, font=("Courier", 8, "bold"))
                
                self.radar_points[f'{label}_point'] = point_id
                self.radar_points[f'{label}_text'] = text_id
                
        except Exception as e:
            print(f" Erreur radar: {e}")

    def mettre_a_jour_stats_gaming(self, t1, t2, t3):
        """ Mise à jour stats gaming"""
        
        try:
            # Valeurs à afficher
            values = {
                " TEMP": max(t1, t2, t3),
                " HEAT": self.puissance_chauffe.get(),
                " COOL": self.puissance_vent.get(),
                " AI": 100 if self.ia_active else 0,
                " PWR": (self.puissance_chauffe.get() + self.puissance_vent.get()) / 2
            }
            
            for name, value in values.items():
                if name in self.stats_gaming:
                    stat = self.stats_gaming[name]
                    
                    # Calculer pourcentage
                    pct = (value - stat['min']) / (stat['max'] - stat['min']) * 100
                    pct = max(0, min(100, pct))
                    
                    # Mettre à jour barre
                    width = pct * 2  # 200px max
                    stat['canvas'].coords(stat['barre'], 0, 0, width, 18)
                    stat['canvas'].itemconfig(stat['text'], text=f"{value:.1f}")
                    
        except Exception as e:
            print(f" Erreur stats gaming: {e}")

    # =============== MÉTHODES INTERFACE ===============
    
    def update_pwm_chauffe(self, *args):
        """ Mise à jour PWM résistance chauffante"""
        value = self.puissance_chauffe.get()
        self.texte_puissance_chauffe.set(f"{value:.0f} %")
        if RPI and self.chauffage_etat:
            pwm_chauffe.ChangeDutyCycle(value)

    def update_pwm_vent(self, *args):
        """ Mise à jour PWM ventilateur"""
        value = self.puissance_vent.get()
        self.texte_puissance_vent.set(f"{value:.0f} %")
        if RPI and self.ventilateur_etat:
            pwm_vent.ChangeDutyCycle(value)

    def toggle_chauffage(self):
        """ Basculement état résistance chauffante"""
        self.chauffage_etat = not self.chauffage_etat
        etat = 'ON' if self.chauffage_etat else 'OFF'
        color = 'green' if self.chauffage_etat else 'red'
        self.status_chauffage_label.config(text=f"Résistance : {etat}", foreground=color)
        
        if RPI:
            if self.chauffage_etat:
                pwm_chauffe.ChangeDutyCycle(self.puissance_chauffe.get())
            else:
                pwm_chauffe.ChangeDutyCycle(0)
            GPIO.output(HEAT_IN4, GPIO.LOW)

    def toggle_ventilateur(self):
        """ Basculement état ventilateur"""
        self.ventilateur_etat = not self.ventilateur_etat
        etat = 'ON' if self.ventilateur_etat else 'OFF'
        color = 'green' if self.ventilateur_etat else 'red'
        self.status_ventil_label.config(text=f"Ventilateur : {etat}", foreground=color)
        
        if RPI:
            if self.ventilateur_etat:
                pwm_vent.ChangeDutyCycle(self.puissance_vent.get())
            else:
                pwm_vent.ChangeDutyCycle(0)
            GPIO.output(VENT_IN2, GPIO.LOW)

    def toggle_regulation(self):
        """ Basculement régulation automatique"""
        self.regulation = not self.regulation
        etat = 'Activée' if self.regulation else 'Désactivée'
        color = 'green' if self.regulation else 'red'
        self.status_regulation_label.config(text=f"Régulation : {etat}", foreground=color)

    def toggle_automatisation(self):
        """Basculement automatisation jumeau numérique"""
        self.automatisation_active = not self.automatisation_active
        etat = 'ON' if self.automatisation_active else 'OFF'
        color = 'green' if self.automatisation_active else 'red'
        self.status_auto_label.config(text=f"Automatisation : {etat}", foreground=color)
        
        if self.automatisation_active:
            self.prochaine_automatisation = time.time() + 300  # 5 min
            print(" Automatisation activée - Prochaine action dans 5 minutes")
        else:
            self.timer_auto_label.config(text="Automatisation désactivée")

    # =============== MÉTHODES AUTOMATISATION ===============
    
    def executer_automatisation(self):
        """ Exécution séquence d'automatisation intelligente avec ventilateur optimisé"""
        print("\n === AUTOMATISATION JUMEAU NUMÉRIQUE ===")
        
        # Génération séquence aléatoire mais réaliste
        action_vent = random.random() < 0.8
        action_chauffe = random.random() < 0.8
        
        # Application états ON/OFF
        if action_vent != self.ventilateur_etat:
            self.toggle_ventilateur()
        if action_chauffe != self.chauffage_etat:
            self.toggle_chauffage()
        
        #  MODIFICATION: Ventilateur entre 66% et 100% (au lieu de 10-100%)
        if self.ventilateur_etat:
            puissance_vent = random.randint(66, 100)  # Plage optimisée
            self.puissance_vent.set(puissance_vent)
            self.update_pwm_vent()
            print(f" Ventilateur: {puissance_vent}% (plage optimisée 66-100%)")
        else:
            print(" Ventilateur: OFF")
        
        # Réglage chauffage (reste 10-100%)
        if self.chauffage_etat:
            puissance_chauffe = random.randint(10, 100)
            self.puissance_chauffe.set(puissance_chauffe)
            self.update_pwm_chauffe()
            print(f" Chauffage: {puissance_chauffe}%")
        else:
            print(" Chauffage: OFF")
        
        # Gestion régulation
        if random.random() < 0.8:
            consigne = random.randint(25, 60)
            self.consigne.set(consigne)
            if not self.regulation:
                self.toggle_regulation()
            print(f" Régulation: {consigne}°C")
        else:
            if self.regulation:
                self.toggle_regulation()
            print("Régulation: OFF")
        
        # Programmation prochaine automatisation
        self.prochaine_automatisation = time.time() + 300  # 5 min
        print(" === AUTOMATISATION TERMINÉE ===\n")
        
        # Sauvegarde après automatisation
        self.sauvegarde_incrementale()

    # =============== MÉTHODES SAUVEGARDE ===============
    
    def creer_fichier_session(self):
        """ Création fichier de session unique"""
        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.fichier_session_actuel = f"session_thermique_{timestamp}.csv"
            
            with open(self.fichier_session_actuel, "w", newline='') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow([
                    "Temps", "T1", "T2", "T3", "Consigne", 
                    "Puissance Chauffe", "Puissance Vent", 
                    "Ventilateur ON", "Chauffage ON", "Regulation Active"
                ])
            
            print(f" Session créée: {self.fichier_session_actuel}")
            
        except Exception as e:
            print(f" Erreur création session: {e}")
            self.fichier_session_actuel = None

    def sauvegarde_incrementale(self):
        """ Sauvegarde incrémentale (nouveaux points seulement)"""
        try:
            if not self.fichier_session_actuel:
                return False
            
            nouveaux_points = self.historique_etats[self.points_depuis_derniere_sauvegarde:]
            
            if len(nouveaux_points) == 0:
                return True
            
            with open(self.fichier_session_actuel, "a", newline='') as f:
                writer = csv.writer(f, delimiter=';')
                
                for etat in nouveaux_points:
                    writer.writerow([
                        etat['temps'], etat['t1'], etat['t2'], etat['t3'], etat['consigne'],
                        etat['puissance_chauffe'], etat['puissance_vent'],
                        etat['ventilateur_on'], etat['chauffage_on'], etat['regulation_active']
                    ])
            
            self.points_depuis_derniere_sauvegarde = len(self.historique_etats)
            
            self.status_sauvegarde_label.config(
                text=f" Sauvegarde : OK ({len(self.historique_etats)} points)", 
                foreground='green'
            )
            
            return True
            
        except Exception as e:
            print(f" Erreur sauvegarde: {e}")
            self.status_sauvegarde_label.config(
                text=f" Erreur: {str(e)[:20]}...", 
                foreground='red'
            )
            return False

    def sauvegarder(self):
        """ Sauvegarde complète de sécurité"""
        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"donnees_thermiques_backup_{timestamp}.csv"
            
            with open(filename, "w", newline='') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow([
                    "Temps", "T1", "T2", "T3", "Consigne", 
                    "Puissance Chauffe", "Puissance Vent", 
                    "Ventilateur ON", "Chauffage ON", "Regulation Active"
                ])
                
                for etat in self.historique_etats:
                    writer.writerow([
                        etat['temps'], etat['t1'], etat['t2'], etat['t3'], etat['consigne'],
                        etat['puissance_chauffe'], etat['puissance_vent'],
                        etat['ventilateur_on'], etat['chauffage_on'], etat['regulation_active']
                    ])
            
            print(f" Sauvegarde complète: {filename}")
            self.status_sauvegarde_label.config(
                text=f" Backup : {len(self.historique_etats)} points", 
                foreground='green'
            )
            
        except Exception as e:
            print(f" Erreur sauvegarde complète: {e}")
            self.status_sauvegarde_label.config(
                text=f" Erreur backup: {str(e)[:20]}...", 
                foreground='red'
            )

    def stop(self):
        """ Arrêt propre de l'application"""
        print("\n === ARRÊT APPLICATION ===")
        self.running = False
        self.automatisation_active = False
        
        # Sauvegarde finale
        print("Sauvegarde finale...")
        self.sauvegarde_incrementale()
        self.sauvegarder()
        self.sauvegarder_config_ia()
        
        # Nettoyage GPIO
        if RPI:
            try:
                pwm_vent.stop()
                pwm_chauffe.stop()
                GPIO.cleanup()
                print("🔧 GPIO nettoyé")
            except:
                pass
        
        print(f" Session finale: {self.fichier_session_actuel}")
        print(" Application fermée proprement")
        self.root.destroy()


# =============== POINT D'ENTRÉE PRINCIPAL ===============

if __name__ == "__main__":
    """
     LANCEMENT DE L'APPLICATION
    
    FONCTIONNEMENT:
    1. Initialisation interface graphique
    2. Démarrage acquisition capteurs (10 Hz)
    3. Chargement jumeau numérique (si données disponibles)
    4. Boucle principale: mesures → prédictions → affichage
    5. Arrêt propre avec sauvegarde
    
    🔧 OPTIMISATIONS:
    - Enregistrement données: toutes les 10s (au lieu de 0.3s)
    - Sauvegarde sécurité: toutes les 10 minutes
    - Ventilateur automatisation: 66-100% (optimisé)
    """
    print(" === DÉMARRAGE JUMEAU NUMÉRIQUE THERMIQUE OPTIMISÉ ===")
    print(" Objectif: Prédiction T2 via apprentissage automatique")
    print(" Données: Capteurs physiques vs Prédictions IA")
    print(" Algorithmes: RandomForest, GradientBoosting, Neural Network")
    print(" OPTIMISATIONS:")
    print("    Enregistrement: toutes les 5s (compromis optimal)")
    print("    Sauvegarde sécurité: toutes les 5 minutes")
    print("    Ventilateur auto: 66-100% (plage optimisée)")
    print("=" * 60)
    
    try:
        root = tk.Tk()
        app = ControleThermique(root)
        
        # Gestionnaire fermeture propre
        root.protocol("WM_DELETE_WINDOW", app.stop)
        
        # Démarrage interface
        root.mainloop()
        
    except KeyboardInterrupt:
        print("\n Interruption clavier détectée")
        if 'app' in locals():
            app.stop()
    except Exception as e:
        print(f" Erreur critique: {e}")
        if RPI:
            try:
                GPIO.cleanup()
            except:
                pass
