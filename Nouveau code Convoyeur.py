from tkinter import ttk, messagebox
import RPi.GPIO.Emulator as GPIO
import tkinter as tk
import time
import threading
import datetime
import csv
import os

class SystemeDetectionIntegre:
    def __init__(self, root):
        # --- CONFIGURATION PINS ---
        # Capteur ultrason
        self.TRIG = 6             # Capteur ultrasons TRIG
        self.ECHO = 27            # Capteur ultrasons ECHO
        # Servomoteur
        self.SERVO_PIN = 21       # Servomoteur 1 sur GPIO21
        self.SERVO2_PIN = 20      # Servomoteur 2 sur GPIO20
        # Moteur DC
        self.IN1 = 26             # Direction moteur 1
        self.IN2 = 16             # Direction moteur 2  
        self.ENA = 24             # Enable moteur (PWM)
        # Capteur NPN
        self.CAPTEUR_NPN = 17     # Capteur NPN inductif
        
        # --- PARAMETRES SYSTEME ---
        self.SEUIL_CM = 50        # Seuil de détection par défaut
        self.HISTORIQUE_MAX = 15  # Nombre d'entrées d'historique à afficher
        self.FREQ_MOTEUR = 1000   # Fréquence PWM moteur
        self.FREQ_SERVO = 50      # Fréquence PWM servo
        
        # --- VARIABLES D'ÉTAT ---
        self.detection_active = False
        self.dernier_etat_ultrason = False
        self.dernier_etat_npn = True  # Initialisé à True (état par défaut NPN avec pull-up)
        self.thread_detection = None
        self.stop_thread = False
        self.log_fichier = "detection_log_integre.csv"
        self.historique = []
        
        # Variables moteur
        self.vitesse_dc = 60
        self.motor_direction = "STOP"
        
        # Variables servo - LOGIQUE FINALE CORRECTE
        self.angle_ouvert = 100   # Position ouverte servo 1 (position initiale)
        self.angle_ferme = 10     # Position fermée servo 1 (fermeture = autre sens)
        self.servo_etat = "ouvert"  # État initial : ouvert à 100°
        
        # Variables servo 2
        self.angle_ouvert2 = 0    # Position ouverte servo 2
        self.angle_ferme2 = 50    # Position fermée servo 2
        self.servo2_etat = "fermé"  # État initial : fermé
        
        # VARIABLES POUR LE CONVOYEUR
        self.mode_convoyeur = False           # Mode convoyeur activé/désactivé
        self.convoyeur_en_marche = False      # État du convoyeur
        self.detection_en_cours = False       # Un objet est en cours de traitement
        
        # TEMPORISATIONS DU CYCLE
        self.tempo_continue_tapis = 3.0       # 3s de continuation du tapis après détection ultrason
        self.tempo_servo1_ferme = 20.0        # 20s de maintien servo 1 fermé
        self.tempo_arret_npn = 3.0           # 3s d'arrêt tapis lors détection NPN
        self.tempo_servo2_ouvert = 10.0       # 10s de maintien servo 2 ouvert
        self.tempo_blocage_npn = 15.0        # 15s de blocage NPN après détection (anti-redétection)
        
        # Variables de gestion des timers
        self.timer_continue_tapis = None
        self.timer_servo1_ferme = None
        self.timer_arret_npn = None
        self.timer_servo2_ouvert = None
        self.timer_blocage_npn = None
        
        # Variables de blocage anti-redétection
        self.npn_bloque = False              # Blocage temporaire du NPN
        self.derniere_detection_npn = 0      # Timestamp de la dernière détection NPN
        
        # Variables PWM pour éviter les références perdues
        self.pwm_servo = None
        self.pwm_servo2 = None
        self.pwm_motor = None
        
        # --- INITIALISATION GPIO ---
        self.initialiser_gpio()
        
        # --- INTERFACE UTILISATEUR ---
        self.root = root
        self.setup_interface()
        
        # --- INITIALISATION ---
        self.bouger_servo(self.angle_ouvert)   # Position initiale servo 1 (OUVERTE à 100°)
        self.bouger_servo2(self.angle_ferme2)  # Position initiale servo 2 (fermée)
        self.creer_fichier_log()
        
    # ===============================
    # INITIALISATION GPIO
    # ===============================
    
    def initialiser_gpio(self):
        """Initialise tous les GPIO de manière sécurisée"""
        try:
            # Nettoyer d'abord au cas où
            GPIO.cleanup()
            
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            
            # Configuration ultrason
            GPIO.setup(self.TRIG, GPIO.OUT)
            GPIO.output(self.TRIG, False)
            GPIO.setup(self.ECHO, GPIO.IN)
            
            # Configuration servomoteur 1
            GPIO.setup(self.SERVO_PIN, GPIO.OUT)
            self.pwm_servo = GPIO.PWM(self.SERVO_PIN, self.FREQ_SERVO)
            self.pwm_servo.start(0)
            
            # Configuration servomoteur 2
            GPIO.setup(self.SERVO2_PIN, GPIO.OUT)
            self.pwm_servo2 = GPIO.PWM(self.SERVO2_PIN, self.FREQ_SERVO)
            self.pwm_servo2.start(0)
            
            # Configuration moteur DC
            GPIO.setup(self.IN1, GPIO.OUT)
            GPIO.setup(self.IN2, GPIO.OUT)
            GPIO.setup(self.ENA, GPIO.OUT)
            self.pwm_motor = GPIO.PWM(self.ENA, self.FREQ_MOTEUR)
            self.pwm_motor.start(0)
            
            # Configuration capteur NPN
            GPIO.setup(self.CAPTEUR_NPN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            
            print("GPIO initialisés avec succès")
            
        except Exception as e:
            print(f"Erreur lors de l'initialisation GPIO: {e}")
            raise
    
    # ===============================
    # INTERFACE UTILISATEUR
    # ===============================
    
    def setup_interface(self):
        """Configure l'interface utilisateur complète"""
        self.root.title("🧠 Système de Convoyeur Intelligent")
        self.root.geometry("800x700")
        self.root.minsize(700, 600)
        self.root.configure(bg="#2c3e50")

        # --- Structure scrollable ---
        self.canvas_frame = tk.Frame(self.root, bg="#2c3e50")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, bg="#2c3e50")
        self.scrollbar = ttk.Scrollbar(self.canvas_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # --- Scroll avec molette ---
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # --- Style ---
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TButton", font=("Arial", 11), padding=6)
        style.configure("TLabel", font=("Arial", 11))

        # --- Titre principal ---
        title = tk.Label(self.scrollable_frame, text="🏭 Système de Convoyeur Intelligent", 
                        font=("Arial", 18, "bold"), fg="white", bg="#2c3e50")
        title.pack(pady=15)

        # --- Onglets ---
        self.notebook = ttk.Notebook(self.scrollable_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # === ONGLET PRINCIPAL ===
        self.tab_main = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_main, text="🏠 Principal")

        # État du système
        frame_status = ttk.LabelFrame(self.tab_main, text="📊 État du Système")
        frame_status.pack(fill=tk.X, padx=10, pady=5)

        self.label_etat = ttk.Label(frame_status, text="Système prêt", font=("Helvetica", 12))
        self.label_etat.pack(pady=5)

        self.label_distance = ttk.Label(frame_status, text="Distance: -- cm", font=("Helvetica", 12))
        self.label_distance.pack(pady=5)

        self.label_npn = ttk.Label(frame_status, text="Capteur NPN: --", font=("Helvetica", 12))
        self.label_npn.pack(pady=5)

        # Indicateurs visuels
        indicators_frame = tk.Frame(frame_status)
        indicators_frame.pack(pady=5)

        tk.Label(indicators_frame, text="Ultrason:").pack(side=tk.LEFT, padx=5)
        self.canvas_ultrason = tk.Canvas(indicators_frame, width=30, height=30, bg="lightgray")
        self.indicator_ultrason = self.canvas_ultrason.create_oval(5, 5, 25, 25, fill="gray")
        self.canvas_ultrason.pack(side=tk.LEFT, padx=5)

        tk.Label(indicators_frame, text="NPN:").pack(side=tk.LEFT, padx=5)
        self.canvas_npn = tk.Canvas(indicators_frame, width=30, height=30, bg="lightgray")
        self.indicator_npn = self.canvas_npn.create_oval(5, 5, 25, 25, fill="gray")
        self.canvas_npn.pack(side=tk.LEFT, padx=5)

        # État des servos
        servo_frame = tk.Frame(frame_status)
        servo_frame.pack(pady=5)

        tk.Label(servo_frame, text="Servo 1:").pack(side=tk.LEFT, padx=5)
        self.label_servo1 = tk.Label(servo_frame, text="Ouvert", bg="lightgreen", width=8)
        self.label_servo1.pack(side=tk.LEFT, padx=2)

        tk.Label(servo_frame, text="Servo 2:").pack(side=tk.LEFT, padx=5)
        self.label_servo2 = tk.Label(servo_frame, text="Fermé", bg="lightcoral", width=8)
        self.label_servo2.pack(side=tk.LEFT, padx=2)

        # Contrôles principaux
        frame_control = ttk.LabelFrame(self.tab_main, text="⚡ Contrôles Principaux")
        frame_control.pack(fill=tk.X, padx=10, pady=10)

        btn_frame = ttk.Frame(frame_control)
        btn_frame.pack(pady=5)

        self.btn_start = ttk.Button(btn_frame, text="▶️ Démarrer Système", command=self.demarrer_detection)
        self.btn_start.pack(side=tk.LEFT, padx=5)

        self.btn_stop = ttk.Button(btn_frame, text="⏹️ Arrêter Système", command=self.arreter_detection, state="disabled")
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        self.btn_mode_convoyeur = ttk.Button(btn_frame, text="🏭 Mode Convoyeur", command=self.toggle_mode_convoyeur)
        self.btn_mode_convoyeur.pack(side=tk.LEFT, padx=5)

        # Tests
        test_frame = ttk.Frame(frame_control)
        test_frame.pack(pady=5)
        
        ttk.Button(test_frame, text="🔧 Test Servo 1", command=self.test_servo).pack(side=tk.LEFT, padx=5)
        ttk.Button(test_frame, text="🔧 Test Servo 2", command=self.test_servo2).pack(side=tk.LEFT, padx=5)
        
        # === ONGLET MOTEUR ===
        self.tab_motor = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_motor, text="🚗 Moteur DC")

        motor_frame = ttk.LabelFrame(self.tab_motor, text="🔧 Contrôle Moteur DC")
        motor_frame.pack(fill=tk.X, padx=10, pady=10)

        # Vitesse moteur
        tk.Label(motor_frame, text="🚀 Vitesse moteur DC (%)").pack(pady=5)
        self.vitesse_var = tk.IntVar(value=self.vitesse_dc)
        self.scale_vitesse = ttk.Scale(motor_frame, from_=0, to=100, variable=self.vitesse_var,
                                      orient="horizontal", command=self.update_speed)
        self.scale_vitesse.pack(fill=tk.X, padx=10, pady=5)
        
        self.lbl_vitesse_value = ttk.Label(motor_frame, text=f"{self.vitesse_dc}%")
        self.lbl_vitesse_value.pack(pady=2)

        # Direction moteur
        direction_frame = ttk.Frame(motor_frame)
        direction_frame.pack(pady=10)

        ttk.Button(direction_frame, text="⬆️ AVANT", command=lambda: self.set_direction("AVANT")).pack(side=tk.LEFT, padx=5)
        ttk.Button(direction_frame, text="⬇️ ARRIÈRE", command=lambda: self.set_direction("ARRIERE")).pack(side=tk.LEFT, padx=5)
        ttk.Button(direction_frame, text="⏹️ STOP", command=lambda: self.set_direction("STOP")).pack(side=tk.LEFT, padx=5)

        # === ONGLET CONFIGURATION ===
        self.tab_config = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_config, text="⚙️ Configuration")

        # Configuration détection
        frame_detection_config = ttk.LabelFrame(self.tab_config, text="📡 Configuration Détection")
        frame_detection_config.pack(fill=tk.X, padx=10, pady=5)

        seuil_frame = ttk.Frame(frame_detection_config)
        seuil_frame.pack(fill=tk.X, pady=5)

        ttk.Label(seuil_frame, text="Seuil de détection (cm):").pack(side=tk.LEFT, padx=5)
        self.seuil_var = tk.IntVar(value=self.SEUIL_CM)
        self.scale_seuil = ttk.Scale(seuil_frame, from_=10, to=200, variable=self.seuil_var,
                                   orient="horizontal", command=self.update_seuil_label)
        self.scale_seuil.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.lbl_seuil_value = ttk.Label(seuil_frame, text=str(self.SEUIL_CM))
        self.lbl_seuil_value.pack(side=tk.LEFT, padx=5)

        # Configuration temporisations
        frame_tempo_config = ttk.LabelFrame(self.tab_config, text="⏱️ Temporisations du Cycle")
        frame_tempo_config.pack(fill=tk.X, padx=10, pady=5)

        # Tempo continuation tapis (ultrason)
        tempo_continue_frame = ttk.Frame(frame_tempo_config)
        tempo_continue_frame.pack(fill=tk.X, pady=2)
        ttk.Label(tempo_continue_frame, text="Continue tapis ultrason (sec):").pack(side=tk.LEFT, padx=5)
        self.tempo_continue_var = tk.DoubleVar(value=self.tempo_continue_tapis)
        self.scale_tempo_continue = ttk.Scale(tempo_continue_frame, from_=1.0, to=10.0, variable=self.tempo_continue_var, 
                                            orient="horizontal", command=self.update_tempo_continue_label)
        self.scale_tempo_continue.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.lbl_tempo_continue_value = ttk.Label(tempo_continue_frame, text=f"{self.tempo_continue_tapis:.1f}s")
        self.lbl_tempo_continue_value.pack(side=tk.LEFT, padx=5)

        # Tempo servo 1 fermé
        tempo_servo1_frame = ttk.Frame(frame_tempo_config)
        tempo_servo1_frame.pack(fill=tk.X, pady=2)
        ttk.Label(tempo_servo1_frame, text="Servo 1 fermé (sec):").pack(side=tk.LEFT, padx=5)
        self.tempo_servo1_var = tk.DoubleVar(value=self.tempo_servo1_ferme)
        self.scale_tempo_servo1 = ttk.Scale(tempo_servo1_frame, from_=5.0, to=30.0, variable=self.tempo_servo1_var, 
                                          orient="horizontal", command=self.update_tempo_servo1_label)
        self.scale_tempo_servo1.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.lbl_tempo_servo1_value = ttk.Label(tempo_servo1_frame, text=f"{self.tempo_servo1_ferme:.1f}s")
        self.lbl_tempo_servo1_value.pack(side=tk.LEFT, padx=5)

        # Tempo arrêt NPN
        tempo_npn_frame = ttk.Frame(frame_tempo_config)
        tempo_npn_frame.pack(fill=tk.X, pady=2)
        ttk.Label(tempo_npn_frame, text="Arrêt tapis NPN (sec):").pack(side=tk.LEFT, padx=5)
        self.tempo_npn_var = tk.DoubleVar(value=self.tempo_arret_npn)
        self.scale_tempo_npn = ttk.Scale(tempo_npn_frame, from_=1.0, to=10.0, variable=self.tempo_npn_var, 
                                       orient="horizontal", command=self.update_tempo_npn_label)
        self.scale_tempo_npn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.lbl_tempo_npn_value = ttk.Label(tempo_npn_frame, text=f"{self.tempo_arret_npn:.1f}s")
        self.lbl_tempo_npn_value.pack(side=tk.LEFT, padx=5)

        # Tempo servo 2 ouvert
        tempo_servo2_frame = ttk.Frame(frame_tempo_config)
        tempo_servo2_frame.pack(fill=tk.X, pady=2)
        ttk.Label(tempo_servo2_frame, text="Servo 2 ouvert (sec):").pack(side=tk.LEFT, padx=5)
        self.tempo_servo2_var = tk.DoubleVar(value=self.tempo_servo2_ouvert)
        self.scale_tempo_servo2 = ttk.Scale(tempo_servo2_frame, from_=5.0, to=20.0, variable=self.tempo_servo2_var, 
                                          orient="horizontal", command=self.update_tempo_servo2_label)
        self.scale_tempo_servo2.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.lbl_tempo_servo2_value = ttk.Label(tempo_servo2_frame, text=f"{self.tempo_servo2_ouvert:.1f}s")
        self.lbl_tempo_servo2_value.pack(side=tk.LEFT, padx=5)

        # Tempo blocage NPN (NOUVEAU)
        tempo_blocage_frame = ttk.Frame(frame_tempo_config)
        tempo_blocage_frame.pack(fill=tk.X, pady=2)
        ttk.Label(tempo_blocage_frame, text="Blocage NPN anti-redétection (sec):").pack(side=tk.LEFT, padx=5)
        self.tempo_blocage_var = tk.DoubleVar(value=self.tempo_blocage_npn)
        self.scale_tempo_blocage = ttk.Scale(tempo_blocage_frame, from_=10.0, to=30.0, variable=self.tempo_blocage_var, 
                                           orient="horizontal", command=self.update_tempo_blocage_label)
        self.scale_tempo_blocage.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.lbl_tempo_blocage_value = ttk.Label(tempo_blocage_frame, text=f"{self.tempo_blocage_npn:.1f}s")
        self.lbl_tempo_blocage_value.pack(side=tk.LEFT, padx=5)

        # Configuration servo
        frame_servo_config = ttk.LabelFrame(self.tab_config, text="🔄 Configuration Servomoteurs")
        frame_servo_config.pack(fill=tk.X, padx=10, pady=5)

        servo_config_frame = ttk.Frame(frame_servo_config)
        servo_config_frame.pack(fill=tk.X, pady=5)

        # Servo 1
        servo1_frame = ttk.LabelFrame(servo_config_frame, text="Servo 1")
        servo1_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        ttk.Label(servo1_frame, text="Angle ouvert (°):").grid(row=0, column=0, padx=5, pady=2)
        self.angle_ouvert_var = tk.IntVar(value=self.angle_ouvert)
        ttk.Spinbox(servo1_frame, from_=0, to=180, textvariable=self.angle_ouvert_var, width=5).grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(servo1_frame, text="Angle fermé (°):").grid(row=1, column=0, padx=5, pady=2)
        self.angle_ferme_var = tk.IntVar(value=self.angle_ferme)
        ttk.Spinbox(servo1_frame, from_=0, to=180, textvariable=self.angle_ferme_var, width=5).grid(row=1, column=1, padx=5, pady=2)

        # Servo 2
        servo2_frame = ttk.LabelFrame(servo_config_frame, text="Servo 2")
        servo2_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        ttk.Label(servo2_frame, text="Angle ouvert (°):").grid(row=0, column=0, padx=5, pady=2)
        self.angle_ouvert2_var = tk.IntVar(value=self.angle_ouvert2)
        ttk.Spinbox(servo2_frame, from_=0, to=180, textvariable=self.angle_ouvert2_var, width=5).grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(servo2_frame, text="Angle fermé (°):").grid(row=1, column=0, padx=5, pady=2)
        self.angle_ferme2_var = tk.IntVar(value=self.angle_ferme2)
        ttk.Spinbox(servo2_frame, from_=0, to=180, textvariable=self.angle_ferme2_var, width=5).grid(row=1, column=1, padx=5, pady=2)

        ttk.Button(frame_servo_config, text="✅ Appliquer Configuration", command=self.appliquer_config).pack(pady=10)

        # === ONGLET HISTORIQUE ===
        self.tab_historique = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_historique, text="📚 Historique")

        frame_historique = ttk.LabelFrame(self.tab_historique, text="📋 Événements du Système")
        frame_historique.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        columns = ("timestamp", "event", "capteur", "valeur")
        self.tree = ttk.Treeview(frame_historique, columns=columns, show="headings")
        self.tree.heading("timestamp", text="Horodatage")
        self.tree.heading("event", text="Événement")
        self.tree.heading("capteur", text="Capteur")
        self.tree.heading("valeur", text="Valeur")

        self.tree.column("timestamp", width=150)
        self.tree.column("event", width=200)
        self.tree.column("capteur", width=100)
        self.tree.column("valeur", width=100)

        self.tree.pack(fill=tk.BOTH, expand=True, pady=5)

        ttk.Button(frame_historique, text="🗑️ Effacer historique", command=self.effacer_historique).pack(pady=5)

        # --- Barre de statut ---
        self.statusbar = ttk.Label(self.root, text="Prêt - Mode Normal", relief="sunken", anchor="w")
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)

    # ===============================
    # FONCTIONS DE MISE À JOUR INTERFACE
    # ===============================
    
    def update_tempo_continue_label(self, *args):
        """Met à jour l'affichage du délai de continuation du tapis"""
        tempo = float(self.tempo_continue_var.get())
        self.lbl_tempo_continue_value.config(text=f"{tempo:.1f}s")
        self.tempo_continue_tapis = tempo

    def update_tempo_servo1_label(self, *args):
        """Met à jour l'affichage du délai servo 1 fermé"""
        tempo = float(self.tempo_servo1_var.get())
        self.lbl_tempo_servo1_value.config(text=f"{tempo:.1f}s")
        self.tempo_servo1_ferme = tempo

    def update_tempo_npn_label(self, *args):
        """Met à jour l'affichage du délai arrêt NPN"""
        tempo = float(self.tempo_npn_var.get())
        self.lbl_tempo_npn_value.config(text=f"{tempo:.1f}s")
        self.tempo_arret_npn = tempo

    def update_tempo_servo2_label(self, *args):
        """Met à jour l'affichage du délai servo 2 ouvert"""
        tempo = float(self.tempo_servo2_var.get())
        self.lbl_tempo_servo2_value.config(text=f"{tempo:.1f}s")
        self.tempo_servo2_ouvert = tempo

    def update_tempo_blocage_label(self, *args):
        """Met à jour l'affichage du délai de blocage NPN"""
        tempo = float(self.tempo_blocage_var.get())
        self.lbl_tempo_blocage_value.config(text=f"{tempo:.1f}s")
        self.tempo_blocage_npn = tempo
        
    def update_seuil_label(self, *args):
        """Met à jour l'affichage de la valeur du seuil"""
        self.lbl_seuil_value.config(text=str(int(float(self.seuil_var.get()))))
        self.SEUIL_CM = int(float(self.seuil_var.get()))

    def update_speed(self, *args):
        """Met à jour la vitesse du moteur"""
        vitesse = int(float(self.vitesse_var.get()))
        self.lbl_vitesse_value.config(text=f"{vitesse}%")
        self.vitesse_dc = vitesse
        if self.motor_direction != "STOP" and self.pwm_motor:
            self.pwm_motor.ChangeDutyCycle(vitesse)

    def update_servo_status(self):
        """Met à jour l'affichage des servos"""
        if self.servo_etat == "ouvert":
            self.label_servo1.config(text="Ouvert", bg="lightgreen")
        else:
            self.label_servo1.config(text="Fermé", bg="lightcoral")
            
        if self.servo2_etat == "ouvert":
            self.label_servo2.config(text="Ouvert", bg="lightgreen")
        else:
            self.label_servo2.config(text="Fermé", bg="lightcoral")

    # ===============================
    # FONCTIONS MOTEUR
    # ===============================

    def set_direction(self, direction):
        """Change la direction du moteur"""
        self.motor_direction = direction
        self.statusbar.config(text=f"🔁 Direction moteur : {direction}")
        self.start_motor()
        self.ajouter_evenement(f"Moteur {direction}", "Moteur", f"{self.vitesse_dc}%")

    def start_motor(self):
        """Démarre le moteur dans la direction spécifiée"""
        if not self.pwm_motor:
            print("Erreur: PWM moteur non initialisé")
            return
            
        speed = self.vitesse_dc
        try:
            if self.motor_direction == "AVANT":
                GPIO.output(self.IN1, GPIO.HIGH)
                GPIO.output(self.IN2, GPIO.LOW)
                self.pwm_motor.ChangeDutyCycle(speed)
            elif self.motor_direction == "ARRIERE":
                GPIO.output(self.IN1, GPIO.LOW)
                GPIO.output(self.IN2, GPIO.HIGH)
                self.pwm_motor.ChangeDutyCycle(speed)
            else:  # STOP
                GPIO.output(self.IN1, GPIO.LOW)
                GPIO.output(self.IN2, GPIO.LOW)
                self.pwm_motor.ChangeDutyCycle(0)
        except Exception as e:
            print(f"Erreur contrôle moteur: {e}")

    def demarrer_convoyeur(self):
        """Démarre le convoyeur en marche avant"""
        if self.mode_convoyeur:
            self.convoyeur_en_marche = True
            self.motor_direction = "AVANT"
            self.start_motor()
            self.root.after(0, lambda: self.statusbar.config(text="🏭 Convoyeur en marche"))

    def arreter_convoyeur(self):
        """Arrête le convoyeur"""
        self.convoyeur_en_marche = False
        self.motor_direction = "STOP"
        self.start_motor()

    # ===============================
    # FONCTIONS SERVOMOTEURS
    # ===============================

    def bouger_servo(self, angle):
        """Bouge le servomoteur 1 à l'angle spécifié"""
        if not self.pwm_servo:
            print("Erreur: PWM servo non initialisé")
            return False
            
        try:
            if angle < 0:
                angle = 0
            elif angle > 180:
                angle = 180
                
            duty = 2.5 + (angle / 18.0)
            self.pwm_servo.ChangeDutyCycle(duty)
            time.sleep(0.5)
            self.pwm_servo.ChangeDutyCycle(0)
            
            # Mettre à jour l'état
            if angle == self.angle_ouvert:
                self.servo_etat = "ouvert"
            else:
                self.servo_etat = "fermé"
            
            self.root.after(0, self.update_servo_status)
            return True
        except Exception as e:
            print(f"Erreur servomoteur 1: {e}")
            return False
        
    def bouger_servo2(self, angle):
        """Bouge le servomoteur 2 à l'angle spécifié"""
        if not self.pwm_servo2:
            print("Erreur: PWM servo 2 non initialisé")
            return False
            
        try:
            if angle < 0:
                angle = 0
            elif angle > 180:
                angle = 180
                
            duty = 2.5 + (angle / 18.0)
            self.pwm_servo2.ChangeDutyCycle(duty)
            time.sleep(0.5)
            self.pwm_servo2.ChangeDutyCycle(0)
            
            # Mettre à jour l'état
            if angle == self.angle_ouvert2:
                self.servo2_etat = "ouvert"
            else:
                self.servo2_etat = "fermé"
            
            self.root.after(0, self.update_servo_status)
            return True
        except Exception as e:
            print(f"Erreur servomoteur 2: {e}")
            return False

    # ===============================
    # FONCTIONS CAPTEURS
    # ===============================

    def mesurer_distance(self):
        """Mesure la distance avec le capteur ultrason HC-SR04"""
        try:
            # Vérifier que les GPIO sont toujours disponibles
            GPIO.output(self.TRIG, True)
            time.sleep(0.00001)
            GPIO.output(self.TRIG, False)
            
            pulse_start = time.time()
            timeout_start = time.time()
            
            while GPIO.input(self.ECHO) == 0:
                pulse_start = time.time()
                if time.time() - timeout_start > 0.1:
                    return float('inf')
            
            pulse_end = time.time()
            timeout_start = time.time()
            
            while GPIO.input(self.ECHO) == 1:
                pulse_end = time.time()
                if time.time() - timeout_start > 0.1:
                    return float('inf')
            
            pulse_duration = pulse_end - pulse_start
            distance = round(pulse_duration * 17150, 1)
            
            return distance
        except Exception as e:
            print(f"Erreur capteur ultrason: {e}")
            return float('inf')

    def lire_capteur_npn(self):
        """Lit l'état du capteur NPN"""
        try:
            return GPIO.input(self.CAPTEUR_NPN)
        except Exception as e:
            print(f"Erreur capteur NPN: {e}")
            return None

    # ===============================
    # LOGIQUE PRINCIPALE DU CYCLE CONVOYEUR
    # ===============================

    def traiter_detection_ultrason(self, distance):
        """Traite la détection ultrason avec le cycle final complet"""
        if not self.mode_convoyeur:
            # Mode normal (ancien comportement)
            if distance < self.SEUIL_CM and distance != float('inf'):
                self.root.after(0, lambda: self.label_etat.config(text="Présence détectée ✓"))
                self.root.after(0, lambda: self.canvas_ultrason.itemconfig(self.indicator_ultrason, fill="green"))
                
                if not self.dernier_etat_ultrason:
                    self.root.after(0, lambda: self.set_direction("STOP"))
                    if self.bouger_servo(self.angle_ferme):  # Ferme vers 10°
                        self.dernier_etat_ultrason = True
                        self.root.after(0, lambda: self.ajouter_evenement("Présence détectée - Servo 1 fermé", "Ultrason", f"{distance:.1f} cm"))
            else:
                self.root.after(0, lambda: self.label_etat.config(text="Aucune présence"))
                self.root.after(0, lambda: self.canvas_ultrason.itemconfig(self.indicator_ultrason, fill="red"))
                
                if self.dernier_etat_ultrason:
                    if self.bouger_servo(self.angle_ouvert):  # Rouvre vers 100°
                        self.dernier_etat_ultrason = False
                        self.root.after(0, lambda: self.ajouter_evenement("Fin de présence - Servo 1 ouvert", "Ultrason", f"{distance:.1f} cm"))
            return
        
        # ====== CYCLE CONVOYEUR - DÉTECTION ULTRASON ======
        if distance < self.SEUIL_CM and distance != float('inf'):
            self.root.after(0, lambda: self.canvas_ultrason.itemconfig(self.indicator_ultrason, fill="green"))
            
            if not self.dernier_etat_ultrason and self.convoyeur_en_marche and not self.detection_en_cours:
                self.detection_en_cours = True
                
                self.root.after(0, lambda: self.label_etat.config(text=f"🔍 Objet détecté - Tapis continue {self.tempo_continue_tapis}s"))
                self.root.after(0, lambda: self.ajouter_evenement("CYCLE: Objet détecté", "Ultrason", f"{distance:.1f} cm"))
                
                # PHASE 1: TAPIS CONTINUE PENDANT 3s
                self.timer_continue_tapis = threading.Timer(self.tempo_continue_tapis, self.phase_arret_et_fermeture_servo1)
                self.timer_continue_tapis.start()
                
                self.dernier_etat_ultrason = True
        else:
            self.root.after(0, lambda: self.canvas_ultrason.itemconfig(self.indicator_ultrason, fill="red"))
            if self.dernier_etat_ultrason and not self.detection_en_cours:
                self.dernier_etat_ultrason = False

    def phase_arret_et_fermeture_servo1(self):
        """PHASE 1: Arrêt tapis + Fermeture servo 1 + Redémarrage immédiat"""
        if not self.mode_convoyeur or not self.detection_active:
            return
            
        # ÉTAPE 1: ARRÊTER LE TAPIS
        self.arreter_convoyeur()
        self.root.after(0, lambda: self.label_etat.config(text="⏹️ Tapis arrêté - Fermeture servo 1"))
        self.root.after(0, lambda: self.ajouter_evenement("CYCLE: Tapis arrêté", "Système", "3s écoulées"))
        
        # ÉTAPE 2: FERMER LE SERVO 1 (va de 100° vers 10° = fermeture)
        if self.bouger_servo(self.angle_ferme):
            self.root.after(0, lambda: self.ajouter_evenement("CYCLE: Servo 1 fermé (100°→10°)", "Servo1", f"{self.tempo_servo1_ferme}s"))
            
            # ÉTAPE 3: REDÉMARRER LE TAPIS IMMÉDIATEMENT (servo 1 reste fermé)
            self.demarrer_convoyeur()
            self.root.after(0, lambda: self.label_etat.config(text=f"🏭 Tapis redémarré - Servo 1 fermé {self.tempo_servo1_ferme}s"))
            self.root.after(0, lambda: self.ajouter_evenement("CYCLE: Tapis redémarré (servo 1 fermé)", "Système", "Auto"))
            
            # ÉTAPE 4: PROGRAMMER LA RÉOUVERTURE DU SERVO 1 APRÈS 20s
            self.timer_servo1_ferme = threading.Timer(self.tempo_servo1_ferme, self.phase_reouverture_servo1)
            self.timer_servo1_ferme.start()

    def phase_reouverture_servo1(self):
        """PHASE 2: Réouverture servo 1 après 20s (tapis continue) - va de 10° vers 100°"""
        if self.mode_convoyeur and self.detection_active:
            if self.bouger_servo(self.angle_ouvert):
                self.root.after(0, lambda: self.label_etat.config(text="✅ Servo 1 rouvert (10°→100°) - Cycle normal"))
                self.root.after(0, lambda: self.ajouter_evenement("CYCLE: Servo 1 rouvert (10°→100°)", "Servo1", "20s écoulées"))
        
        # Fin du cycle ultrason
        self.detection_en_cours = False

    def traiter_detection_npn(self, etat_npn):
        """Traite la détection du capteur NPN avec anti-redétection"""
        if not self.mode_convoyeur:
            # Mode normal (ancien comportement)
            if etat_npn != self.dernier_etat_npn:
                if etat_npn == 0:  # Signal bas = détection métal (NPN)
                    self.root.after(0, lambda: self.set_direction("STOP"))
                    if self.bouger_servo2(self.angle_ouvert2):  # Ouvre servo 2
                        self.root.after(0, lambda: self.ajouter_evenement("Métal détecté - Servo 2 ouvert", "NPN", "Actif"))
                else:
                    if self.bouger_servo2(self.angle_ferme2):  # Ferme servo 2
                        self.root.after(0, lambda: self.ajouter_evenement("Fin détection métal - Servo 2 fermé", "NPN", "Inactif"))
                
                self.dernier_etat_npn = etat_npn
            return
        
        # ====== CYCLE CONVOYEUR - DÉTECTION NPN AVEC ANTI-REDÉTECTION ======
        if etat_npn != self.dernier_etat_npn:
            if etat_npn == 0:  # Métal détecté
                # VÉRIFIER SI NPN N'EST PAS BLOQUÉ
                if not self.npn_bloque and self.convoyeur_en_marche:
                    # BLOQUER LE NPN POUR ÉVITER LES REDÉTECTIONS
                    self.npn_bloque = True
                    self.derniere_detection_npn = time.time()
                    
                    self.root.after(0, lambda: self.label_etat.config(text="🔍 Métal détecté - Séquence NPN (bloqué)"))
                    self.root.after(0, lambda: self.ajouter_evenement("CYCLE: Métal détecté (NPN bloqué)", "NPN", f"Blocage {self.tempo_blocage_npn}s"))
                    
                    # LANCER LA SÉQUENCE NPN COMPLÈTE
                    threading.Timer(0.1, self.sequence_npn_complete).start()
                    
                    # PROGRAMMER LE DÉBLOCAGE DU NPN
                    self.timer_blocage_npn = threading.Timer(self.tempo_blocage_npn, self.debloquer_npn)
                    self.timer_blocage_npn.start()
                    
                elif self.npn_bloque:
                    # NPN bloqué - ignorer la détection
                    temps_restant = self.tempo_blocage_npn - (time.time() - self.derniere_detection_npn)
                    if temps_restant > 0:
                        self.root.after(0, lambda: self.ajouter_evenement("NPN bloqué - Détection ignorée", "NPN", f"Reste {temps_restant:.1f}s"))
            
            self.dernier_etat_npn = etat_npn

    def debloquer_npn(self):
        """Débloque le capteur NPN après la temporisation"""
        self.npn_bloque = False
        self.root.after(0, lambda: self.ajouter_evenement("CYCLE: NPN débloqué - Prêt nouvelle détection", "NPN", "Déblocage"))
        print("NPN débloqué - Prêt pour nouvelle détection")

    def sequence_npn_complete(self):
        """Séquence NPN complète: Arrêt 3s + Servo 2 ouvre + Tapis redémarre + Servo 2 ferme après 10s"""
        if not self.mode_convoyeur or not self.detection_active:
            return
        
        # PHASE 1: ARRÊT TAPIS 3s
        self.arreter_convoyeur()
        self.root.after(0, lambda: self.label_etat.config(text=f"⏹️ Métal - Arrêt tapis {self.tempo_arret_npn}s"))
        self.root.after(0, lambda: self.ajouter_evenement("CYCLE: Arrêt tapis (métal)", "NPN", f"{self.tempo_arret_npn}s"))
        
        # PHASE 2: OUVRIR SERVO 2 IMMÉDIATEMENT
        if self.bouger_servo2(self.angle_ouvert2):
            self.root.after(0, lambda: self.ajouter_evenement("CYCLE: Servo 2 ouvert (métal)", "Servo2", f"{self.tempo_servo2_ouvert}s"))
        
        # PHASE 3: PROGRAMMER LE REDÉMARRAGE APRÈS 3s
        self.timer_arret_npn = threading.Timer(self.tempo_arret_npn, self.redemarrer_tapis_apres_npn)
        self.timer_arret_npn.start()
        
        # PHASE 4: PROGRAMMER LA FERMETURE SERVO 2 APRÈS 10s TOTAL
        self.timer_servo2_ouvert = threading.Timer(self.tempo_servo2_ouvert, self.fermer_servo2_final)
        self.timer_servo2_ouvert.start()

    def redemarrer_tapis_apres_npn(self):
        """Redémarre le tapis après 3s d'arrêt (servo 2 reste ouvert)"""
        if self.mode_convoyeur and self.detection_active:
            self.demarrer_convoyeur()
            self.root.after(0, lambda: self.label_etat.config(text="🏭 Tapis redémarré - Servo 2 ouvert"))
            self.root.after(0, lambda: self.ajouter_evenement("CYCLE: Tapis redémarré (servo 2 ouvert)", "NPN", "Auto"))

    def fermer_servo2_final(self):
        """Ferme le servo 2 après 10s total depuis la détection"""
        if self.bouger_servo2(self.angle_ferme2):
            self.root.after(0, lambda: self.label_etat.config(text="✅ Servo 2 fermé - Cycle normal"))
            self.root.after(0, lambda: self.ajouter_evenement("CYCLE: Servo 2 fermé - Fin séquence", "Servo2", "10s écoulées"))

    # ===============================
    # GESTION DES TIMERS
    # ===============================

    def annuler_timers(self):
        """Annule tous les timers en cours"""
        timers = [
            ('timer_continue_tapis', self.timer_continue_tapis),
            ('timer_servo1_ferme', self.timer_servo1_ferme),
            ('timer_arret_npn', self.timer_arret_npn),
            ('timer_servo2_ouvert', self.timer_servo2_ouvert),
            ('timer_blocage_npn', self.timer_blocage_npn)
        ]
        
        for nom, timer in timers:
            if timer:
                timer.cancel()
                setattr(self, nom, None)
                print(f"Timer {nom} annulé")

    # ===============================
    # THREAD DE DÉTECTION PRINCIPAL
    # ===============================

    def detection_presence(self):
        """Fonction de détection exécutée dans un thread séparé"""
        self.statusbar.config(text="Détection en cours...")
        
        while not self.stop_thread:
            try:
                # Mesurer la distance avec ultrason
                distance = self.mesurer_distance()
                
                # Lire capteur NPN
                etat_npn = self.lire_capteur_npn()
                
                # Mettre à jour l'interface (thread-safe)
                self.root.after(0, self.update_ui, distance, etat_npn)
                
                # Traitement ultrason
                self.traiter_detection_ultrason(distance)
                
                # Traitement capteur NPN
                if etat_npn is not None:
                    self.traiter_detection_npn(etat_npn)
                
                time.sleep(0.5)
                
            except Exception as e:
                self.root.after(0, lambda: self.label_etat.config(text=f"Erreur: {str(e)}"))
                print(f"Erreur dans detection_presence: {e}")
                time.sleep(1)
        
        self.root.after(0, lambda: self.statusbar.config(text="Détection arrêtée"))

    def update_ui(self, distance, etat_npn):
        """Met à jour l'interface utilisateur de manière thread-safe"""
        # Mise à jour distance
        if distance != float('inf'):
            self.label_distance.config(text=f"Distance: {distance:.1f} cm")
        else:
            self.label_distance.config(text="Distance: Erreur de mesure")
        
        # Mise à jour capteur NPN avec statut de blocage
        if etat_npn is not None:
            statut_npn = "Métal détecté" if etat_npn == 0 else "Pas de métal"
            if hasattr(self, 'npn_bloque') and self.npn_bloque:
                statut_npn += " (BLOQUÉ)"
            self.label_npn.config(text=f"Capteur NPN: {statut_npn}")
            couleur = "orange" if etat_npn == 0 else "green"
            if hasattr(self, 'npn_bloque') and self.npn_bloque:
                couleur = "red"  # Rouge quand bloqué
            self.canvas_npn.itemconfig(self.indicator_npn, fill=couleur)

    # ===============================
    # CONTRÔLES SYSTÈME
    # ===============================

    def toggle_mode_convoyeur(self):
        """Active/désactive le mode convoyeur"""
        self.mode_convoyeur = not self.mode_convoyeur
        if self.mode_convoyeur:
            self.btn_mode_convoyeur.config(text="🏭 Mode Normal")
            self.statusbar.config(text="🏭 Mode convoyeur activé")
            # Démarrer le convoyeur si détection active
            if self.detection_active:
                self.demarrer_convoyeur()
        else:
            self.btn_mode_convoyeur.config(text="🏭 Mode Convoyeur")
            self.arreter_convoyeur()
            self.annuler_timers()  # Annuler tous les timers en cours
            self.detection_en_cours = False
            self.npn_bloque = False  # Débloquer le NPN
            self.statusbar.config(text="Mode normal activé")
        
        self.ajouter_evenement(f"Mode {'convoyeur' if self.mode_convoyeur else 'normal'}", "Système", "Activé")

    def demarrer_detection(self):
        """Démarre le thread de détection"""
        if not self.detection_active:
            self.detection_active = True
            self.stop_thread = False
            self.thread_detection = threading.Thread(target=self.detection_presence)
            self.thread_detection.daemon = True
            self.thread_detection.start()
            
            # DÉMARRER LE CONVOYEUR SI MODE ACTIVÉ
            if self.mode_convoyeur:
                self.demarrer_convoyeur()
            
            self.btn_start.config(state="disabled")
            self.btn_stop.config(state="normal")
            self.statusbar.config(text="✅ Système démarré")
            self.ajouter_evenement("Système démarré", "Système", "Actif")

    def arreter_detection(self):
        """Arrête le thread de détection"""
        if self.detection_active:
            self.stop_thread = True
            self.detection_active = False
            
            # Annuler tous les timers et réinitialiser les blocages
            self.annuler_timers()
            self.detection_en_cours = False
            self.npn_bloque = False  # Débloquer le NPN à l'arrêt
            
            # Arrêter le convoyeur
            if self.mode_convoyeur:
                self.arreter_convoyeur()
            
            # Attendre que le thread se termine
            if self.thread_detection and self.thread_detection.is_alive():
                self.thread_detection.join(timeout=1.0)
            
            self.btn_start.config(state="normal")
            self.btn_stop.config(state="disabled")
            self.statusbar.config(text="⏹️ Système arrêté")
            self.ajouter_evenement("Système arrêté", "Système", "Inactif")

    # ===============================
    # TESTS ET CONFIGURATION
    # ===============================

    def test_servo(self):
        """Test le servomoteur 1 en faisant un balayage"""
        self.statusbar.config(text="Test du servomoteur 1 en cours...")
        
        def run_test():
            positions = [self.angle_ouvert, self.angle_ferme, 90, self.angle_ouvert]
            for pos in positions:
                self.bouger_servo(pos)
                time.sleep(1)
            
            self.root.after(0, lambda: self.statusbar.config(text="Test du servomoteur 1 terminé"))
            self.root.after(0, lambda: self.ajouter_evenement("Test servo 1 effectué", "Servo1", "Test complet"))
        
        threading.Thread(target=run_test, daemon=True).start()

    def test_servo2(self):
        """Test le servomoteur 2 en faisant un balayage"""
        self.statusbar.config(text="Test du servomoteur 2 en cours...")
        
        def run_test():
            positions = [self.angle_ouvert2, self.angle_ferme2, 25, self.angle_ferme2]
            for pos in positions:
                self.bouger_servo2(pos)
                time.sleep(1)
            
            self.root.after(0, lambda: self.statusbar.config(text="Test du servomoteur 2 terminé"))
            self.root.after(0, lambda: self.ajouter_evenement("Test servo 2 effectué", "Servo2", "Test complet"))
        
        threading.Thread(target=run_test, daemon=True).start()

    def appliquer_config(self):
        """Appliquer les changements de configuration"""
        self.angle_ouvert = self.angle_ouvert_var.get()
        self.angle_ferme = self.angle_ferme_var.get()
        self.angle_ouvert2 = self.angle_ouvert2_var.get()
        self.angle_ferme2 = self.angle_ferme2_var.get()
        self.SEUIL_CM = int(float(self.seuil_var.get()))
        
        # Appliquer les temporisations
        self.tempo_continue_tapis = float(self.tempo_continue_var.get())
        self.tempo_servo1_ferme = float(self.tempo_servo1_var.get())
        self.tempo_arret_npn = float(self.tempo_npn_var.get())
        self.tempo_servo2_ouvert = float(self.tempo_servo2_var.get())
        self.tempo_blocage_npn = float(self.tempo_blocage_var.get())
        
        self.statusbar.config(text="✅ Configuration mise à jour")
        self.ajouter_evenement("Configuration mise à jour", "Système", f"Seuil={self.SEUIL_CM}cm")

    # ===============================
    # HISTORIQUE ET LOGS
    # ===============================

    def ajouter_evenement(self, evenement, capteur, valeur):
        """Ajoute un événement à l'historique"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        self.historique.append((timestamp, evenement, capteur, str(valeur)))
        
        if len(self.historique) > 100:
            self.historique = self.historique[-100:]
        
        self.mettre_a_jour_historique()
        self.enregistrer_evenement(timestamp, evenement, capteur, valeur)

    def mettre_a_jour_historique(self):
        """Met à jour l'affichage de l'historique"""
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        for event in reversed(self.historique[-self.HISTORIQUE_MAX:]):
            self.tree.insert("", "end", values=event)

    def effacer_historique(self):
        """Efface l'historique affiché"""
        if messagebox.askyesno("Confirmation", "Voulez-vous effacer l'historique?"):
            self.historique = []
            self.mettre_a_jour_historique()

    def creer_fichier_log(self):
        """Crée le fichier de log si nécessaire"""
        if not os.path.exists(self.log_fichier):
            with open(self.log_fichier, 'w', newline='', encoding='utf-8') as fichier:
                writer = csv.writer(fichier)
                writer.writerow(["Horodatage", "Événement", "Capteur", "Valeur"])

    def enregistrer_evenement(self, timestamp, evenement, capteur, valeur):
        """Enregistre un événement dans le fichier CSV"""
        try:
            with open(self.log_fichier, 'a', newline='', encoding='utf-8') as fichier:
                writer = csv.writer(fichier)
                writer.writerow([timestamp, evenement, capteur, str(valeur)])
        except Exception as e:
            print(f"Erreur d'enregistrement: {e}")

    # ===============================
    # NETTOYAGE ET FERMETURE
    # ===============================

    def nettoyer(self):
        """Nettoie les ressources GPIO avant de quitter"""
        if self.detection_active:
            self.arreter_detection()
            time.sleep(0.5)
        
        # Annuler tous les timers
        self.annuler_timers()
        
        if self.pwm_servo:
            self.pwm_servo.stop()
        if self.pwm_servo2:
            self.pwm_servo2.stop()
        if self.pwm_motor:
            self.pwm_motor.stop()
        GPIO.cleanup()
        
        print("Nettoyage GPIO terminé")

# ===============================
# FONCTION PRINCIPALE
# ===============================

def main():
    root = tk.Tk()
    app = SystemeDetectionIntegre(root)
    
    def on_closing():
        """Gestion propre de la fermeture de l'application"""
        app.nettoyer()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()