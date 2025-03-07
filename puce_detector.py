import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor
import io
import os
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="Hermine Detector",
    page_icon="🐱🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS amélioré - suppression des blocs inutiles
st.markdown("""
<style>
    /* Variables de couleurs */
    :root {
        --bg-dark: #121212;
        --bg-card: #1e1e1e;
        --accent-primary: #ff9f1c;
        --text-light: #f8f9fa;
        --text-secondary: #cccccc;
        --positive: #4caf50;
        --negative: #f44336;
        --card-border: #2c2c2c;
    }
    
    /* Style global */
    .stApp {
        background-color: var(--bg-dark);
    }
    
    .main .block-container {
        padding: 1.5rem;
    }
    
    /* Suppression des blocs vides */
    [data-testid="stFileUploadDropzone"] > div:empty {
        display: none !important;
    }
    
    [data-testid="stVerticalBlock"] > div:empty {
        display: none !important;
    }
    
    /* Suppression des marges inutiles */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    
    /* Masquer les conteneurs vides */
    div:empty {
        display: none !important;
    }
    
    /* Style amélioré pour l'uploader de fichiers */
    [data-testid="stFileUploader"] {
        background-color: var(--bg-card);
        border-radius: 8px;
        padding: 1rem;
        border: 2px dashed var(--accent-primary);
    }
    
    /* Reset des styles par défaut de Streamlit */
    h1, h2, h3, h4, h5, h6, p, div {
        color: var(--text-light);
    }
    
    /* Cartes avec style unifié */
    .card {
        background-color: var(--bg-card);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border-left: 3px solid var(--accent-primary);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Titres avec style accentué */
    .card-title {
        color: var(--accent-primary);
        font-size: 1.4rem;
        margin-top: 0;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    /* Style pour le titre principal */
    .main-title {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 1.5rem;
        font-size: 2.5rem;
        color: var(--text-light);
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
    }
    
    .main-title-icon {
        color: var(--accent-primary);
        margin-right: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: var(--text-secondary);
        font-size: 1.2rem;
        margin-bottom: 1.5rem;
    }
    
    /* Styles des résultats */
    .result-card {
        text-align: center;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .positive-result {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.2), rgba(76, 175, 80, 0.05));
        border: 1px solid var(--positive);
    }
    
    .negative-result {
        background: linear-gradient(135deg, rgba(244, 67, 54, 0.2), rgba(244, 67, 54, 0.05));
        border: 1px solid var(--negative);
    }
    
    .result-title {
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .result-value {
        font-size: 2.8rem;
        font-weight: bold;
    }
    
    .positive-text {
        color: var(--positive);
    }
    
    .negative-text {
        color: var(--negative);
    }
    
    /* Personnalisation du style des boutons */
    .stButton button, .css-1x8cf1d {
        background-color: var(--accent-primary) !important;
        color: var(--text-light) !important;
        border: none !important;
        padding: 0.6rem 1.2rem !important;
        border-radius: 50px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton button:hover, .css-1x8cf1d:hover {
        background-color: #e58e00 !important;
        box-shadow: 0 4px 12px rgba(255, 159, 28, 0.4) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Style pour les upload/fileuploader */
    .st-bd, .st-at, .st-ae {
        background-color: var(--bg-card) !important;
        border: 1px dashed var(--accent-primary) !important;
        border-radius: 8px !important;
    }
    
    /* Style pour éliminer les espaces indésirables */
    .element-container {
        margin-bottom: 0.5rem !important;
    }
    
    /* Suppression des marges inutiles autour des images */
    [data-testid="stImage"] {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Fonction pour charger le modèle
@st.cache_resource
def charger_modele(chemin_modele):
    """Charge le modèle TensorFlow et le met en cache."""
    return load_model(chemin_modele)

# Fonction pour prétraiter l'image et faire une prédiction
def predire_image(image, modele):
    # Redimensionner à 224x224
    img_redim = image.resize((224, 224))
    
    # Convertir en array et normaliser
    img_array = np.array(img_redim) / 255.0
    
    # Vérifier les canaux RGB
    if img_array.shape[-1] != 3:
        st.error("L'image doit être en couleur (RGB)")
        return None
    
    # Ajouter la dimension du batch
    img_array = np.expand_dims(img_array, axis=0)
    
    # Faire la prédiction
    prediction = modele.predict(img_array, verbose=0)[0][0]
    
    return prediction

# Fonction pour créer un sceau de certification élégant et discret
def ajouter_sceau_image(image, est_hermine=True):
    """Crée un sceau de certification élégant avec 'certifié Puce'."""
    # Créer une copie de l'image originale en RGBA
    img_resultat = image.copy().convert("RGBA")
    largeur, hauteur = img_resultat.size
    
    # Taille réduite pour le sceau (environ 20% de la plus petite dimension)
    taille_sceau = int(min(largeur, hauteur) * 0.22)
    
    # Création du sceau avec canal alpha
    sceau = Image.new('RGBA', (taille_sceau, taille_sceau), (0, 0, 0, 0))
    draw = ImageDraw.Draw(sceau)
    
    # Centre du sceau
    centre_x = taille_sceau // 2
    centre_y = taille_sceau // 2
    
    # Rayon du cercle principal
    rayon = (taille_sceau // 2) - 2
    
    # Paramètres selon le résultat
    if est_hermine:
        # Sceau positif (vert)
        couleur_cercle = (76, 175, 80, 220)  # Vert translucide
        couleur_interieur = (0, 0, 0, 160)   # Fond noir semi-transparent
        symbole = "✓"  # Coche
    else:
        # Sceau négatif (rouge)
        couleur_cercle = (244, 67, 54, 220)  # Rouge translucide
        couleur_interieur = (0, 0, 0, 160)   # Fond noir semi-transparent
        symbole = "✗"  # Croix
    
    # Dessiner le cercle extérieur
    draw.ellipse((0, 0, taille_sceau-1, taille_sceau-1), 
                 fill=couleur_interieur, outline=couleur_cercle, width=3)
    
    # Dessiner un cercle intérieur décoratif
    draw.ellipse((centre_x - rayon*0.7, centre_y - rayon*0.7, 
                  centre_x + rayon*0.7, centre_y + rayon*0.7), 
                 fill=None, outline=couleur_cercle, width=1)
    
    # Dessiner les lettres "PUCE" en utilisant des traits simples
    stroke_width = max(1, int(taille_sceau * 0.02))
    
    # Position pour "PUCE"
    text_y = int(taille_sceau * 0.32)
    letter_spacing = int(taille_sceau * 0.17)  # Increased spacing between letters
    
    # Position de départ
    start_x = int(taille_sceau * 0.25)
    
    # # Dessiner "P"
    # p_x = start_x
    # p_height = int(taille_sceau * 0.12)
    # draw.line((p_x, text_y - p_height//2, p_x, text_y + p_height//2), 
    #           fill=couleur_cercle, width=stroke_width)
    # draw.arc((p_x, text_y - p_height//2, 
    #           p_x + int(taille_sceau * 0.08), text_y), 
    #          start=270, end=90, fill=couleur_cercle, width=stroke_width)
    
    # # Improved U drawing
    # u_x = start_x + letter_spacing
    # u_height = int(taille_sceau * 0.12)
    # u_width = int(taille_sceau * 0.08)
    
    # # Left vertical line of U
    # draw.line((u_x, text_y - u_height//2, u_x, text_y + u_height//3), 
    #           fill=couleur_cercle, width=stroke_width)
    
    # # Bottom curve of U
    # draw.arc((u_x, text_y + u_height//3 - u_height//4, 
    #           u_x + u_width, text_y + u_height//3 + u_height//4), 
    #          start=180, end=0, fill=couleur_cercle, width=stroke_width)
    
    # # Right vertical line of U
    # draw.line((u_x + u_width, text_y - u_height//2, u_x + u_width, text_y + u_height//3), 
    #           fill=couleur_cercle, width=stroke_width)
    
    # # Dessiner "C"
    # c_x = start_x + letter_spacing * 2
    # c_radius = int(taille_sceau * 0.08)
    # draw.arc((c_x - c_radius, text_y - c_radius, 
    #           c_x + c_radius, text_y + c_radius),
    #          start=60, end=300, fill=couleur_cercle, width=stroke_width)
    
    # # Dessiner "E"
    # e_x = start_x + letter_spacing * 3
    # e_height = int(taille_sceau * 0.12)
    # draw.line((e_x, text_y - e_height//2, e_x, text_y + e_height//2), 
    #           fill=couleur_cercle, width=stroke_width)
    # draw.line((e_x, text_y - e_height//2, e_x + int(taille_sceau * 0.07), text_y - e_height//2), 
    #           fill=couleur_cercle, width=stroke_width)
    # draw.line((e_x, text_y, e_x + int(taille_sceau * 0.05), text_y), 
    #           fill=couleur_cercle, width=stroke_width)
    # draw.line((e_x, text_y + e_height//2, e_x + int(taille_sceau * 0.07), text_y + e_height//2), 
    #           fill=couleur_cercle, width=stroke_width)

    # Dessiner le symbole au milieu (✓ ou ✗)
    if est_hermine:
        # Coche (✓)
        check_size = int(taille_sceau * 0.2)
        draw.line((centre_x - check_size, centre_y, 
                   centre_x - check_size//2, centre_y + check_size//2),
                 fill=couleur_cercle, width=stroke_width*2)
        draw.line((centre_x - check_size//2, centre_y + check_size//2,
                   centre_x + check_size, centre_y - check_size//2),
                 fill=couleur_cercle, width=stroke_width*2)
    else:
        # Croix (✗)
        cross_size = int(taille_sceau * 0.15)
        draw.line((centre_x - cross_size, centre_y - cross_size,
                   centre_x + cross_size, centre_y + cross_size),
                 fill=couleur_cercle, width=stroke_width*2)
        draw.line((centre_x + cross_size, centre_y - cross_size,
                   centre_x - cross_size, centre_y + cross_size),
                 fill=couleur_cercle, width=stroke_width*2)
    
    # Position du sceau - coin inférieur droit mais discret
    position = (largeur - taille_sceau - 20, hauteur - taille_sceau - 20)
    
    # Superposer le sceau sur l'image
    img_resultat.paste(sceau, position, sceau)
    
    return img_resultat

# Titre principal
st.markdown("""
<div style="margin-top: 20px;">
    <div class="main-title">
        <span class="main-title-icon">🐱</span> Hermine Detector
    </div>
    <div class="subtitle">
        Reconnaître mimine en un clin d'oeil!
    </div>
</div>
""", unsafe_allow_html=True)

# Structure principale sans espaces inutiles
main_col, sidebar_col = st.columns([3, 1])

# Contenu principal
with main_col:
    # Zone d'introduction avec style unifié
    st.markdown("""
    <div class="card">
        <h2 class="card-title">Analysez votre photo</h2>
        <p>Téléversez une image pour savoir si c'est Hermine, l'IA analysera instantanément votre photo.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Zone de téléversement simplifiée
    fichier_image = st.file_uploader("", type=["jpg", "jpeg", "png"])
    
    # Chemin du modèle
    chemin_modele = r"C:\Users\vyuof\OneDrive\Bureau\projet chat\cat_model_final.h5"
    
    # Si une image est téléversée
    if fichier_image is not None:
        # Charger le modèle
        if os.path.exists(chemin_modele):
            with st.spinner(""):
                modele = charger_modele(chemin_modele)
        else:
            st.error(f"Modèle non trouvé: '{chemin_modele}'")
            st.stop()
        
        # Ouvrir l'image
        image = Image.open(fichier_image)
        
        # Disposition pour image et résultats sans conteneurs vides
        img_col, res_col = st.columns([1, 1])
        
        # Colonne de l'image sans marges inutiles
        with img_col:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Colonne des résultats
        with res_col:
            # Faire la prédiction
            with st.spinner(""):
                prediction = predire_image(image, modele)
            
            if prediction is not None:
                est_hermine = prediction > 0.5
                confiance = prediction if est_hermine else 1 - prediction
                confiance_pct = f"{confiance:.1%}"
                
                # Résultat avec design amélioré
                st.markdown('<div class="card">', unsafe_allow_html=True)
                
                if est_hermine:
                    st.markdown(f"""
                    <div class="result-card positive-result">
                        <div class="result-title positive-text">C'est Hermine! 🐱</div>
                        <div class="result-value positive-text">{confiance_pct}</div>
                        <div style="margin-top: 1rem; height: 12px; background-color: rgba(76, 175, 80, 0.2); border-radius: 6px;">
                            <div style="width: {confiance_pct}; height: 12px; background-color: var(--positive); border-radius: 6px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-card negative-result">
                        <div class="result-title negative-text">Ce n'est pas Hermine ❌</div>
                        <div class="result-value negative-text">{confiance_pct}</div>
                        <div style="margin-top: 1rem; height: 12px; background-color: rgba(244, 67, 54, 0.2); border-radius: 6px;">
                            <div style="width: {confiance_pct}; height: 12px; background-color: var(--negative); border-radius: 6px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Section de certification
        if prediction is not None:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<h2 class="card-title">Certification officielle</h2>', unsafe_allow_html=True)
            st.write("Téléchargez l'image avec le sceau officiel Hermine Detector.")
            
            # Créer l'image certifiée avec sceau au texte ÉNORME
            est_hermine = prediction > 0.5
            img_certifiee = ajouter_sceau_image(image, est_hermine)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Bouton de téléchargement
                buf = io.BytesIO()
                img_certifiee.save(buf, format="PNG")
                st.download_button(
                    label="Télécharger l'image certifiée",
                    data=buf.getvalue(),
                    file_name=f"hermine_certification_{datetime.now().strftime('%Y%m%d_%H%M')}.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            with col2:
                # Aperçu de l'image certifiée
                st.image(img_certifiee, caption="Aperçu", width=200)
            
            st.markdown('</div>', unsafe_allow_html=True)

# Barre latérale avec style unifié
with sidebar_col:
    st.markdown("""
    <div class="card">
        <h3 class="card-title">Technologie</h3>
        <ul>
            <li>MobileNetV2 avec Transfer Learning</li>
            <li>Analyse en temps réel</li>
            <li>Certification officielle avec sceau</li>
        </ul>
    </div>
    
    <div class="card">
        <h3 class="card-title">Comment utiliser</h3>
        <ol>
            <li>Téléversez une photo de chat</li>
            <li>Attendez l'analyse de l'IA</li>
            <li>Obtenez votre certification officielle</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)