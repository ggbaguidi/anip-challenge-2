# ANIP Challenge 2 - Solutions

**Auteur:** Ahonakpon Guy GBAGUIDI  
**Institution:** IFRI, Université d'Abomey-Calavi  
**Date:** Novembre 2025

## 📋 Description

Ce repository contient les solutions pour les trois tâches du Challenge ANIP (Agence Nationale d'Identification des Personnes), implémentées avec **JAX/Flax** pour des performances optimales.

## 🎯 Tâches

### Task 1: Reconnaissance Faciale
- **Approche:** Réseau Siamois avec Triplet Loss
- **Architecture:** ResNet-50 partagé pour l'extraction de features
- **Performance:** 99.6% de précision sur 1000 paires de test
- **Techniques:** Hard Negative Mining, similarité cosinus

### Task 2: Estimation d'Âge
- **Approche:** Multi-task Learning (âge + genre)
- **Architecture:** ResNet-50 avec deux têtes de prédiction
- **Performance:** MAE de 3.73 ans, RMSE de 4.89 ans
- **Dataset:** 2000 échantillons d'entraînement

### Task 3: Détection de Fraude Documentaire
- **Approche:** Fusion multimodale (vision + texte)
- **Composants:** ResNet-50 + Tesseract OCR + Late Fusion
- **Performance:** 78-82% accuracy sur documents de 4 pays
- **Détection:** 4 types de falsifications (police, couleur, texture, mise en page)

## 🚀 Stack Technique

- **Framework:** JAX/Flax
- **Vision:** ResNet-50 pré-entraîné
- **OCR:** Tesseract
- **Optimisation:** Adam, learning rate scheduling
- **Augmentation:** Rotation, flip, crop, normalisation

## 📂 Structure

```
anip-challenge-2/
├── task1-facialrecognition-jax.ipynb    # Reconnaissance faciale
├── task2-ageestimation-jax.ipynb        # Estimation d'âge
├── task3-ocr-frauddetection-jax.ipynb   # Détection de fraude OCR
└── README.md
```

## 🛠️ Installation

```bash
# Cloner le repository
git clone https://github.com/VOTRE_USERNAME/anip-challenge-2.git
cd anip-challenge-2

# Installer les dépendances
pip install jax[cuda] flax optax pillow matplotlib scikit-learn
pip install pytesseract  # Pour Task 3
```

## 📊 Résultats

| Tâche | Métrique Principale | Score |
|-------|---------------------|-------|
| Task 1 | Précision | 99.6% |
| Task 2 | MAE | 3.73 ans |
| Task 3 | Accuracy | 78-82% |

## 🎓 À propos

Projet réalisé dans le cadre du Challenge ANIP 2025 pour développer des solutions d'intelligence artificielle appliquées à l'identification et la sécurité documentaire.

## 📝 License

MIT License - voir le fichier LICENSE pour plus de détails

## 📧 Contact

Pour toute question ou collaboration:
- **Email:** guygbaguidi123root@gmail.com
- **LinkedIn:** [linkedin.com/in/aggbaguidi]
- **GitHub:** [@ggbaguidi]

---

⭐ N'hésitez pas à mettre une étoile si ce projet vous a été utile!
