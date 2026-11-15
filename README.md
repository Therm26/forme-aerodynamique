# Forme aérodynamique

Application Streamlit qui compare la traînée aérodynamique de formes simples dans l'air et permet d'explorer, avec un curseur, l'effet de l'élancement d'un corps sur son coefficient de traînée.

Projet d'apprentissage construit avec Claude Code. Les formules et les sources sont documentées dans l'application, section "Sources et limites".

## Ce que fait l'application

Tu entres trois paramètres dans la barre latérale :

- la vitesse de déplacement (km/h)
- une taille caractéristique L de l'objet (m)
- la température de l'air (°C)

L'application calcule ensuite :

1. La densité de l'air (loi des gaz parfaits) et sa viscosité (loi de Sutherland), à partir de la température.
2. Le nombre de Reynolds : `Re = ρ · v · L / µ`.
3. Le coefficient de traînée Cx de chaque forme, selon le régime d'écoulement.
4. La force de traînée : `F = 0,5 · ρ · v² · Cx · A`, et la puissance nécessaire `P = F · v`.
5. Un graphique comparatif, avec la forme la plus aérodynamique mise en avant.

Formes comparées : sphère, cylindre, cube, plaque plate, profilé en goutte d'eau. L'aire frontale est fixée à L² pour toutes les formes, afin de comparer à taille égale.

## Visualisation de l'écoulement

Une section dédiée dessine la silhouette de la forme choisie, des lignes de courant autour, et une zone de sillage pour les formes non profilées.

Ce n'est pas une simulation CFD. Les lignes de courant sont déformées géométriquement autour du profil, et la taille du sillage est illustrative. Elle sert à voir pourquoi un profilé traîne moins qu'un cube, pas à mesurer quoi que ce soit.

## Explorateur d'élancement

Un curseur (longueur/diamètre, de 0,2 à 8) fait varier en continu la silhouette d'un corps de révolution, du disque jusqu'à la goutte d'eau. Le diamètre maximal reste égal à L, donc l'aire frontale ne change pas, seule la longueur varie.

Le Cx est interpolé en log-log entre trois ancrages documentés : disque ou plaque (Cx ≈ 1,17), sphère (corrélation de Morrison au Reynolds courant) et profilé (Cx ≈ 0,04 à élancement 8). Entre ces points, la valeur est une approximation, pas une mesure.

## Qualité des données

| Forme | Source du Cx | Statut |
|---|---|---|
| Sphère | Morrison (2013), corrélation continue de Stokes jusqu'à Re ≈ 10⁶, crise de traînée incluse | Corrélation documentée |
| Cylindre, Re ≤ 2×10⁵ | Sucker & Brauer (1975) | Corrélation documentée |
| Cylindre, Re > 2×10⁵ | Table approximative, raccordée sans discontinuité à Sucker & Brauer | Approximation |
| Cube, plaque plate, profilé | Valeurs de plateau approximatives | Approximation |

Références :

- Morrison, F. A., *An Introduction to Fluid Mechanics*, Cambridge University Press, 2013.
- Sucker, D. et Brauer, H., *Wärme- und Stoffübertragung*, vol. 8, 1975, p. 149-158.

## Installation

Python 3.10 ou plus récent est recommandé.

```
git clone https://github.com/Therm26/forme-aerodynamique.git
cd forme-aerodynamique
pip install streamlit numpy matplotlib
streamlit run app.py
```

L'application s'ouvre sur `http://localhost:8501`. Si un fichier `requirements.txt` est présent dans le dépôt, utilise `pip install -r requirements.txt` à la place.

## Limites connues

- Pas de simulation numérique de l'écoulement : les visualisations sont pédagogiques.
- Trois formes sur cinq reposent sur des valeurs de Cx approximatives.
- Le Cx de l'explorateur d'élancement est interpolé à partir de trois points seulement.
- L'aire frontale est supposée égale à L² pour toutes les formes, ce qui n'est pas la vraie aire de la sphère ou du cylindre.
- L'air est le seul fluide géré, à pression atmosphérique standard.

## Pistes d'amélioration

- Remplacer les tables approximatives du cube, de la plaque et du profilé par des corrélations sourcées.
- Ajouter l'altitude (densité et viscosité de l'air variables).
- Ajouter des unités impériales.
- Proposer des formes 3D interactives.

## Licence

À définir.
