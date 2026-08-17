"""
Comparateur de traînée aérodynamique pour des formes simples.

Modèle physique :
- Densité et viscosité de l'air calculées à partir de la température (loi des gaz
  parfaits + loi de Sutherland), pression atmosphérique standard.
- Nombre de Reynolds Re = rho * v * L / mu, avec L la taille caractéristique saisie
  par l'utilisateur (diamètre pour la sphère/le cylindre, côté pour le cube/la
  plaque, longueur pour le profilé).
- Coefficient de traînée Cx(Re) par forme, à partir de courbes standards de la
  littérature (interpolation log-log entre points documentés). Sphère et cylindre
  reproduisent la "crise de traînée" autour de Re ~ 3.10^5 ; cube, plaque et
  profilé sont représentés par un plateau quasi constant à grand Re, conforme aux
  tables usuelles de mécanique des fluides (voir la section "Sources" dans l'appli).
- Force de traînée F = 0.5 * rho * v^2 * Cx * A.

C'est une première version avec des approximations documentées mais simplifiées :
à affiner (courbes plus fines, formes 3D plus réalistes, altitude, etc.).
"""

import math

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Constantes physiques
# ----------------------------------------------------------------------------

# Palette (voir skill dataviz : accent + gris de désaccentuation, statut rouge réservé)
ACCENT = "#2a78d6"  # bleu, slot catégoriel 1
NEUTRAL = "#c9c8c2"  # gris de désaccentuation
BODY_COLOR = "#8a8983"  # gris plus soutenu pour les silhouettes de formes
WAKE_COLOR = "#d03b3b"  # rouge statut "critical" — zone de sillage/traînée
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

P_ATM = 101_325.0  # Pa, pression atmosphérique standard au niveau de la mer
R_AIR = 287.05  # J/(kg.K), constante spécifique de l'air sec
# Loi de Sutherland pour la viscosité dynamique de l'air
MU_0 = 1.716e-5  # Pa.s, viscosité de référence
T_0 = 273.15  # K
C_SUTHERLAND = 110.4  # K


def air_density(temp_k: float) -> float:
    """Densité de l'air (kg/m^3) via la loi des gaz parfaits, pression standard."""
    return P_ATM / (R_AIR * temp_k)


def air_viscosity(temp_k: float) -> float:
    """Viscosité dynamique de l'air (Pa.s) via la loi de Sutherland."""
    return MU_0 * (T_0 + C_SUTHERLAND) / (temp_k + C_SUTHERLAND) * (temp_k / T_0) ** 1.5


# ----------------------------------------------------------------------------
# Coefficients de traînée Cx(Re) par forme
#
# Chaque forme est définie par une table de points (Re, Cx) documentés,
# interpolés linéairement en échelle log-log et prolongés en constante au-delà
# des bornes. C'est une approximation volontairement simple : ajuster les
# tables ci-dessous pour affiner le modèle par la suite.
# ----------------------------------------------------------------------------

SPHERE_PTS = [
    (1e-1, 240), (1, 27), (10, 4.3), (1e2, 1.1), (1e3, 0.47),
    (1e4, 0.47), (1e5, 0.5), (2e5, 0.5), (3e5, 0.09),  # crise de traînée
    (5e5, 0.13), (1e6, 0.2), (1e7, 0.2),
]

CYLINDER_PTS = [
    (1e-1, 100), (1, 10), (10, 2.8), (1e2, 1.4), (1e3, 1.2),
    (1e4, 1.2), (1e5, 1.2), (2e5, 1.1), (3e5, 0.3),  # crise de traînée
    (5e5, 0.35), (1e6, 0.7), (1e7, 0.7),
]

CUBE_PTS = [
    (1e-1, 240), (1, 24), (10, 5), (1e2, 1.6), (1e3, 1.05), (1e7, 1.05),
]

PLATE_PTS = [
    (1e-1, 240), (1, 24), (10, 5), (1e2, 1.6), (1e3, 1.17), (1e7, 1.17),
]

STREAMLINED_PTS = [
    (1e-1, 240), (1, 24), (10, 3), (1e2, 0.3), (1e3, 0.06),
    (1e4, 0.04), (1e7, 0.04),
]


def cd_from_table(re: float, points: list[tuple[float, float]]) -> float:
    """Interpolation log-log de Cx en fonction de Re, à partir d'une table de points."""
    res, cds = zip(*points)
    log_re = np.log10(res)
    log_cd = np.log10(cds)
    lr = np.clip(np.log10(max(re, 1e-6)), log_re[0], log_re[-1])
    return float(10 ** np.interp(lr, log_re, log_cd))


# ----------------------------------------------------------------------------
# Formes : fonction Cx(Re), aire frontale A(L), description, source
# ----------------------------------------------------------------------------


def frontal_area_disk(l: float) -> float:
    return math.pi / 4 * l**2


def frontal_area_square(l: float) -> float:
    return l**2


SHAPES = {
    "Sphère": {
        "cd": lambda re: cd_from_table(re, SPHERE_PTS),
        "area": frontal_area_disk,
        "taille": "diamètre",
        "source": "Courbe standard de traînée d'une sphère lisse (crise de traînée vers Re ≈ 3.10⁵), "
        "valeurs usuelles de la littérature (ex. Schlichting, Munson et al.).",
    },
    "Cylindre (travers du flux)": {
        "cd": lambda re: cd_from_table(re, CYLINDER_PTS),
        "area": frontal_area_square,
        "taille": "diamètre (longueur supposée égale)",
        "source": "Courbe standard de traînée d'un cylindre circulaire en écoulement transverse "
        "(crise de traînée similaire à la sphère, valeurs approximatives d'après Roshko 1961 "
        "et tables usuelles).",
    },
    "Cube": {
        "cd": lambda re: cd_from_table(re, CUBE_PTS),
        "area": frontal_area_square,
        "taille": "côté",
        "source": "Cx ≈ 1.05 à grand Re (arêtes vives, décollement fixé par la géométrie), "
        "valeur classique de table (Munson, Young & Okiishi, Fundamentals of Fluid Mechanics).",
    },
    "Plaque plate (perpendiculaire)": {
        "cd": lambda re: cd_from_table(re, PLATE_PTS),
        "area": frontal_area_square,
        "taille": "côté",
        "source": "Cx ≈ 1.17 pour une plaque carrée perpendiculaire à l'écoulement, "
        "valeur classique de table (Hoerner, Fluid-Dynamic Drag ; Munson et al.).",
    },
    "Profilé aérodynamique": {
        "cd": lambda re: cd_from_table(re, STREAMLINED_PTS),
        "area": frontal_area_square,
        "taille": "envergure/section (bounding box)",
        "source": "Cx ≈ 0.04 pour un corps profilé/effilé (mât profilé, corps de révolution "
        "profilé), valeur classique de table (Hoerner ; Munson et al.).",
    },
}


# ----------------------------------------------------------------------------
# Visualisation 2D de l'écoulement autour de chaque forme
#
# Schéma pédagogique (pas une simulation CFD) : les lignes de courant sont
# déformées géométriquement pour épouser le profil réel de la forme, avec un
# décollement (fin des lignes internes + zone de sillage ombrée) pour les
# formes "non profilées". Le profilé aérodynamique n'a pas de décollement :
# les lignes restent attachées jusqu'au bord de fuite.
# ----------------------------------------------------------------------------

# (demi-largeur du sillage, longueur du sillage) en multiples de L ; (0, 0) = pas de décollement
WAKE_PARAMS = {
    "Sphère": dict(half=0.35, length=1.6),
    "Cylindre (travers du flux)": dict(half=0.42, length=1.9),
    "Cube": dict(half=0.5, length=2.3),
    "Plaque plate (perpendiculaire)": dict(half=0.55, length=2.7),
    "Profilé aérodynamique": dict(half=0.0, length=0.0),
}


def shape_profile(name: str, l: float):
    """Retourne (h(x) = demi-hauteur du profil en x, (x_min_corps, x_max_corps))."""
    if name in ("Sphère", "Cylindre (travers du flux)"):
        R = l / 2

        def h(x):
            x = np.asarray(x, dtype=float)
            return np.sqrt(np.maximum(R**2 - x**2, 0))

        return h, (-R, R)
    if name == "Cube":
        half = l / 2

        def h(x):
            x = np.asarray(x, dtype=float)
            return np.where(np.abs(x) <= half, half, 0.0)

        return h, (-half, half)
    if name == "Plaque plate (perpendiculaire)":
        thick = l * 0.025
        half = l / 2

        def h(x):
            x = np.asarray(x, dtype=float)
            return np.where(np.abs(x) <= thick, half, 0.0)

        return h, (-thick, thick)
    if name == "Profilé aérodynamique":
        half_l = l / 2

        def h(x):
            x = np.asarray(x, dtype=float)
            xn = np.clip((x + half_l) / l, 0, 1)
            hh = 0.39 * l * np.sqrt(xn) * (1 - xn)
            return np.where((x >= -half_l) & (x <= half_l), hh, 0.0)

        return h, (-half_l, half_l)
    raise ValueError(name)


def _smooth(y: np.ndarray, sigma_pts: int) -> np.ndarray:
    """Lissage gaussien simple pour éviter les angles vifs dans les lignes de courant."""
    if sigma_pts <= 0:
        return y
    k = sigma_pts * 4 + 1
    kernel = np.exp(-0.5 * ((np.arange(k) - k // 2) / sigma_pts) ** 2)
    kernel /= kernel.sum()
    yp = np.pad(y, k // 2, mode="edge")
    return np.convolve(yp, kernel, mode="valid")


def _draw_flow_on_ax(ax, name: str, l: float, compact: bool = False):
    """Dessine la silhouette + lignes de courant approx. d'une forme sur un axe donné."""
    h_fn, (x0, x1) = shape_profile(name, l)
    wp = WAKE_PARAMS[name]
    wake_len = wp["length"] * l
    wake_half = wp["half"] * l

    x_min = -2.2 * l
    x_max = x1 + max(wake_len, 0.5 * l) + 1.0 * l
    y_max = 0.85 * l
    margin = 0.055 * l
    fs = 7 if compact else 8.5

    xs = np.linspace(x_min, x_max, 300)
    body_h = h_fn(xs)

    y0s = np.linspace(-y_max * 0.9, y_max * 0.9, 10)

    ax.set_facecolor(SURFACE)

    if wake_len > 0:
        wake_patch = mpatches.Rectangle(
            (x1, -wake_half), wake_len, 2 * wake_half,
            facecolor=WAKE_COLOR, alpha=0.12, edgecolor=WAKE_COLOR, linestyle="--", linewidth=1, zorder=1,
        )
        ax.add_patch(wake_patch)
        ax.text(
            x1 + wake_len / 2, -y_max * 0.97,
            "sillage" if compact else "sillage turbulent\n(basse pression)",
            ha="center", va="top", fontsize=fs, color=WAKE_COLOR,
        )

    for y0 in y0s:
        disp = np.maximum(0.0, body_h + margin - abs(y0))
        y = y0 + np.sign(y0) * disp
        y = _smooth(y, sigma_pts=4)
        keep = (xs <= x1) if (wake_len > 0 and abs(y0) < wake_half) else np.ones_like(xs, dtype=bool)
        ax.plot(xs[keep], y[keep], color=ACCENT, linewidth=1.1, zorder=2)

    if name in ("Sphère", "Cylindre (travers du flux)"):
        patch = mpatches.Circle((0, 0), l / 2, facecolor=BODY_COLOR, edgecolor=INK, linewidth=1.2, zorder=3)
    elif name == "Cube":
        half = l / 2
        patch = mpatches.Rectangle((-half, -half), l, l, facecolor=BODY_COLOR, edgecolor=INK, linewidth=1.2, zorder=3)
    elif name == "Plaque plate (perpendiculaire)":
        thick = l * 0.025
        patch = mpatches.Rectangle((-thick, -l / 2), 2 * thick, l, facecolor=BODY_COLOR, edgecolor=INK, linewidth=1.2, zorder=3)
    else:  # Profilé aérodynamique
        xs_body = np.linspace(x0, x1, 60)
        top = h_fn(xs_body)
        poly_x = np.concatenate([xs_body, xs_body[::-1]])
        poly_y = np.concatenate([top, -top[::-1]])
        patch = mpatches.Polygon(
            np.column_stack([poly_x, poly_y]), closed=True,
            facecolor=BODY_COLOR, edgecolor=INK, linewidth=1.2, zorder=3,
        )
    ax.add_patch(patch)

    if wake_len == 0 and not compact:
        ax.annotate(
            "écoulement recollé\njusqu'au bord de fuite",
            xy=(x1, 0), xytext=(x1 + 0.5 * l, y_max * 0.6),
            fontsize=fs, color=INK_SECONDARY,
            arrowprops=dict(arrowstyle="->", color=INK_SECONDARY, lw=1),
        )
    elif wake_len == 0 and compact:
        ax.text(
            x1 + 0.5 * l, -y_max * 0.97, "pas de décollement",
            ha="center", va="top", fontsize=fs, color=INK_SECONDARY,
        )

    ax.annotate(
        "", xy=(x_min + 0.35 * l, y_max * 0.78), xytext=(x_min, y_max * 0.78),
        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3),
    )
    if not compact:
        ax.text(x_min, y_max * 0.92, "écoulement", fontsize=fs, color=INK)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-y_max, y_max)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_flow_figure(name: str, l: float):
    """Figure matplotlib détaillée pour une seule forme."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    fig.patch.set_facecolor(SURFACE)
    _draw_flow_on_ax(ax, name, l, compact=False)
    fig.tight_layout()
    return fig


def draw_flow_comparison_figure(names: list[str], l: float, best: str):
    """Figure matplotlib avec toutes les formes côte à côte (petits multiples)."""
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.0))
    fig.patch.set_facecolor(SURFACE)
    for ax, name in zip(axes, names):
        _draw_flow_on_ax(ax, name, l, compact=True)
        is_best = name == best
        ax.set_title(
            ("★ " if is_best else "") + name,
            fontsize=10.5,
            color=ACCENT if is_best else INK,
            fontweight="bold" if is_best else "normal",
        )
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Explorateur d'élancement : silhouette continue disque -> sphère -> goutte d'eau
#
# La silhouette est un corps de révolution paramétré par l'élancement
# E = longueur / diamètre : nez toujours arrondi (exposant a=0.5 fixe), bord de
# fuite qui passe progressivement d'arrondi (b=0.5, ellipse/cercle à E=1) à
# pointu (b=1.0, goutte d'eau à E=ELONG_MAX). Le diamètre max reste égal à L
# quel que soit E, donc l'aire frontale (π/4·L², un disque) ne change pas :
# seule la longueur varie, exactement comme dans les abaques Cx-vs-élancement
# de la littérature (ex. Hoerner, Fluid-Dynamic Drag, corps de révolution).
#
# Le Cx n'est PAS mesuré pour chaque élancement : il est interpolé (log-log,
# même méthode que cd_from_table) entre les 3 SEULES valeurs documentées déjà
# utilisées ailleurs dans l'appli : disque/plaque (E→0, Cx≈1.17), sphère
# (E=1, Cx≈0.47) et profilé très élancé (E=ELONG_MAX, Cx≈0.04).
# ----------------------------------------------------------------------------

ELONG_MAX = 8.0  # élancement (L/D) du profilé le plus effilé du curseur
ELONG_ANCHORS = [0.05, 1.0, ELONG_MAX]  # position des 3 formes documentées sur l'axe élancement


def elongation_profile(elong: float, l: float):
    """Silhouette h(x) d'un corps de révolution d'élancement `elong`, diamètre `l`."""
    D = l
    Lc = max(elong, 0.05) * l
    a = 0.5  # nez arrondi, fixe (réaliste pour toute la famille)
    b = 0.5 if elong <= 1 else 0.5 + 0.5 * min(elong - 1, ELONG_MAX - 1) / (ELONG_MAX - 1)
    xn_peak = a / (a + b)
    peak_val = xn_peak**a * (1 - xn_peak) ** b
    C = (D / 2) / peak_val
    x0, x1 = -Lc / 2, Lc / 2

    def h(x):
        x = np.asarray(x, dtype=float)
        xn = np.clip((x - x0) / Lc, 0, 1)
        return np.where((x >= x0) & (x <= x1), C * xn**a * (1 - xn) ** b, 0.0)

    return h, (x0, x1)


def elongation_wake_params(elong: float):
    """Demi-largeur et longueur de sillage (en multiples de L), interpolées linéairement
    entre celles des 3 formes documentées (voir WAKE_PARAMS)."""
    halves = [
        WAKE_PARAMS["Plaque plate (perpendiculaire)"]["half"],
        WAKE_PARAMS["Sphère"]["half"],
        WAKE_PARAMS["Profilé aérodynamique"]["half"],
    ]
    lengths = [
        WAKE_PARAMS["Plaque plate (perpendiculaire)"]["length"],
        WAKE_PARAMS["Sphère"]["length"],
        WAKE_PARAMS["Profilé aérodynamique"]["length"],
    ]
    half = float(np.interp(elong, ELONG_ANCHORS, halves))
    length = float(np.interp(elong, ELONG_ANCHORS, lengths))
    return half, length


def cd_elongation(elong: float, re: float) -> float:
    """Cx interpolé (log-log) entre les 3 formes documentées, à l'élancement `elong`."""
    cd_disk = cd_from_table(re, PLATE_PTS)
    cd_sphere = cd_from_table(re, SPHERE_PTS)
    cd_streamlined = cd_from_table(re, STREAMLINED_PTS)
    anchors = list(zip(ELONG_ANCHORS, [cd_disk, cd_sphere, cd_streamlined]))
    return cd_from_table(elong, anchors)


def draw_elongation_figure(elong: float, l: float):
    """Figure matplotlib : silhouette + lignes de courant pour l'élancement `elong`."""
    h_fn, (x0, x1) = elongation_profile(elong, l)
    wake_half_ratio, wake_len_ratio = elongation_wake_params(elong)
    wake_half = wake_half_ratio * l
    wake_len = wake_len_ratio * l
    attached = wake_len < 0.08 * l

    x_min = -2.2 * l
    x_max = x1 + max(wake_len, 0.5 * l) + 1.0 * l
    y_max = 0.85 * l
    margin = 0.055 * l

    xs = np.linspace(x_min, x_max, 320)
    body_h = h_fn(xs)
    y0s = np.linspace(-y_max * 0.9, y_max * 0.9, 10)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    if not attached:
        wake_patch = mpatches.Rectangle(
            (x1, -wake_half), wake_len, 2 * wake_half,
            facecolor=WAKE_COLOR, alpha=0.12, edgecolor=WAKE_COLOR, linestyle="--", linewidth=1, zorder=1,
        )
        ax.add_patch(wake_patch)
        ax.text(
            x1 + wake_len / 2, -y_max * 0.97,
            "sillage turbulent\n(basse pression)",
            ha="center", va="top", fontsize=8.5, color=WAKE_COLOR,
        )

    for y0 in y0s:
        disp = np.maximum(0.0, body_h + margin - abs(y0))
        y = y0 + np.sign(y0) * disp
        y = _smooth(y, sigma_pts=4)
        keep = (xs <= x1) if (not attached and abs(y0) < wake_half) else np.ones_like(xs, dtype=bool)
        ax.plot(xs[keep], y[keep], color=ACCENT, linewidth=1.1, zorder=2)

    xs_body = np.linspace(x0, x1, 80)
    top = h_fn(xs_body)
    poly_x = np.concatenate([xs_body, xs_body[::-1]])
    poly_y = np.concatenate([top, -top[::-1]])
    patch = mpatches.Polygon(
        np.column_stack([poly_x, poly_y]), closed=True,
        facecolor=BODY_COLOR, edgecolor=INK, linewidth=1.2, zorder=3,
    )
    ax.add_patch(patch)

    if attached:
        ax.annotate(
            "écoulement quasi\nentièrement recollé",
            xy=(x1, 0), xytext=(x1 + 0.5 * l, y_max * 0.6),
            fontsize=8.5, color=INK_SECONDARY,
            arrowprops=dict(arrowstyle="->", color=INK_SECONDARY, lw=1),
        )

    ax.annotate(
        "", xy=(x_min + 0.35 * l, y_max * 0.78), xytext=(x_min, y_max * 0.78),
        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5),
    )
    ax.text(x_min, y_max * 0.92, "écoulement", fontsize=8.5, color=INK)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-y_max, y_max)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Interface Streamlit
# ----------------------------------------------------------------------------

st.set_page_config(page_title="Comparateur de traînée aérodynamique", page_icon="💨", layout="wide")

st.title("💨 Comparateur de traînée aérodynamique")
st.caption(
    "Compare la force de traînée dans l'air de cinq formes simples, pour une même "
    "taille caractéristique, à partir du nombre de Reynolds et de coefficients de "
    "traînée documentés."
)

with st.sidebar:
    st.header("Paramètres")
    v_kmh = st.number_input("Vitesse (km/h)", min_value=0.1, max_value=1000.0, value=50.0, step=1.0)
    taille = st.number_input("Taille caractéristique L (m)", min_value=0.001, max_value=10.0, value=0.3, step=0.05)
    temp_c = st.number_input("Température de l'air (°C)", min_value=-40.0, max_value=60.0, value=20.0, step=1.0)
    st.caption(
        "L est utilisée comme diamètre (sphère/cylindre) ou comme côté "
        "(cube/plaque/profilé) — voir le détail par forme plus bas."
    )

v = v_kmh / 3.6  # m/s
temp_k = temp_c + 273.15

rho = air_density(temp_k)
mu = air_viscosity(temp_k)

col1, col2, col3 = st.columns(3)
col1.metric("Densité de l'air", f"{rho:.3f} kg/m³")
col2.metric("Viscosité dynamique", f"{mu * 1e6:.2f} µPa·s")
col3.metric("Vitesse", f"{v:.2f} m/s")

# ----------------------------------------------------------------------------
# Calculs par forme
# ----------------------------------------------------------------------------

rows = []
for nom, s in SHAPES.items():
    A = s["area"](taille)
    re = rho * v * taille / mu
    cx = s["cd"](re)
    F = 0.5 * rho * v**2 * cx * A
    puissance = F * v
    rows.append(
        {
            "Forme": nom,
            "Reynolds": re,
            "Cx": cx,
            "Aire frontale (m²)": A,
            "Force de traînée (N)": F,
            "Puissance nécessaire (W)": puissance,
        }
    )

df = pd.DataFrame(rows).sort_values("Force de traînée (N)").reset_index(drop=True)
best = df.iloc[0]["Forme"]

st.subheader("Résultats")
st.dataframe(
    df.style.format(
        {
            "Reynolds": "{:,.0f}",
            "Cx": "{:.3f}",
            "Aire frontale (m²)": "{:.4f}",
            "Force de traînée (N)": "{:.2f}",
            "Puissance nécessaire (W)": "{:.1f}",
        }
    ),
    width="stretch",
    hide_index=True,
)

# ----------------------------------------------------------------------------
# Graphique comparatif (emphasis : meilleure forme en accent, reste en gris)
# ----------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 4.5))
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

colors = [ACCENT if forme == best else NEUTRAL for forme in df["Forme"]]
bars = ax.bar(df["Forme"], df["Force de traînée (N)"], color=colors, width=0.6, zorder=3)

for bar, val in zip(bars, df["Force de traînée (N)"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        val + df["Force de traînée (N)"].max() * 0.015,
        f"{val:.2f} N",
        ha="center",
        va="bottom",
        fontsize=9,
        color=INK,
    )

ax.set_ylabel("Force de traînée (N)", color=INK_SECONDARY)
ax.set_ylim(0, df["Force de traînée (N)"].max() * 1.15)
ax.tick_params(axis="x", colors=INK, labelrotation=15)
ax.tick_params(axis="y", colors=INK_SECONDARY)
ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color(GRID)

handles = [
    plt.Rectangle((0, 0), 1, 1, color=ACCENT),
    plt.Rectangle((0, 0), 1, 1, color=NEUTRAL),
]
ax.legend(
    handles,
    [f"Meilleure forme ({best})", "Autres formes"],
    loc="upper right",
    frameon=False,
    labelcolor=INK_SECONDARY,
)

fig.tight_layout()
st.pyplot(fig)

st.success(f"✅ À {v_kmh:.0f} km/h, la forme la plus aérodynamique est : **{best}**.")

# ----------------------------------------------------------------------------
# Visualisation de l'écoulement autour de la forme sélectionnée
# ----------------------------------------------------------------------------

st.subheader("🔍 Comparaison de toutes les formes")
st.pyplot(draw_flow_comparison_figure(list(SHAPES.keys()), taille, best))
st.caption(
    "Les cinq formes à la même taille caractéristique L, écoulement de gauche à droite. "
    "La zone rouge marque le sillage turbulent créé par le décollement de l'écoulement — "
    "plus elle est large, plus la traînée de pression est importante."
)

st.subheader("🌀 Détail d'une forme")
shape_choice = st.selectbox(
    "Forme à visualiser",
    options=list(SHAPES.keys()),
    index=list(SHAPES.keys()).index(best),
)
st.pyplot(draw_flow_figure(shape_choice, taille))
st.caption(
    "Schéma pédagogique (pas une simulation CFD) : les lignes bleues représentent des "
    "lignes de courant approximatives. Quand l'écoulement **décolle** de la forme (sphère, "
    "cylindre, cube, plaque), il laisse derrière lui un large **sillage turbulent à basse "
    "pression** (zone rouge) qui aspire l'objet vers l'arrière — c'est la principale source "
    "de traînée de pression. Un **profilé aérodynamique** est dessiné pour que l'écoulement "
    "reste attaché jusqu'au bord de fuite : le sillage est quasi inexistant, d'où un Cx "
    "beaucoup plus faible. La taille du sillage dessiné est représentative de l'écart entre "
    "formes, pas une mesure physique précise."
)

# ----------------------------------------------------------------------------
# Explorateur d'élancement (curseur continu)
# ----------------------------------------------------------------------------

st.subheader("📐 Explorateur d'élancement")
st.caption(
    "Fait varier en continu la silhouette d'un corps de révolution de diamètre L : "
    "émoussé proche du disque/sphère à faible élancement, jusqu'à une goutte d'eau "
    "très allongée à fort élancement."
)

elong = st.slider(
    "Élancement (longueur / diamètre)",
    min_value=0.2,
    max_value=ELONG_MAX,
    value=2.0,
    step=0.1,
)

re_elong = rho * v * taille / mu
cd_elong = cd_elongation(elong, re_elong)
area_elong = frontal_area_disk(taille)
F_elong = 0.5 * rho * v**2 * cd_elong * area_elong

ec1, ec2, ec3 = st.columns(3)
ec1.metric("Élancement L/D", f"{elong:.1f}")
ec2.metric("Cx interpolé", f"{cd_elong:.3f}")
ec3.metric("Force de traînée", f"{F_elong:.2f} N")

st.pyplot(draw_elongation_figure(elong, taille))

st.warning(
    "⚠️ **Le Cx affiché ici n'est pas une mesure directe.** C'est une interpolation "
    "log-log entre les 3 seules valeurs réellement documentées de cette famille de "
    "formes (corps de révolution), déjà citées plus haut dans l'appli : disque/plaque "
    "perpendiculaire (Cx ≈ 1,17, élancement → 0), sphère (Cx ≈ 0,47, élancement = 1) et "
    "profilé très élancé (Cx ≈ 0,04, élancement = 8). Pour tout élancement intermédiaire "
    "affiché par le curseur, le Cx est une **approximation raisonnable** entre ces trois "
    "ancrages, pas une valeur mesurée en soufflerie ou en CFD à cet élancement précis."
)

# ----------------------------------------------------------------------------
# Sources et limites
# ----------------------------------------------------------------------------

with st.expander("📚 Sources et limites du modèle"):
    st.markdown(
        "- **Densité de l'air** : loi des gaz parfaits, pression atmosphérique standard "
        "(101 325 Pa), altitude non prise en compte.\n"
        "- **Viscosité** : loi de Sutherland pour l'air.\n"
        "- **Nombre de Reynolds** : Re = ρ·v·L/µ, avec L la taille caractéristique saisie.\n"
        "- **Coefficients de traînée** : interpolés à partir de tables de points "
        "(Re, Cx) documentées, différentes pour chaque forme :\n"
    )
    for nom, s in SHAPES.items():
        st.markdown(f"  - **{nom}** ({s['taille']}) : {s['source']}")
    st.markdown(
        "\n- **Aire frontale** : pour comparer les formes à taille caractéristique égale, "
        "toutes les formes utilisent une aire frontale L² (carré/cube englobant), sauf la "
        "sphère qui utilise l'aire réelle d'un disque π/4·L². Cela permet d'isoler l'effet "
        "du **Cx** (forme) plutôt que de simplement comparer des tailles différentes.\n"
        "- **Limites connues** : le comportement à bas Reynolds (< 1000) pour le cube, la "
        "plaque et le profilé est une approximation simplifiée, non issue de mesures "
        "précises ; le cylindre est traité comme ayant une longueur égale à son diamètre. "
        "Ces points seront affinés dans une prochaine version.\n"
        "- **Explorateur d'élancement** : le Cx y est interpolé (log-log, sur l'axe "
        "élancement) entre les 3 mêmes valeurs documentées listées ci-dessus pour le "
        "disque/la plaque, la sphère et le profilé — aucune nouvelle donnée mesurée n'est "
        "utilisée. Les élancements intermédiaires ne correspondent pas à des essais "
        "réels ; c'est une approximation à affiner avec des données de soufflerie/CFD."
    )
