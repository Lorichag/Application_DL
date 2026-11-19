import marimo

__generated_with = "0.25.1"
app = marimo.App(
    width="medium",
    app_title="Usine 4.0 - Brique à nommer",
    auto_download=["html"],
)


@app.cell(hide_code=True)
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    # Le dossier outils/ est à la racine du dépôt, un niveau au-dessus de projet/.
    sys.path.insert(0, str(Path(mo.notebook_dir()).parent / "outils"))
    import tp_utils as tp

    plt.rcParams["figure.dpi"] = 90
    return mo, np, torch, tp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Projet Usine 4.0 — Brique 4 : Défauts rares

    **Groupe :** B · **Membres :** Loric HAGARD, Florianne PAN, Abdellah DJEDIDI, Junjie (Eric) SHAO

    **Mode d'emploi de ce modèle**

    - Copiez ce fichier dans `projet/` (par exemple sous le nom indiqué à la fin du TD,
      `brique1_classification.py` ; le sujet est dans `projet/SUJET.md`),
      puis ouvrez la copie : `marimo edit projet/brique1_classification.py`.
    - Les fonctions fournies restent accessibles par `tp.` (chargement des données, affichage, cache).
      Les fonctions que vous avez écrites dans un TD ne s'importent pas d'un notebook à l'autre : copiez dans
      une cellule de cette brique celles dont vous avez besoin.
    - Dans une cellule, une variable dont le nom commence par un tiret bas reste locale à la cellule.
    - Fixez les graines (`tp.fixer_graine(0)`) et gardez les résultats des calculs longs
      (`tp.sauver_cache`, `tp.charger_cache`) pour ne pas les relancer à chaque ouverture.
    - Ne réglez rien sur le jeu de test : gardez une partie des données pour la validation, et ne mesurez sur
      le test qu'une fois vos choix faits.
    - Remplacez les `...` par votre travail et supprimez ce mode d'emploi avant le rendu.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. La question et les données

    Quelle question métier cette brique traite-t-elle ? Quelles données, en quelle quantité, comment
    sont-elles découpées (entraînement, validation, test) ?

    Cette brique traite la question des défauts rares par exemple dans le cas où notre modèle serait confronté à un défaut qu'il a peu vu lors de son entrainement.
    ...
    """)
    return


@app.cell
def _(mo, tp):
    # Chargement et découpage des données. Exemple pour les images NEU (voir outils/tp_utils.py) :
    # (images_train, y_train, images_val, y_val, images_test, y_test,
    #  noms) = tp.charger_neu(n_train_par_classe=30, n_val_par_classe=20, n_test_par_classe=100)
    dossier_donnees = tp.DONNEES
    try:
        hist, y_hist, req, y_req, noms = tp.charger_neu(n_train_par_classe=60, n_test_par_classe=10)
    except FileNotFoundError as _e:
        mo.stop(True, mo.callout(mo.md(str(_e)), kind="danger"))
    return hist, noms, req, y_hist, y_req


@app.cell
def _(np):
    def similarite_cosinus(A, B):
        """A : (n, d), B : (m, d) -> tableau (n, m) des similarités cosinus."""
        A_norm = A / (np.linalg.norm(A,axis=1, keepdims=True))
        B_norm = B / (np.linalg.norm(B,axis=1, keepdims=True))

        return A_norm @ B_norm.T

    return (similarite_cosinus,)


@app.cell
def _(np, similarite_cosinus):
    def rechercher(emb_requete, emb_corpus, k=5):
        """Indices des k éléments du corpus les plus proches de la requête (similarité cosinus décroissante)."""
        valeurs = similarite_cosinus(emb_requete[None, :],emb_corpus)
        valeurs = np.argsort(-valeurs[0])
        return valeurs[:k]

    return (rechercher,)


@app.cell
def _(np):
    def precision_a_k(indices, pertinents, k):
        """Proportion d'images pertinentes parmi les k premiers résultats."""
        k_indices=indices[:k]
        precision = np.sum(pertinents[k_indices]) / k
        return precision

    return (precision_a_k,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. La baseline classique

    Une méthode sans réseau de neurones, raisonnable pour ce problème : laquelle, pourquoi, quel score ?

    ...
    """)
    return


@app.cell
def _(hist, mo, noms, np, precision_a_k, rechercher, req, tp, y_hist, y_req):
    _h_hist = np.stack([tp.descripteur_pixels(tp.Image.open(p)) for p in hist])
    _h_req = np.stack([tp.descripteur_pixels(tp.Image.open(p)) for p in req])
    _lignes = []
    for _k, _nom in enumerate(noms):
        _sel = np.flatnonzero(np.asarray(y_req) == _k)
        _pert = np.asarray(y_hist) == _k
        _lignes.append({
            "type de défaut": _nom,
            "précision à 5, classique (pixels + histogrammes)": round(float(np.mean([precision_a_k(rechercher(_h_req[i], _h_hist, 5), _pert, 5) for i in _sel])), 2),
        })
    mo.vstack([mo.md("Historique : 60 images par type. Requêtes : 10 images par type. Une réponse est pertinente si elle montre le même type de défaut."),
               tp.tableau(_lignes, mo)])
    return


@app.cell
def _():
    # Votre baseline : remplacez cette ligne par votre code ; réglez-la sur la validation,
    # puis gardez son score sur le test, mesuré une seule fois.
    score_baseline = None
    return (score_baseline,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. L'approche par deep learning

    Quel modèle, quelle stratégie d'adaptation, quels réglages essayés (comparés sur la validation) ?

    ...
    """)
    return


@app.cell
def _(hist, mo, noms, np, precision_a_k, rechercher, req, tp, y_hist, y_req):
    clip_modele, clip_pretraitement,clip_tokenizer = tp.charger_clip()
    _e_hist = tp.encoder_images(clip_modele, clip_pretraitement, hist)
    _e_req = tp.encoder_images(clip_modele, clip_pretraitement, req)
    _lignes = []
    for _k, _nom in enumerate(noms):
        _sel = np.flatnonzero(np.asarray(y_req) == _k)
        _pert = np.asarray(y_hist) == _k
        _lignes.append({
            "type de défaut": _nom,
            "précision à 5, Clip": round(float(np.mean([precision_a_k(rechercher(_e_req[i], _e_hist, 5), _pert, 5) for i in _sel])), 2),
        })
    mo.vstack([mo.md("Historique : 60 images par type. Requêtes : 10 images par type. Une réponse est pertinente si elle montre le même type de défaut."),
               tp.tableau(_lignes, mo)])
    return


@app.cell
def _(np, torch):
    from PIL import Image

    def encoder_images_dino(modele, pretraitement, images, taille_lot: int = 32, progression=None) -> np.ndarray:
        """Calcule les embeddings d'une liste d'images avec CLIP ou DINOv2."""
        device = next(modele.parameters()).device
        lots = range(0, len(images), taille_lot)
        lots = lots if progression is None else progression(lots)
        sorties = []

        modele.eval()

        for i in lots:
            lot = [
                im if isinstance(im, Image.Image) else Image.open(im)
                for im in images[i:i + taille_lot]
            ]

            tenseurs = torch.stack([
                pretraitement(im.convert("RGB")) for im in lot
            ]).to(device)

            with torch.no_grad():
                if hasattr(modele, "encode_image"):
                    embeddings = modele.encode_image(tenseurs)
                else:
                    embeddings = modele(tenseurs)

            sorties.append(embeddings.float().cpu())

        return torch.cat(sorties).numpy()

    return (encoder_images_dino,)


@app.cell
def _(
    encoder_images_dino,
    hist,
    mo,
    noms,
    np,
    precision_a_k,
    rechercher,
    req,
    tp,
    y_hist,
    y_req,
):
    dino_modele, dino_pretraitement = tp.charger_dino()
    _e_hist = encoder_images_dino(dino_modele, dino_pretraitement, hist)
    _e_req = encoder_images_dino(dino_modele, dino_pretraitement, req)
    _lignes = []
    for _k, _nom in enumerate(noms):
        _sel = np.flatnonzero(np.asarray(y_req) == _k)
        _pert = np.asarray(y_hist) == _k
        _lignes.append({
            "type de défaut": _nom,
            "précision à 5, Dino": round(float(np.mean([precision_a_k(rechercher(_e_req[i], _e_hist, 5), _pert, 5) for i in _sel])), 2),
        })
    mo.vstack([mo.md("Historique : 60 images par type. Requêtes : 10 images par type. Une réponse est pertinente si elle montre le même type de défaut."),
               tp.tableau(_lignes, mo)])
    return


@app.cell
def _():
    # Votre modèle : remplacez cette ligne par votre code ; réglez-le sur la validation,
    # puis gardez son score sur le test, mesuré une seule fois.
    score_modele = None
    return (score_modele,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Résultats et analyse

    Tableau comparatif (mêmes données, même métrique pour toutes les méthodes), cas d'échec, seuil ou réglage
    retenu et sa justification en termes de coût métier.

    ...
    """)
    return


@app.cell
def _(mo, score_baseline, score_modele, tp):
    # Une ligne par méthode, la même métrique pour toutes ; ajoutez ensuite vos figures et vos cas d'échec.
    tableau_final = [
        {"méthode": "baseline classique", "score sur le test": score_baseline},
        {"méthode": "deep learning", "score sur le test": score_modele},
    ]
    tp.tableau(tableau_final, mo)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Fiche de décision

    | Rubrique | Votre réponse |
    |---|---|
    | Problème métier | ... |
    | Baseline classique et son score | ... |
    | Modèle retenu et pourquoi | ... |
    | Métrique principale et score obtenu | ... |
    | Seuil ou réglage retenu, justification métier | ... |
    | Coût (annotation, entraînement, inférence, matériel) | ... |
    | Principaux cas d'échec | ... |
    | Le deep learning est-il justifié ici ? | ... |
    | Déployable en l'état ? Sinon, que manque-t-il ? | ... |
    """)
    return


if __name__ == "__main__":
    app.run()
