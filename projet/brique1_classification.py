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
    return Path, mo, np, plt, sys, torch, tp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Projet Usine 4.0 — Brique 1 : classification

    **Groupe :** B · **Membres :** Loric, Eric, Florianne, Abdellah

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

    Cette question traite de la question de si le modèle arrive correctement à classifier les différentes erreur sur les pièces. Les données sont des images annotées par leur défaut.
    ...
    """)
    return


@app.cell
def _(Path, mo, plt, sys):
    import time

    import torch.nn as nn
    from PIL import Image
    import skimage
    from skimage import data as skdata
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    sys.path.insert(0, str(Path(mo.notebook_dir()) / "outils"))
    import tests_tp1 as tests

    plt.rcParams["figure.dpi"] = 90
    return LogisticRegression, SVC, StandardScaler, make_pipeline, nn, skimage


@app.cell
def _(mo, tp):
    # Chargement et découpage des données. Exemple pour les images NEU (voir outils/tp_utils.py) :
    # (images_train, y_train, images_val, y_val, images_test, y_test,
    #  noms) = tp.charger_neu(n_train_par_classe=30, n_val_par_classe=20, n_test_par_classe=100)
    dossier_donnees = tp.DONNEES
    try:
        img_tr, y_tr, img_val, y_val, images_test, y_test, noms = tp.charger_neu(n_train_par_classe=30, n_val_par_classe=20, n_test_par_classe=100, graine=tp.fixer_graine(0))
    except FileNotFoundError as _e:
        mo.stop(True, mo.callout(mo.md(str(_e)), kind="danger"))
    return images_test, img_tr, img_val, noms, y_test, y_tr, y_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. La baseline classique

    Une méthode sans réseau de neurones, raisonnable pour ce problème : laquelle, pourquoi, quel score ?

    ...
    """)
    return


@app.cell
def _(np):
    from skimage.feature import hog

    def descripteurs_classiques(image):
        """image : PIL.Image -> vecteur numpy de longueur 1812 (HOG + histogrammes de couleur)."""
        # TODO 2
        image = image.resize((128,128)).convert("RGB")
        img=np.array(image)
        img = img / 255.0

        hog_features = hog(
            img,
            orientations=9,
            pixels_per_cell=(16, 16), 
            cells_per_block=(2, 2),
            channel_axis=-1
        )

        color_features = []

        for channel in range(3):  # R, G, B
            hist, _ = np.histogram(
                img[:, :, channel],
                bins=16,
                range=(0, 256)
            )

            # Normaliser pour que la somme soit égale à 1
            hist = hist / hist.sum()
            color_features.extend(hist)

        color_features = np.array(color_features)

        features = np.concatenate([hog_features, color_features])

        return features
        raise NotImplementedError("À compléter")

    return


@app.cell
def _(np, skimage):
    def extraire_lbp(img):
        img = np.asarray(img)

        # Conversion RGB vers niveaux de gris
        if img.ndim == 3:
            img = skimage.color.rgb2gray(img)

        # Conversion en entier pour éviter le warning LBP
        if img.dtype != np.uint8:
            img = skimage.util.img_as_ubyte(img)

        lbp = skimage.feature.local_binary_pattern(
            img, 24, 3, method="uniform"
        )

        # Histogramme LBP plutôt que matrice aplatie
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=np.arange(27),
            range=(0, 26),
            density=True
        )

        return hist

    return (extraire_lbp,)


@app.cell
def _(
    SVC,
    StandardScaler,
    extraire_lbp,
    img_tr,
    img_val,
    make_pipeline,
    mo,
    np,
    tp,
    y_tr,
    y_val,
):
    _ds_tr = tp.DatasetImages(img_tr, y_tr, tp.transformation(224))
    _ds_va = tp.DatasetImages(img_val, y_val, tp.transformation(224))
    f_tr = np.stack([extraire_lbp(_ds_tr.image_brute(i)) for i in range(len(_ds_tr))])
    f_va = np.stack([extraire_lbp(_ds_va.image_brute(i)) for i in range(len(_ds_va))])
    # C = 10 fixé ici pour aller vite ; dans le projet, choisissez-le sur la validation comme à l'étape 5
    acc_cl = np.mean(make_pipeline(StandardScaler(), SVC(C=10)).fit(f_tr, y_tr).predict(f_va) == np.asarray(y_val))
    mo.md(f"""
    | Méthode (30 images par type) | Accuracy validation |
    |---|---|
    | descripteur de texture  + SVM | {acc_cl:.1%} |
    """)
    return f_tr, f_va


@app.cell
def _(SVC, StandardScaler, f_tr, f_va, make_pipeline, np, plt, y_tr, y_val):

    # Différentes quantités d'images d'entraînement
    _tailles = [5, 10, 15, 20, 25, 30]
    _accuracies = []

    # Nombre de classes
    _classes = np.unique(y_tr)

    for _n in _tailles:
        _indices = []

        # Sélectionner n images par classe
        for _c in _classes:
            _idx = np.where(np.asarray(y_tr) == _c)[0][:_n]
            _indices.extend(_idx)

        _indices = np.array(_indices)

        # Entraînement
        _clf = make_pipeline(
            StandardScaler(),
            SVC(C=10)
        )

        _clf.fit(f_tr[_indices], np.asarray(y_tr)[_indices])

        # Évaluation sur toute la validation
        _predictions = _clf.predict(f_va)
        _acc = np.mean(_predictions == np.asarray(y_val))

        _accuracies.append(_acc)

    # Tracé
    plt.figure(figsize=(8, 5))
    plt.plot(_tailles, _accuracies, marker='o')
    plt.xlabel("Nombre d'images d'entraînement par classe")
    plt.ylabel("Accuracy validation")
    plt.title("Évolution de l'accuracy selon le nombre d'images")
    plt.grid(True)
    plt.show()

    return


@app.cell
def _(
    SVC,
    StandardScaler,
    extraire_lbp,
    images_test,
    img_tr,
    make_pipeline,
    mo,
    np,
    tp,
    y_test,
    y_tr,
):
    # Votre baseline : remplacez cette ligne par votre code ; réglez-la sur la validation,
    # puis gardez son score sur le test, mesuré une seule fois.
    _ds_tr = tp.DatasetImages(img_tr, y_tr, tp.transformation(224))
    _f_tr = np.stack([extraire_lbp(_ds_tr.image_brute(i)) for i in range(len(_ds_tr))])
    _ds_test = tp.DatasetImages(images_test, y_test, tp.transformation(224))
    _f_test = np.stack([extraire_lbp(_ds_test.image_brute(i)) for i in range(len(_ds_test))])
    # C = 10 fixé ici pour aller vite ; dans le projet, choisissez-le sur la validation comme à l'étape 5
    score_baseline = np.mean(make_pipeline(StandardScaler(), SVC(C=10)).fit(_f_tr, y_tr).predict(_f_test) == np.asarray(y_test))
    mo.md(f"""
    | Méthode (30 images par type) | Accuracy test |
    |---|---|
    | descripteur de texture  + SVM | {score_baseline:.1%} |
    """)
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
def _(torch):
    def retirer_tete(modele):
        """Remplace la couche de classification par nn.Identity() et renvoie le modèle."""
        # TODO 1 : trouvez la tête de classification et remplacez-la par nn.Identity().
        # La même fonction doit marcher pour les trois modèles.
        if hasattr(modele, "fc"):
            modele.fc = torch.nn.Identity()
        elif hasattr(modele, "classifier"):
            modele.classifier = torch.nn.Identity()
        else:
            modele.head = torch.nn.Identity()
        return modele
        raise NotImplementedError("À compléter")

    return (retirer_tete,)


@app.function
def geler_backbone(modele, n_etages_degeles):
    """Gèle tout sauf les n derniers étages de modele.features et modele.classifier. Renvoie le modèle."""
    # TODO 3
    for p in modele.parameters():
        p.requires_grad = False
        
    if n_etages_degeles != 0:
        for feature in modele.features[-n_etages_degeles:]:
            for p in feature.parameters():
                p.requires_grad = True
            
    for p in modele.classifier.parameters():
        p.requires_grad = True
    
    return modele
    raise NotImplementedError("À compléter")


@app.cell
def _(torch):
    def creer_optimiseur(modele, lr_tete, lr_backbone):
        """Renvoie un AdamW avec un learning rate pour la tête et un autre pour le backbone dégelé."""
        # TODO 4
        param_tete = [p for p in modele.classifier.parameters() if p.requires_grad == True]
        param_backbone = [p for p in modele.features.parameters() if p.requires_grad == True]
        if param_backbone != []:
            optimiseur = torch.optim.AdamW([
                 {"params": param_tete, "lr": lr_tete}, 
                 {"params": param_backbone, "lr": lr_backbone}, 
            ], weight_decay=1e-4)
        else :
            optimiseur = torch.optim.AdamW([
                 {"params": param_tete, "lr": lr_tete}, 
            ], weight_decay=1e-4)
        return optimiseur
        raise NotImplementedError("À compléter")

    return (creer_optimiseur,)


@app.function
def entrainer_une_epoque(modele, chargeur, optimiseur, critere, appareil):
    """Entraîne le modèle sur un passage complet du chargeur. Renvoie la perte moyenne (float)."""
    # TODO 5
    perte_total = 0.0
    nb_examples = 0
    modele.train() # BatchNorm, dropout : mode entraînement
    
    for x, y in chargeur:
        x = x.to(appareil)
        y = y.to(appareil)
        
        optimiseur.zero_grad()
        perte = critere(modele(x), y)
        perte.backward()
        optimiseur.step()

        perte_total += float(perte.item()) * x.size(0)
        nb_examples += x.size(0)
             
    return perte_total / nb_examples
    raise NotImplementedError("À compléter")


@app.cell
def _(mo):
    cfg_blocs = mo.ui.slider(0, 4, value=2, label="Étages dégelés", show_value=True)
    cfg_epoques = mo.ui.slider(1, 6, value=3, label="Époques", show_value=True)
    cfg_lr_tete = mo.ui.dropdown(["1e-2", "3e-3", "1e-3"], value="3e-3", label="LR tête")
    cfg_lr_backbone = mo.ui.dropdown(["1e-3", "3e-4", "1e-4", "3e-5"], value="3e-4", label="LR backbone")
    bouton_entrainement = mo.ui.run_button(label="Lancer le fine-tuning")
    mo.vstack([mo.hstack([cfg_blocs, cfg_epoques]), mo.hstack([cfg_lr_tete, cfg_lr_backbone]), bouton_entrainement])
    return (
        bouton_entrainement,
        cfg_blocs,
        cfg_epoques,
        cfg_lr_backbone,
        cfg_lr_tete,
    )


@app.cell
def _(
    bouton_entrainement,
    cfg_blocs,
    cfg_epoques,
    cfg_lr_backbone,
    cfg_lr_tete,
    creer_optimiseur,
    img_tr,
    img_val,
    mo,
    nn,
    noms,
    np,
    torch,
    tp,
    y_tr,
    y_val,
):
    _chemin = tp.SORTIES / ("tp1_finetune_test.pt" if tp.MODE_TEST else "tp1_finetune.pt")
    mo.stop(not (tp.lance(bouton_entrainement) or _chemin.exists()), mo.md("Cliquez sur le bouton pour lancer l'entraînement."))

    def _construire(n_classes):
        _m = tp.charger_modele("efficientnet_b0")
        _m.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, n_classes))
        return _m

    _dev = tp.appareil()
    if tp.lance(bouton_entrainement):
        tp.fixer_graine(0)
        _ds_tr = tp.DatasetImages(img_tr, y_tr, tp.transformation(160, augmentation=True), pre_redim=184)
        _ds_va = tp.DatasetImages(img_val, y_val, tp.transformation(160), pre_redim=184)
        _chargeur = torch.utils.data.DataLoader(_ds_tr, batch_size=32, shuffle=True, num_workers=0)
        modele_ft = geler_backbone(_construire(len(noms)), cfg_blocs.value).to(_dev)
        _opt = creer_optimiseur(modele_ft, float(cfg_lr_tete.value), float(cfg_lr_backbone.value))
        _crit = nn.CrossEntropyLoss(label_smoothing=0.1)
        historique_ft = {"perte": [], "accuracy": [], "config": {
            "étages dégelés": cfg_blocs.value, "époques": cfg_epoques.value,
            "lr tête": cfg_lr_tete.value, "lr backbone": cfg_lr_backbone.value,
            "paramètres entraînables (M)": round(tp.nombre_parametres(modele_ft, entrainables=True) / 1e6, 2)}}
        with tp.Chrono() as _c:
            for _ep in mo.status.progress_bar(range(cfg_epoques.value), title="Fine-tuning", subtitle="une époque à la fois"):
                historique_ft["perte"].append(entrainer_une_epoque(modele_ft, _chargeur, _opt, _crit, _dev))
                _pred, _ = tp.predire(modele_ft, _ds_va)      # suivi sur la validation : le test reste intact
                historique_ft["accuracy"].append(float(np.mean(_pred == np.asarray(y_val))))
        historique_ft["durée (s)"] = round(_c.secondes)
        _etat = {k: v.detach().cpu() for k, v in modele_ft.state_dict().items()}
        torch.save({"etat": _etat, "historique": historique_ft, "n_classes": len(noms)}, _chemin)
    else:
        _sauvegarde = torch.load(_chemin, map_location="cpu")
        modele_ft = _construire(_sauvegarde["n_classes"])
        modele_ft.load_state_dict(_sauvegarde["etat"])
        historique_ft = _sauvegarde["historique"]
    modele_ft = modele_ft.to(_dev).eval()
    return (modele_ft,)


@app.cell
def _(
    LogisticRegression,
    StandardScaler,
    img_tr,
    img_val,
    make_pipeline,
    mo,
    modele_ft,
    np,
    retirer_tete,
    tp,
    y_tr,
    y_val,
):
    ds_tr = tp.DatasetImages(img_tr, y_tr, tp.transformation(224))
    ds_va = tp.DatasetImages(img_val, y_val, tp.transformation(224))
    m = retirer_tete(modele_ft)
    e_tr = tp.extraire_embeddings(m, ds_tr)
    e_va = tp.extraire_embeddings(m, ds_va)
    acc_deep = np.mean(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(e_tr, y_tr).predict(e_va) == np.asarray(y_val))
    mo.md(f"""
    | Méthode (30 images par type) | Accuracy validation |
    |---|---|
    | EfficientNet-B0 (feature extraction) + régression logistique | {acc_deep:.1%} |
    """)
    return e_tr, e_va


@app.cell
def _(
    LogisticRegression,
    StandardScaler,
    e_tr,
    e_va,
    make_pipeline,
    np,
    plt,
    y_tr,
    y_val,
):

    # Différentes quantités d'images d'entraînement
    _tailles = [5,10, 15, 20, 25, 30]
    _accuracies = []

    # Nombre de classes
    _classes = np.unique(y_tr)

    for _n in _tailles:
        _indices = []

        # Sélectionner n images par classe
        for _c in _classes:
            _idx = np.where(np.asarray(y_tr) == _c)[0][:_n]
            _indices.extend(_idx)

        _indices = np.array(_indices)

        # Entraînement
        _clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000)
        )

        _clf.fit(e_tr[_indices], np.asarray(y_tr)[_indices])

        # Évaluation sur toute la validation
        _predictions = _clf.predict(e_va)
        _acc = np.mean(_predictions == np.asarray(y_val))

        _accuracies.append(_acc)

    # Tracé
    plt.figure(figsize=(8, 5))
    plt.plot(_tailles, _accuracies, marker='o')
    plt.xlabel("Nombre d'images d'entraînement par classe")
    plt.ylabel("Accuracy validation")
    plt.title("Évolution de l'accuracy selon le nombre d'images")
    plt.grid(True)
    plt.show()

    return


@app.cell
def _(
    LogisticRegression,
    StandardScaler,
    images_test,
    img_tr,
    make_pipeline,
    mo,
    modele_ft,
    np,
    retirer_tete,
    tp,
    y_test,
    y_tr,
):
    # Votre modèle : remplacez cette ligne par votre code ; réglez-le sur la validation,
    # puis gardez son score sur le test, mesuré une seule fois.
    _ds_tr = tp.DatasetImages(img_tr, y_tr, tp.transformation(224))
    _ds_test = tp.DatasetImages(images_test, y_test, tp.transformation(224))
    _m = retirer_tete(modele_ft)
    _e_tr = tp.extraire_embeddings(_m, _ds_tr)
    _e_test = tp.extraire_embeddings(_m, _ds_test)
    score_modele = np.mean(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(_e_tr, y_tr).predict(_e_test) == np.asarray(y_test))
    mo.md(f"""
    | Méthode (30 images par type) | Accuracy test |
    |---|---|
    | EfficientNet-B0 (feature extraction) + régression logistique | {score_modele:.1%} |
    """)
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
