# Falak 3D

**Écrivez « une lampe de bureau », récupérez un objet 3D prêt à l'emploi. Sur une carte graphique ordinaire, sans service cloud.**

Les modèles qui génèrent de la 3D à partir de texte existent, mais ce qu'ils
produisent brut est rarement utilisable : une forme couchée sur le côté, à une
échelle arbitraire, avec des fragments qui flottent autour. Ce projet prend
[Shap-E](https://github.com/openai/shap-e), un modèle open source d'OpenAI, et
en fait un vrai service : on choisit un usage, on décrit l'objet en français,
et on récupère un fichier `.glb` propre, à la bonne taille, qui s'ouvre dans un
navigateur, Blender ou Unity.

C'est le service derrière la galerie de [falakion.com/falak3d](https://falakion.com/fr/falak3d).

![Architecture](docs/architecture.svg)

## Pour quoi faire ?

Le même modèle sert des besoins différents grâce à des **presets** :

| Preset | Pour qui | Taille de l'objet | Niveau de détail |
|---|---|---|---|
| `ecommerce` | une fiche produit, un configurateur | 35 cm | élevé |
| `packaging` | une première idée d'emballage | 25 cm | moyen |
| `jeu` | un décor de jeu vidéo | 1 m | faible (low-poly) |
| `amenagement` | une esquisse de mobilier dans une pièce | 1,2 m | élevé |

Un preset est un petit fichier YAML : en ajouter un ne demande pas de toucher
au code. C'est une esquisse, pas un modèle final : l'intérêt est d'aller vite
pour visualiser une idée.

## Ce qui se passe entre la phrase et le fichier

1. **Traduction.** Shap-E ne comprend que l'anglais. La description est
   traduite par le LLM local s'il est disponible, sinon par un petit lexique.
2. **Génération.** Shap-E produit une forme en 40 à 60 secondes sur une GTX 1060.
3. **Nettoyage.** On supprime les fragments parasites, on réduit le nombre de
   faces en gardant les couleurs, on redresse l'objet (Shap-E le couche sur le
   côté par rapport au format glTF), on le met à la taille du preset et on le
   pose au sol.
4. **Livraison.** Le fichier est disponible par l'API et s'ouvre dans la
   visionneuse fournie (`viewer/index.html`).

## Ce que j'ai appris en production

**Une carte graphique ancienne, c'est des versions à verrouiller.** La GTX 1060
n'est plus prise en charge par les versions récentes de PyTorch. Une dépendance
de Shap-E faisait monter PyTorch en silence : l'installation réussissait, et
tout plantait au premier objet. Les versions sont maintenant verrouillées, et
la construction de l'image échoue tout de suite si la carte n'est pas couverte.

**Un service qui meurt sans le dire est le pire des cas.** Sur un disque plein,
le processus qui traite la file d'attente s'arrêtait, alors que le service
répondait toujours « tout va bien ». Désormais, une erreur ne peut plus arrêter
la file, et le contrôle de santé vérifie vraiment qu'elle tourne.

**Un GPU de 6 Go pour deux modèles.** L'assistant Falak Chat et Shap-E
partagent la même carte. Avant chaque objet, le service 3D demande au LLM de
libérer la mémoire, puis le recharge quand il a fini.

Le dossier complet est dans [docs/falak-3d.pdf](docs/falak-3d.pdf).

## L'essayer

Avec Docker et une carte NVIDIA :

```bash
cp .env.example .env
docker compose up -d --build
curl -X POST localhost:8001/v1/jobs -H "X-API-Key: $FALAK3D_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "une lampe de bureau", "preset": "ecommerce"}'
```

Le premier objet est plus long : il faut télécharger les poids du modèle
(2,7 Go, gardés ensuite) et les charger.

Sans GPU, pour voir toute la chaîne avec une forme simple à la place de Shap-E :

```bash
pip install -e .
falak3d generate "une tasse" --fake -o out/tasse.glb
```

Ouvrez ensuite `viewer/index.html` et glissez-y le fichier.

## Sous le capot

- **Shap-E** et **PyTorch 2.5** pour la génération (environ 3 Go de mémoire graphique).
- **trimesh** pour nettoyer, réduire et redresser les maillages.
- **FastAPI** pour l'API : file d'attente avec position, annulation, rétention
  de 7 jours, métriques Prometheus.
- **three.js** pour la visionneuse.

```
src/falak3d/   le code (presets, traduction, nettoyage, file, API)
presets/       un fichier par usage
viewer/        la visionneuse web
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Les 17 tests remplacent Shap-E par des formes simples : ils vérifient le
nettoyage, l'orientation, l'échelle, la file d'attente et l'API sans carte
graphique.

---

Projet sous licence MIT, par Sassi Benalouache ([Falakion](https://falakion.com)).
Shap-E est distribué sous licence MIT par OpenAI.
