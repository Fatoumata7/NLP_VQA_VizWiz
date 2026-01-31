# Multimodal NLP: Visual Question Answering avec VizWiz dataset

## Dataset choisie

Le jeu de données choisie est [VizWiz](https://vizwiz.org/tasks-and-datasets/vqa/), un corpus de questions-réponses visuelles conçu pour soutenir le développement de technologies d’assistance destinées aux personnes aveugles. Il a été établit dans un contexte de défi d’intelligence artificielle dont l’objectif était de concevoir des algorithmes capables de répondre à des questions visuelles formulées par des personnes aveugles.

Les données proviennent d’un contexte réel et naturel d’usage : des personnes aveugles ont capturé des images de leur environnement quotidien, puis ont enregistré oralement des questions portant sur ces images. Chaque question visuelle est associée à dix réponses fournies par des annotateurs humains, reflétant la variabilité et l’ambiguïté inhérentes aux scènes du monde réel.

Dnas ce dataset, chaque échantillon contient :
- "image" : une **image**
- "question" : une **question**
- "answers" : **10 réponses humaines** (de 10 annotateurs différents), chacune contenant la réponse définie par l'annotateur "answer" (réponse ou "unanswerable") et sa confiance dans cette réponse "answer_confidence" (yes/no/maybe).
- "answer_type" : type de question -> "other", "yes/no", "unanswerable", ou "number"
- "answerable" : booléen qui indique si cet échantillon possède une réponse, à partir des réponses founies par les annotateurs

## Problématique choisie

Le défi VizWiz reposait sur deux tâches principales :
1. La prédiction de la réponse à une question visuelle donnée (Visual Question Answering, VQA).
2. La détection des questions sans réponse, c’est-à-dire l’identification des cas où l’image ne permet pas de répondre de manière fiable à la question posée.

Dans notre étude, on se focalise sur le premier objectif : **la prédiction de la réponse à une question visuelle donnée**. Le dataset sera donc filtré pour ne conserver que les instances où une réponse est effectivement fournie ("answerable" == 1). 

## Solution appliquée

L'idée ici est d'utiliser le modèle CLIP (Contrastive Language-Image Pre-Training) pour réaliser une tâche de **Visual Question Answering (VQA)** formulée ici comme un **problème d’apprentissage contrastif** : le modèle doit sélectionner la bonne réponse parmi un ensemble de réponses candidates.

### Assumptions

**Définition du label .** On définit le **label** comme la **réponse majoritaire** parmi les 10 annotations. En cas d’égalité, une réponse est **choisie aléatoirement** parmi les ex æquo.

**Réponses candidates .** Pour chaque échantillon, on fixe **20 réponses candidates** :
- **1 réponse positive** : le label
- **19 réponses négatives** : tirées aléatoirement parmi toutes les réponses du dataset, hors label

**Objectifs .** Le modèle est entraîné par **apprentissage contrastif** :
- maximiser la similarité image–question–réponse correcte,
- minimiser la similarité avec les réponses incorrectes.

**Évaluation .** À l’inférence, le modèle choisit la réponse ayant le **score de similarité le plus élevé** parmi les 20 candidates.

### Approches testées

Trois configurations sont testées :

- **CLIP Init** CLIP dans sa version initial (sans finetuning)

- **CLIP Finetuned** CLIP finetuné avec 400 exemples du slit train sur le dataset initial

- **CLIP Finetuned on Reform Q** CLIP finetuné avec 400 exempls du split train sur le dataset preprocessé en changeant les questions initiales par des questions reformulées par un LLM. Les questions reformulées peuvent être retrouvées dans [`reform_val.json`](./data/annotations/reform_val.json) (attribut ajouté "reform_question" par rapport au fichier d'annotation initial [`val.json`](./data/annotations/val.json)).

## Résultats obtenus

| Configuration | Accuracy | Observation |
| --- | --- | --- |
| **CLIP Init** | **0.46** | En utilisant la version initiale de CLIP, sans réentraînement, le modèle réussissait à prédire la bonne réponse parmi les 20 proposées dans la moitié des cas. |
| **CLIP Finetuned** | **0.61** | Le finetuning du modèle avec apprentissage contrastif améliore considérablement sa précision. Pour les prédictions incorrectes, une observation qualitative est que les bonnes prédictions sont mieux classées que dans CLIP Init. Cela suggère qu’un finetuning sur un ensemble de données plus large permettrait probablement d’obtenir de meilleurs résultats. |
| **CLIP Finetuned on Reform Q** | **0.68** | Une observation qui a conduit à cette expérience est que les questions du jeu de données contiennent des répétitions et des informations inutiles (comme des formules de politesse : Please, Thank you, etc.). L’idée était donc de prétraiter ces questions à l’aide du LLM open-source Mistral-7B pour améliorer leur formulation. On remarque que la reformulation des questions permet un gain en précision. |

Toutes les expériences et leurs explications sont disponibles dans le notebook [`NLP_VQA_Project.ipynb`](./NLP_VQA_Project.ipynb). Une démo est proposée pour analyser les performances des trois approches sans avoir à réentraîner les modèles. Les poids des modèles sont disponibles dans le dossier [`weights`](./weights).  L’utilisation de la démo est décrite dans la section suivante.


## Demo Launch

## Prerequisites
- **Python**: 3.12  
- **Libraries**: `numpy`, `matplotlib`, `loguru`, `torch`, `transformers`


## How to Use

1. Cloner le repository du projet.
2. Décompresser `weights.zip` et le placer dans le dossier [projet](./).
3. Décompresser `data.zip` et le placer dans le dossier [projet](./) (le dossier [data](./data) initialement présent dans le repository peut être supprimer).
2. Installer les dépendances :  
   ```bash
   pip install loguru numpy matplotlib torch transformers
   ```
3. Lancer the notebook [`VQA_Demo.ipynb`](./VQA_Demo.ipynb).

---

Note : Pour lancer [`NLP_VQA_Project.ipynb`](./NLP_VQA_Project.ipynb) il faut en plus télécharger les fichiers de données complet de VizWiz : folder `train`/`val` qui contients les images, et fichier d'annotations  `train.json` disponible sur ce [lien](https://vizwiz.org/tasks-and-datasets/vqa/).



## Additional information
- **Author**: Fatoumata WADIOU 
- **Last updated**: January 31, 2026  
- **Version**: 1.0