# c2-compute — calcul pour un projet de mathématiques pures

**Ce que c'est.** Un sous-ensemble (scripts seulement) du dépôt privé de recherche `amazing-source/conway99-c2-experimental`.
Le projet est de la combinatoire algébrique. On cherche à montrer qu'**aucun graphe fortement régulier srg(99,14,1,2)** (le
« problème du graphe de Conway à 99 sommets ») n'admet d'involution fixant exactement un sommet.

**Méthode.**
- Les scripts énumèrent des configurations finies (squelettes, orientations).
- Pour chacune, ils résolvent des programmes linéaires (HiGHS), dont l'infaisabilité est certifiée par des certificats de
  Farkas vérifiés en arithmétique exacte.

Rien ici ne touche un réseau, un système ou un tiers : c'est uniquement du calcul numérique local (numpy, scipy, highspy,
networkx).

**Usage** (depuis une session cloud) :
```
bash scripts/cloud_worker.sh setup
nohup bash scripts/cloud_worker.sh run logs/cloud_tasks_<nom>.txt > /dev/null 2>&1 &
bash scripts/cloud_worker.sh status logs/cloud_tasks_<nom>.txt      # une ligne : fait/total + dernier résultat
bash scripts/cloud_worker.sh push claude/<branche> "message"        # pousse les petits fichiers de résultats
```
Les résultats (`logs/cloud_stage1/*.json`) sont relus et revérifiés dans le dépôt de recherche.

Dans `scripts/vendor_structures/`, le modèle par paires d'une autre session du même projet (voir le README du dossier).
