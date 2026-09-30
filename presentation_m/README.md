# Présentation MIT : extension RL

Présentation en anglais de Marin Hanachowicz, centrée sur son travail dans
l'extension du contrôle hamiltonien implicite à des dynamiques inconnues.
Le projet reste crédité aux quatre auteurs du rapport.

## Afficher les slides

Depuis la racine du dépôt, sous PowerShell :

```powershell
cd presentation_m
npm.cmd ci --include=dev --bin-links=true
npm.cmd run dev
```

Ouvrir <http://localhost:3030>. Les notes orales et le minutage sont accessibles
dans le mode présentateur : <http://localhost:3030/presenter/1>.
Après la première installation, `npm.cmd run dev` suffit.

Sur macOS/Linux, remplacer `npm.cmd` par `npm`.

Le port 3030 doit être libre. Le fichier `package-lock.json` fixe les dépendances.
La dépendance `playwright-chromium` permet l'export et la vérification visuelle.

```powershell
npm.cmd run build
npm.cmd run export
```

Le build est dans `dist/`. L'export PDF est dans `.qa/presentation.pdf`.
Ces sorties et les dépendances sont ignorées par Git.

## Les dix minutes

Neuf slides principales, puis quinze slides d'annexe intégrées dans le même deck.
Les durées comprennent les transitions et sont reprises dans les notes orales.
Arrêter la présentation principale à la slide 9.

| Slide | Sujet | Durée | Cumul |
| --- | --- | --- | --- |
| 1 | Sujet, identité et contribution personnelle | 0:30 | 0:30 |
| 2 | Dynamiques inconnues et contrôle implicite | 1:00 | 1:30 |
| 3 | Méthode originale : point fixe et JFB | 1:05 | 2:35 |
| 4 | Apport RL : estimation des dérivées par RLS | 1:20 | 3:55 |
| 5 | Contrôle, adjoint et entraînement | 1:30 | 5:25 |
| 6 | Benchmark Van der Pol et variantes | 0:50 | 6:15 |
| 7 | Résultat principal Hard Gain | 1:30 | 7:45 |
| 8 | Limites et protocole d'évaluation | 1:15 | 9:00 |
| 9 | Question de recherche : direction de descente | 1:00 | 10:00 |

Annexes : gradient et adjoint, RLS, exploration, baselines, résultats standard
et hard, portfolio et référence analytique, contractivité, architecture,
liquidation et contrôle stochastique, bibliographie.

La liquidation reste une application de la méthode originale et figure seulement
en annexe. Le contrôle stochastique reste une extension distincte des expériences
RL déterministes montrées ici.

## Sources et sélection

- Slides de départ : `../presentation/slides.md` et
  `../presentation/slides-part2-rl.md`.
- Rapport : `../ML_for_finance_complex_systems.pdf`, surtout §§5.1–5.8,
  figures 4–5 et annexe A.6.
- Vérification des conventions : `../jfb-for-implicit-oc/core_RL/`.
- Vérification des expériences et baselines :
  `../jfb-for-implicit-oc/examples-RL/hard_gain_vdp.py`,
  `hard_vdp_comparison.py` et `vanderpol_comparison.py`.

Les anciens contenus RL 2.1–2.8 sont condensés dans les slides 2, 4 et 5.
Le rappel de la méthode originale vient de 1.3. Les variantes de 2.9 conduisent
au résultat Hard Gain de 2.13. Les limites et perspectives reprennent 2.15–2.16.
Le PMP/HJB détaillé, les récursions RLS et les autres benchmarks sont réservés
à la discussion.

## Conventions et précisions scientifiques

Cette version utilise partout le Hamiltonien de minimisation `H = L + pᵀ f`,
avec un pas de descente `u - alpha_fp * grad_u H`, comme le code RL.
Elle ne reproduit pas les incohérences de signes du rapport.

- Les matrices ajustées décrivent la transition discrète. L'estimateur stocke
  `a = (A_disc - I) / dt` et `b = B_discᵀ / dt`. Le produit de contrôle est `b p`,
  sans transposée supplémentaire.
- L'adjoint estimé conserve le terme `p_(k+1)` dans sa récursion.
- Le costate du réseau sert à construire le contrôle. Le costate de la passe
  arrière sert à pondérer le surrogate ; ils sont distingués.
- Les équations de méthode sont affichées avec des poids de coût unitaires.
  Pour le benchmark Hard Gain, le coût terminal effectif est `5 * (x1²+x2²)`
  et les bornes sont `[-2,2]`, conformément au runner.
- La baseline BPTT est une politique MLP directe avec deux couches cachées
  de largeur 64. Elle ne différencie pas le même contrôleur implicite.
  Le rapport parle de réseau identique : ce n'est pas ce que fait le runner.
  L'écart BPTT ne mesure donc pas l'erreur JFB seule.
- Le résultat Hard Gain est celui de la figure existante : coûts moyens
  5.718 / 5.795 / 5.624 pour RLS / oracle / BPTT. L'écart relatif RLS–BPTT
  est 1.67 %, arrondi à 1.7 %. Aucune expérience n'a été relancée.
- Les initialisations d'évaluation sont nouvelles, mais huit rollouts
  d'identification sont réalisés sur le batch de test avant le scoring RLS.
  Ce résultat inclut donc une adaptation de l'estimateur à l'évaluation.
- Aucune équivalence statistique RLS–oracle ni garantie sur plusieurs graines
  n'est affirmée à partir d'une seule figure.
- Le résultat Van der Pol standard, moins favorable, est conservé : écart
  RLS–BPTT de 13.7 % sur la figure exportée.
- La descente du gradient estimé est présentée comme une conjecture.
  L'annexe distingue l'adjoint à contrôles indépendants employé dans le code
  de la différentiation exacte d'une politique dépendant de l'état.
- La contractivité exige une borne uniforme de courbure et un pas adapté.
  Le pas 0.05 des problèmes hard ne certifie pas une contraction sur toute
  la boîte de contrôle.
- L'annexe liquidation corrige le signe de la courbure du Hamiltonien complet.
  Au vrai costate `p_X = -1`, cette courbure ne démontre pas la divergence
  structurelle annoncée dans le rapport. La reformulation reste utile pour
  retirer le costate appris du terme de courbure.
- L'annexe stochastique ne déduit pas de certitude équivalente générale de
  la seule indépendance de la diffusion vis-à-vis du contrôle.

## Figures embarquées

Toutes les images référencées sont dans `public/images/` et servies sous
`/images/`. Aucune image ne dépend de l'ancien dossier de présentation.

| Image | Provenance | Utilisation |
| --- | --- | --- |
| `hard-gain-comparison.png` | Copie intacte de `presentation/public/images/rl/hardg_comparison_oracle_vs_rls.png` ; même résultat que le rapport Fig. 5 | Slide 7 |
| `standard-vdp-comparison.png` | Copie intacte de `presentation/public/images/rl/vanderpol_oracle_vs_rls.png` | Annexe A7 |
| `hard-vdp-comparison.png` | Copie intacte de `presentation/public/images/rl/hard_comparison_oracle_vs_rls.png` | Annexe A8 |
| `portfolio-report.png` | Image PNG originale embarquée dans le rapport, page 7, Fig. 4 | Annexe A10 |

Les figures ne sont pas retracées et leurs valeurs ne sont pas modifiées.
Le portfolio reprend le rapport avec sa référence analytique, plutôt que l'ancien
export sans référence. Les GIF sont omis pour garder un deck statique et compact.

## Périmètre

Le dossier est autonome. Ses figures sont volontairement embarquées pour ne pas
dépendre de la présentation complète. Le renommage du dossier source en
`presentation/` ne change ni ce deck, ni le rapport, ni le code de recherche.
