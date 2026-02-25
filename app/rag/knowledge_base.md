# Retail Forecast — Knowledge Base (RAG)

## Tables

### `sales`
Données transactionnelles / agrégées utilisées pour l'analyse.

Colonnes (principales):
- `date` : date (jour)
- `region` : région
- `macro_category` : catégorie produit (macro)
- `sales` : montant des ventes
- `payment_type` : type de paiement (si présent)
- `customer_id` : identifiant client (si présent)

## KPI definitions
- **Total sales**: somme de `sales` sur une période.
- **Top regions**: régions triées par total sales décroissant.

## Interpretation guidelines
Quand une région performe mieux:
- mix de catégories (catégories à panier moyen plus élevé)
- effet calendrier (périodes de promo, saisonnalité)
- volume client (plus de clients actifs)

> Cette base sert à donner du contexte au LLM pour expliquer les résultats SQL.
