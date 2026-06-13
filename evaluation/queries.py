# Evaluation query set for the AGORA paper.
# 8 questions spanning different domains and question types (discovery vs. analytical).
# "discovery"  = find relevant datasets / describe what is available
# "analytical" = compute, filter, rank, or aggregate over dataset contents

QUERIES = [
    {
        "id": "Q1",
        "question": "Quels jeux de données sont disponibles sur la qualité de l'air en France ?",
        "type": "discovery",
        "domain": "environment",
    },
    {
        "id": "Q2",
        "question": "Quelles données sur les accidents de la route en France sont disponibles sur data.gouv.fr ?",
        "type": "discovery",
        "domain": "transport",
    },
    {
        "id": "Q3",
        "question": "Combien de stations de vélos en libre-service existe-t-il à Paris ?",
        "type": "analytical",
        "domain": "transport",
    },
    {
        "id": "Q4",
        "question": "Quels sont les jeux de données disponibles sur le budget des communes françaises ?",
        "type": "discovery",
        "domain": "finance",
    },
    {
        "id": "Q5",
        "question": "Quelles sont les communes françaises ayant une population supérieure à 100 000 habitants ?",
        "type": "analytical",
        "domain": "demographics",
    },
    {
        "id": "Q6",
        "question": "Quels jeux de données concernent les établissements scolaires en France ?",
        "type": "discovery",
        "domain": "education",
    },
    {
        "id": "Q7",
        "question": "Quels sont les jeux de données sur la production d'énergies renouvelables en France ?",
        "type": "discovery",
        "domain": "energy",
    },
    {
        "id": "Q8",
        "question": "Combien de bornes de recharge pour véhicules électriques sont référencées en France ?",
        "type": "analytical",
        "domain": "energy",
    },
]
