def generate_recommendations(advisory):
    recommendations = []
    actions = []

    rain = advisory.get("cumulative_rain_next_7_days_mm")
    fungal = advisory.get("fungal_risk")
    irrigation = advisory.get("irrigation", {}).get("recommend")

    # Irrigation
    if irrigation:
        recommendations.append("Provide supplemental irrigation.")
        actions.append("Irrigate within the next 1–2 days.")
    else:
        recommendations.append("Avoid irrigation as sufficient rainfall is expected.")

    # Disease
    if fungal == "high":
        recommendations.append("High fungal disease risk detected.")
        actions.append("Inspect crops and apply preventive fungicide if needed.")
    else:
        recommendations.append("Fungal disease risk is low under current conditions.")

    # Rain management
    if rain is not None:
        if rain > 50:
            actions.append("Ensure field drainage to prevent waterlogging.")
        elif rain < 10:
            actions.append("Monitor soil moisture for early signs of stress.")

    # 🔒 GUARANTEED fallback
    if not actions:
        actions.append("Continue regular field monitoring. No immediate action required this week.")

    return recommendations, actions
