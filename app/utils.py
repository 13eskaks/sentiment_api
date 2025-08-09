def polarity_to_stars(positivity: float, negativity: float, neutrality: float) -> float:
    stars = positivity * 5 + negativity * 1 + neutrality * 3
    return round(stars, 2)
