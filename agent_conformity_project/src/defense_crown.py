def crown_guard(initial_prediction, initial_confidence, attacked_prediction, attacked_confidence):
    """
    Simple CROWN defense:
    If the attacked answer differs from the initial private answer,
    keep the initial private answer as the defended prediction.
    """
    if initial_prediction is None:
        return attacked_prediction, attacked_confidence

    if attacked_prediction is None:
        return initial_prediction, initial_confidence

    if attacked_prediction != initial_prediction:
        return initial_prediction, initial_confidence

    return attacked_prediction, attacked_confidence