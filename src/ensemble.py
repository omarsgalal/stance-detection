from collections import Counter

def majority_voting(predictions):
    """
    Performs majority voting among the predictions of three models.

    Args:
    predictions: A list of lists, where each sublist contains the predictions of one model.

    Returns:
    The majority-voted prediction.
    """
    # Transpose the predictions so that each row represents predictions from one data point
    transposed_predictions = zip(*predictions)
    
    # Perform majority voting for each data point
    majority_voted_predictions = [Counter(pred).most_common(1)[0][0] for pred in transposed_predictions]
    
    return majority_voted_predictions