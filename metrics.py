import tensorflow.compat.v1 as tf

def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking.
    Args:
        preds: Predictions from the model.
        labels: True labels.
        mask: Mask to exclude certain elements from the loss calculation.
        
    Returns:
        Weighted mean cross-entropy loss.
    """
    print(preds)
    # Calculate softmax cross-entropy loss
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    # Apply mask and normalize by the mean of the mask
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    # Return the weighted mean loss
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking.
    
    Args:
        preds: Predictions from the model.
        labels: True labels.
        mask: Mask to exclude certain elements from the accuracy calculation.
        
    Returns:
        Weighted mean accuracy.
    """
    # Check if predictions match true labels
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    
    # Convert correct predictions to float32
    accuracy_all = tf.cast(correct_prediction, tf.float32)
     # Apply mask and normalize by the mean of the mask
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    # Return the weighted mean accuracy
    return tf.reduce_mean(accuracy_all)
