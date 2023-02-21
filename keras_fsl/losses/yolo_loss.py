import tensorflow as tf


def yolo_loss(anchors, threshold):
    """

    Args:
        anchors (pandas.DataFrame): dataframe of the anchors with width and height columns.
        threshold:

    """

    def _yolo_loss(y_true, y_pred):
        """
        y_true and y_pred are (batch_size, number of boxes, 4 (+ 1) + number of classes (+ anchor_id for y_pred)).
        The number of boxes is determined by the network architecture as in single-shot detection one can only predict
        grid_width x grid_height boxes per anchor.
        """
        # 1. Find matching anchors: the anchor with the best IoU is chosen for predicting each true box
        y_true_broadcast = tf.expand_dims(y_true, axis=2)
        y_true_broadcast.shape
        y_true_broadcast[..., 2:4].shape

        anchors_tensor = tf.broadcast_to(anchors[["height", "width"]].values, [1, 1, len(anchors), 2])
        anchors_tensor.shape

        height_width_min = tf.minimum(y_true_broadcast[..., 2:4], anchors_tensor)
        height_width_max = tf.maximum(y_true_broadcast[..., 2:4], anchors_tensor)
        height_width_min.shape
        height_width_max.shape
        intersection = tf.reduce_prod(height_width_min, axis=-1)
        intersection.shape
        true_box_area = tf.reduce_prod(y_true_broadcast[..., 2:4], axis=-1)
        true_box_area.shape
        anchor_boxes_area = tf.reduce_prod(anchors_tensor, axis=-1)
        anchor_boxes_area.shape
        union = true_box_area + anchor_boxes_area - intersection
        union.shape
        iou = intersection / union
        iou.shape
        best_anchor = tf.math.argmax(iou, axis=-1)
        best_anchor.shape
        best_anchor[0, 0]

        batch_size, boxes, _ = tf.shape(y_true)
        # 2. Find grid cell: for each selected anchor, select the prediction coming from the cell which contains the true box center
        for image in range(batch_size):
            for box in range(boxes):
                true_box_info = y_true[image, box]
                selected_anchor = tf.cast(best_anchor[image, box], y_pred.dtype)
                prediction_for_anchor = tf.boolean_mask(y_pred[image], y_pred[image, :, -1] == selected_anchor, axis=0)
                prediction_for_anchor.shape
                grid_size = prediction_for_anchor
        y_pred[..., -1].shape == best_anchor
        y_pred.shape

        # 3. For confidence loss: for each selected anchor, compute confidence loss for boxes with IoU < threshold
        non_empty_boxes_mask = tf.cast(tf.math.reduce_prod(y_true[..., 2:4], axis=-1) > 0, tf.bool)
        pass

    return _yolo_loss
