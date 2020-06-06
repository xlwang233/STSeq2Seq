import numpy as np
import torch


def masked_mse_torch(preds, labels, null_val=np.nan):
    """
    Accuracy with masking.
    :param preds:
    :param labels:
    :param null_val:
    :return:
    """
    if np.isnan(null_val):
        # mask = ~tf.is_nan(labels)
        mask = ~torch.isnan(labels)
    else:
        # mask = tf.not_equal(labels, null_val)
        mask = torch.ne(labels, null_val)
    # mask = tf.cast(mask, tf.float32)
    mask = mask.to(torch.float32)
    # mask /= tf.reduce_mean(mask)
    mask /= torch.mean(mask)
    # mask = tf.where(tf.is_nan(mask), tf.zeros_like(mask), mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    # loss = tf.square(tf.subtract(preds, labels))
    loss = torch.pow(preds - labels, 2)
    loss = loss * mask
    # loss = tf.where(tf.is_nan(loss), tf.zeros_like(loss), loss)
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mae_torch(preds, labels, null_val=np.nan):
    """
    Accuracy with masking.
    :param preds:
    :param labels:
    :param null_val:
    :return:
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = torch.ne(labels, null_val)
    mask = mask.to(torch.float32)
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse_torch(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse_torch(preds=preds, labels=labels, null_val=null_val))


def masked_mape_torch(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = torch.ne(labels, null_val)
    mask = mask.to(torch.float32)
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    labels = torch.where(torch.le(labels, 1e-4), torch.ones_like(labels), labels)
    loss = torch.abs((preds - labels) / labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mse_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        rmse = np.square(np.subtract(preds, labels)).astype('float32')
        rmse = np.nan_to_num(rmse * mask)
        return np.mean(rmse)


def masked_mae_np(preds, labels, null_val=0.0):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        return np.mean(mae)


def masked_rmse_np(preds, labels, null_val=0.0):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels, null_val=null_val))


def masked_mape_np(preds, labels, null_val=0.0):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)


# Builds loss function.
def masked_mse_loss(scaler, null_val):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        return masked_mse_torch(preds=preds, labels=labels, null_val=null_val)

    return loss


def masked_mae_loss(scaler, null_val):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mae = masked_mae_torch(preds=preds, labels=labels, null_val=null_val)
        return mae

    return loss


def masked_mse_loss(scaler, null_val):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        return masked_mse_torch(preds=preds, labels=labels, null_val=null_val)

    return loss
