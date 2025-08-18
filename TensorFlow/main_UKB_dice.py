#%%
import os
from utils_UKB import *
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

#%%
data_folder = r'/type/your/path'
X, y, folder_names = load_data(data_folder)# X: DICOM images; y: masks
#%%
train_data, val_data, train_masks, val_masks, folders_train, folders_val = split_data(X, y, folder_names)
print(train_data.shape, val_data.shape, train_masks.shape, val_masks.shape)
#%%
batch_size = 4
train_generator=unite_gen(train_data, train_masks, batch_size, dset = "training")
val_generator=unite_gen(val_data, val_masks, batch_size, dset = "validation")
#%%
#%%
model = FCT(train_data)
#%%
# delete old logs
dirpath = Path("myunet_tflogs")
if dirpath.exists() and dirpath.is_dir():
    shutil.rmtree(dirpath)

name = "myUNet".format(time.strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard("myunet_tflogs/{}".format(name))

warmup_epoch = 8
warmup_run_epochs = 50
normal_run_epochs = 250

rlrop = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    mode='min', 
    patience=5,
    factor=.5, 
    # min_lr=1e-6, 
    min_delta=.001,
    verbose=1)

checkpoint_filepath = "weights/train/myunet.weights_combined_dice_ce.h5"
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    checkpoint_filepath,
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True
    )

earlystop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            restore_best_weights=True,
            min_delta=.001,
            patience=12)

# Define Dice loss
def dice_loss(y_true, y_pred, smooth=1e-6):
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred)
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice

def combined_loss(y_true, y_pred, dice_weight=0.25, cross_entropy_weight=0.75, smooth=1e-6):
    # Dice loss
    dice = dice_loss(y_true, y_pred, smooth)
    # Binary cross-entropy loss
    cross_entropy = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    # Combined weighted loss
    combined = dice_weight * dice + cross_entropy_weight * cross_entropy
    return combined
    

initial_lr = 1e-3
opt = tf.keras.optimizers.Adam(learning_rate = initial_lr)
#%%
# Compute the number of warmup batches
warmup_batches = warmup_epoch * len(train_data)//batch_size
# Create the Learning rate scheduler
warm_up_lr = WarmUpLearningRateScheduler(warmup_batches, init_lr=initial_lr, verbose = 1)

#Use Dice loss instead
model.compile(optimizer = opt, 
               loss = combined_loss
               )

# first training with warmup
history = model.fit(train_generator,
          steps_per_epoch = len(train_data)//batch_size, 
          epochs=warmup_run_epochs,
          callbacks=[warm_up_lr],
          )
np.save('history_warmup_bestmodel.npy',history.history)

# second training with rlrop
history = model.fit(train_generator,
          validation_data = val_generator,
          steps_per_epoch = len(train_data)//batch_size,
          validation_steps = len(val_data)//batch_size,
          epochs=normal_run_epochs,
          callbacks = [rlrop, checkpoint_callback, tensorboard_callback],
          )


np.save('history_rlrop_bestmodel.npy',history.history)

# Make Predictions
predicted = model.predict(val_data, batch_size=batch_size)
print("Predicted Shape:", predicted.shape)

# Ensure the prediction shape matches the expected output
if len(predicted) == 3:
    predicted = predicted[-1]  # Use the highest resolution output from Deep Supervision
print("Highest Resolution Prediction Shape:", predicted.shape)

predicted_classes = (predicted > 0.5).astype(np.uint8)  # Threshold at 0.5
print("Predicted Classes Shape:", predicted_classes.shape)

save_predictions(val_data, val_masks, predicted_classes, 'predictions',folders_val)

# Calculate Metrics
metrics = np.round(np.array(metrics_binary(val_masks[..., 0], predicted_classes[..., 0], 0)), 4)
print("Dice loss:", metrics)

