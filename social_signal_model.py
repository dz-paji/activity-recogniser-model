import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LayerNormalization, Conv2D, MaxPooling2D, LSTM, BatchNormalization, Bidirectional, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from tensorflow import keras
import keras_tuner as kt

your_dataset_path = "model_data/social"
window_size = 20
step_size = 20

def load_files_from_folder(folder_path):
    """
    Load all CSV files from a folder and return a list of file paths.

    Parameters:
    folder_path (str): The path to the folder containing CSV files.

    Returns:
    list: A list of file paths for all CSV files in the folder.
    """

    # Initialize an empty list to store the full file paths of the CSV files
    file_paths = []

    # Loop through all the files in the given folder
    for file_name in os.listdir(folder_path):
        # Check if the file has a .csv extension (ignores other files)
        if file_name.endswith('.csv'):
            # Construct the full file path by joining the folder path and the file name
            full_file_path = os.path.join(folder_path, file_name)

            # Append the full file path to the file_paths list
            file_paths.append(full_file_path)

    # Return the complete list of CSV file paths
    return file_paths

def split_files(file_list, test_size=0.2):
    """
    Split the list of files into training and test sets.

    Parameters:
    file_list (list): List of file paths to be split into train and test sets.
    test_size (float): The proportion of files to allocate to the test set.
                       Default is 0.2, meaning 20% of the files will be used for testing.

    Returns:
    tuple:
        - train_files (list): List of file paths for the training set.
        - test_files (list): List of file paths for the test set.
    """

    # Split the file list into training and test sets using train_test_split from scikit-learn
    # test_size defines the proportion of the data to use as the test set (default is 20%)
    # shuffle=True ensures that the files are shuffled randomly before splitting
    train_files, test_files = train_test_split(file_list, test_size=test_size, shuffle=True)

    # Return the train and test file lists
    return train_files, test_files

def load_and_apply_sliding_windows(file_paths, window_size, step_size, label):
    """
    Load the data from each file, apply sliding windows, and return the windows and labels.

    Parameters:
    file_paths (list): List of file paths to CSV files. Each file contains sensor data (e.g., accelerometer, gyroscope).
    window_size (int): The size of each sliding window (number of time steps).
    step_size (int): The step size (stride) between consecutive windows.
    label (int or str): The label for the activity corresponding to the folder.
                        This label will be assigned to each sliding window extracted from the data.

    Returns:
    tuple:
        - windows (numpy.ndarray): A 3D array of sliding windows, where each window has the shape
                                   (num_windows, window_size, num_features).
        - labels (numpy.ndarray): A 1D array of labels, where each label corresponds to a sliding window.
    """
    # Initialize lists to store sliding windows and their corresponding labels
    windows = []
    labels = []

    # Loop through each file in the provided file paths
    for file_path in file_paths:
        # Load the CSV file into a pandas DataFrame
        data = pd.read_csv(file_path)

        # Select the columns containing the necessary sensor data (acceleration and gyroscope readings)
        # These columns might vary depending on your dataset's structure
        #data = data[['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']]
        # only using acceleration data
        data = data[['accel_x', 'accel_y', 'accel_z']]
        
        acc_x = data['accel_x']
        acc_y = data['accel_y']
        acc_z = data['accel_z']
        
        # Calculate the magnitude of the acceleration vector
        data['acc_magnitude'] = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        
        # Convert the DataFrame into a numpy array for faster processing in the sliding window operation
        data = data.to_numpy()
        
        # Get the number of samples (rows) and features (columns) in the data
        num_samples, num_features = data.shape

        # Apply sliding windows to the data
        # The range function defines the start of each window, moving step_size increments at a time
        for i in range(0, num_samples - window_size + 1, step_size):
            # Extract a window of size 'window_size' from the current position 'i'
            window = data[i:i + window_size, :]

            # Append the window to the windows list
            windows.append(window)

            # Assign the activity label to the window and append it to the labels list
            labels.append(label)

    # Convert the lists of windows and labels into numpy arrays for efficient numerical operations
    return np.array(windows), np.array(labels)

def process_activity(activity, label, dataset_path, window_size, step_size, test_size=0.2):
    """
    Processes an activity folder by loading the file list, splitting them into
    train and test sets, and applying sliding windows to the files.

    Args:
        activity (str): Name of the activity (folder name). This refers to the specific physical activity
                        like 'walking', 'running', etc.
        label (int): Numeric label corresponding to the activity, used for classification.
        dataset_path (str): Base path where the activity folders are located.
        window_size (int): Size of the sliding window, i.e., the number of time steps included in each window.
                           Default is 50.
        step_size (int): Step size for the sliding window, i.e., how far the window moves along the data.
                         Default is 50 (no overlap between windows).
        test_size (float): Proportion of files to use for testing. Default is 0.2, meaning 20% of files will
                           be allocated to the test set.

    Returns:
        tuple:
            - train_windows (numpy.ndarray): Sliding windows from the training files.
            - train_labels (numpy.ndarray): Corresponding labels for the training windows.
            - test_windows (numpy.ndarray): Sliding windows from the test files.
            - test_labels (numpy.ndarray): Corresponding labels for the test windows.
    """
    # Construct the full folder path where the activity files are stored
    folder_path = os.path.join(dataset_path, activity)

    # Load all CSV file paths for the given activity from the folder
    file_list = load_files_from_folder(folder_path)

    # Split the file list into training and testing sets
    # train_files: files used for training
    # test_files: files used for testing
    train_files, test_files = split_files(file_list, test_size=test_size)

    # Apply sliding windows to the training files
    # The function 'load_and_apply_sliding_windows' returns the sliding windows (segments) and their corresponding labels
    train_windows, train_labels = load_and_apply_sliding_windows(train_files, window_size, step_size, label)

    # Apply sliding windows to the testing files
    test_windows, test_labels = load_and_apply_sliding_windows(test_files, window_size, step_size, label)

    # Return the sliding windows and their labels for both training and testing sets
    return train_windows, train_labels, test_windows, test_labels

def combine_data(train_test_data, data_type):
    """
    Combines the sliding windows and labels from all activities into a single
    array for either training or testing.

    Args:
        train_test_data (dict): Dictionary containing the sliding window data for all activities.
                                Each key in the dictionary corresponds to an activity, and the value is another
                                dictionary with the keys 'train_windows', 'train_labels', 'test_windows', 'test_labels'.
        data_type (str): Either 'train' or 'test' to specify which data to combine (e.g., 'train_windows' or 'test_windows').

    Returns:
        tuple:
            - windows (numpy.ndarray): Concatenated windows from all activities for either training or testing.
            - labels (numpy.ndarray): Concatenated labels corresponding to the windows from all activities.
    """

    # Extract the list of sliding windows for the specified data type (either 'train' or 'test') from each activity
    # For example, if data_type is 'train', it extracts 'train_windows' for all activities
    windows_list = [train_test_data[activity][f'{data_type}_windows'] for activity in train_test_data]

    # Similarly, extract the list of labels corresponding to the windows for each activity
    labels_list = [train_test_data[activity][f'{data_type}_labels'] for activity in train_test_data]

    # Concatenate all the sliding windows into a single numpy array along the first axis (rows)
    # This creates one large array of windows from all the activities combined
    concatenated_windows = np.concatenate(windows_list, axis=0)

    # Concatenate all the labels into a single numpy array along the first axis (rows)
    # The labels are now aligned with the concatenated windows
    concatenated_labels = np.concatenate(labels_list, axis=0)

    # Return the concatenated windows and labels as a tuple
    return concatenated_windows, concatenated_labels

def augment_data(X_train, y_train):
    noise_factor = 0.005
    shift_factor = 2
    scale_factor = 1.5
    
    X_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
    X_shifted = np.roll(X_train, shift_factor, axis=1)
    X_scaled = X_train * scale_factor
    
    X_train_new = np.concatenate((X_train, X_noisy, X_shifted, X_scaled), axis=0)
    y_train_new = np.concatenate((y_train, y_train, y_train, y_train), axis=0)
    return X_train_new, y_train_new

def build_1d_cnn_model(hp, input_shape, num_classes):
    """
    Builds and compiles a 1D CNN model for multi-class classification.

    Args:
        input_shape (tuple): The shape of the input data (timesteps, features).
        num_classes (int): The number of output classes.

    Returns:
        model (Sequential): Compiled 1D CNN model.
    """
    model = Sequential()
        
    # First Conv1D layer
    # You can try experimenting with different filters, kernel_size values and activiation functions
    model.add(Conv1D(
        filters=128,
        kernel_size=3,
        activation='relu', input_shape=input_shape))

    
    model.add(MaxPooling1D(3))
    model.add(BatchNormalization())
    # model.add(MaxPooling1D(pool_size=3))
    # model.add(Dropout(0.4))
    
    model.add(Conv1D(filters=192, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=3))
    model.add(BatchNormalization())
    
    # for i in range(hp.Int('num_conv_layers', 1, 2, 3)):
    #     model.add(Conv1D(
    #         filters=hp.Int(f'filters_{i}', min_value=32, max_value=128, step=32),
    #         kernel_size=2,
    #         activation='relu'
    #     ))
    #     model.add(MaxPooling1D(pool_size=hp.Choice(f'pool_size_{i}', values=[2, 3])))
    
    # Flatten the output from the convolutional layers
    model.add(Flatten())

    # Fully connected layer
    # model.add(Dense(448, activation='relu'))
    model.add(Dense(
        units=96,
        activation='relu'
    ))
    
    # Dropout layer for regularization
    # You can try experimenting with different dropout rates
    model.add(Dropout(0.4))
    # model.add(Dropout(hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)))
    
    # Output layer with softmax for multi-class classification
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(
        optimizer=Adam(0.015),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    #  Prints a detailed summary of the model, showing the layers, their output shapes, and the number of trainable parameters
    model.summary()

    return model

def build_cnn_lstm_model(hp, input_shape, num_classes):
    """
    Builds and compiles an optimized CNN-BiLSTM model for multi-class classification with hyperparameter tuning.

    Args:
        hp (HyperParameters): Keras Tuner hyperparameters object.
        input_shape (tuple): The shape of the input data (timesteps, features).
        num_classes (int): The number of output classes.

    Returns:
        model (Sequential): Compiled CNN-BiLSTM model.
    """
    model = Sequential()
    # initial_kernel_size = min(3, input_shape[0])
    # First Conv1D layer with Batch Normalization
    model.add(Conv1D(
        filters=hp.Int('filters', min_value=64, max_value=256, step=32),
        kernel_size=5,
        activation='relu',
        input_shape=input_shape,
        kernel_regularizer=l2(0.001),
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(hp.Choice('pool_size', values=[2, 3, 4])))
    
    # Bidirectional GRU layer
    model.add(Bidirectional(GRU(units=hp.Int('units', min_value=64, max_value=128, step=32),
                                kernel_regularizer=l2(0.001))))
    model.add(Flatten())
    model.add(Dense(hp.Int('dense_units', min_value=32, max_value=128, step=32),
                    activation='relu',
                    kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.4))

    # Output layer with softmax for multi-class classification
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile the model
    model.compile(
        optimizer=Adam(0.00147),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    # Define activity folders and corresponding labels
    # Each key is the name of the physical activity, and the corresponding value is the numeric label
    # These labels will be used as the target variable for classification.
    activities = {
        'breathingNormally': 0,
        'coughing': 1,
        'hyperventilation': 2,
        'other': 3
    }
    
    train_test_data = {}

    # Loop through each activity folder and process the data
    # Note, if you have large amounts of data, this step may take a while
    for activity, label in activities.items():
        # Initialize an empty dictionary for each activity to store train and test windows and labels
        train_test_data[activity] = {}

    # Call process_activity() to process the data for the current activity folder
    # It loads the data, applies sliding windows, splits it into train and test sets,
    # and returns the respective sliding windows and labels for both sets.
        (train_test_data[activity]['train_windows'], train_test_data[activity]['train_labels'],
        train_test_data[activity]['test_windows'], train_test_data[activity]['test_labels']) = process_activity(
            activity, label, your_dataset_path, window_size, step_size)

# Explanation:
    # - 'train_windows' and 'train_labels' store the windows and labels from the training files.
    # - 'test_windows' and 'test_labels' store the windows and labels from the test files.
    # - `your_dataset_path` should be replaced with the actual path to your dataset.
    # - `process_activity` handles all the steps of loading data, splitting it, and applying sliding windows.
    
    # Combine the sliding windows and labels for the training data from all activities
    # The combine_data() function concatenates the windows and labels across activities
    X_train, y_train = combine_data(train_test_data, 'train')

    # Combine the sliding windows and labels for the test data from all activities
    X_test, y_test = combine_data(train_test_data, 'test')
    
    encoder = OneHotEncoder(sparse_output=False)
    y_train_one_hot = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_one_hot = encoder.transform(y_test.reshape(-1, 1))
    
    print("Training data shape:", X_train.shape)
    print("Training labels shape:", y_train_one_hot.shape)
    
    # Augment the training data
    # X_train, y_train_one_hot = augment_data(X_train, y_train_one_hot)
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = y_train_one_hot.shape[1]
    
    # model = build_1d_cnn_model(input_shape, num_classes)
    # model = build_2d_cnn_model(input_shape_2dcnn, num_classes)
    
    earlystop_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    tuner = kt.Hyperband(
        lambda hp: build_cnn_lstm_model(hp, input_shape, num_classes),
        objective='val_accuracy',
        max_epochs=200,
        factor=4,
        directory='my_dir',
        project_name='cnn_lstm_tuning'
    )

    # Perform the search
    tuner.search(X_train, y_train_one_hot, epochs=50, validation_data=(X_test, y_test_one_hot), callbacks=[earlystop_callback])

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Do k-fold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_reports = []
    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train_one_hot[train_index], y_train_one_hot[val_index]

        model = build_cnn_lstm_model(best_hps, input_shape, num_classes)
        
        # Train the model
        model.fit(X_train_fold, y_train_fold,
                epochs=200,
                batch_size=64,
                validation_data=(X_val_fold, y_val_fold),
                callbacks=[earlystop_callback])

        # Predict on the validation set
        y_val_pred_probs = model.predict(X_val_fold)
        y_val_pred_classes = np.argmax(y_val_pred_probs, axis=1)
        y_val_true_classes = np.argmax(y_val_fold, axis=1)

        # Generate the classification report for the current fold
        report = classification_report(y_val_true_classes, y_val_pred_classes, digits=4)
        fold_reports.append(report)

    # write report to file
    with open('social_kfold_report.txt', 'w') as f:
        for i, report in enumerate(fold_reports):
            f.write(f'Fold {i + 1}:\n')
            f.write(report)
            f.write('\n\n')
    
    # Build the model with the optimal hyperparameters
    model = build_cnn_lstm_model(best_hps, input_shape, num_classes)
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
    earlystop_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(X_train, y_train_one_hot,
                    epochs=200,         # Train the model for 20 epochs
                    batch_size=64,     # Use a batch size of 32
                    validation_data=(X_test, y_test_one_hot),
                    callbacks=[earlystop_callback],
                    )  # Validate on the test set after each epoch

    y_pred_probs = model.predict(X_test)

    # Convert the predicted probabilities to class labels (taking the argmax of the probabilities)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)

    # Convert the true test labels from one-hot encoding back to class labels
    y_true_classes = np.argmax(y_test_one_hot, axis=1)

    # Generate the classification report
    report = classification_report(y_true_classes, y_pred_classes, digits=4)

    # Print the classification report
    print(report)
    
    # Convert the trained Keras model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)  # model is your trained Keras model
    converter.experimental_enable_resource_variables = True
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()
    tflite_model = converter.convert()

    # Save the converted model to a .tflite file
    with open('model_social_cnn_lstm.tflite', 'wb') as f:
        f.write(tflite_model)

    print("Model successfully exported to model_social_cnn_lstm.tflite")
