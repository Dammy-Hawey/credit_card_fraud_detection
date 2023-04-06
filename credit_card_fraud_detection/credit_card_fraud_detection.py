# Import the necessary libraries
import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the credit card dataset
dataset = pd.read_csv('C:\Users\hp\Desktop\my_forders\cctv\project\creditcard.csv')

# Split the dataset into training and testing sets
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the CNN model
cnn_model = tf.keras.models.Sequential([
    tf.keras.layers.Reshape((30, 1), input_shape=(30,)),
    tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the CNN model on the training set
cnn_model.fit(X_train.values.reshape(-1, 30, 1), y_train.values, epochs=5, batch_size=64)

# Define the FFNN model
ffnn_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, input_dim=30, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
ffnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the FFNN model on the training set
ffnn_model.fit(X_train, y_train.values, epochs=5, batch_size=64)

# Define the GUI function
def gui():
    # Create a new window
    window = tk.Tk()
    window.title("Credit Card Fraud Detection System")

    # Create a label and entry for the credit card number
    cc_num_label = tk.Label(window, text="Enter credit card number:")
    cc_num_label.pack()
    cc_num_entry = tk.Entry(window)
    cc_num_entry.pack()

    # Create a button to predict whether the credit card is fraudulent or not
    def predict_fraud():
        cc_num = cc_num_entry.get()
        if len(cc_num) == 0:
            messagebox.showwarning("Warning", "Please enter a credit card number.")
        elif len(cc_num) != 30:
            messagebox.showwarning("Warning", "Invalid credit card number. Please enter a 30-digit number.")
        else:
            cc_array = np.array([list(cc_num)], dtype=np.float64)
            cnn_prediction = cnn_model.predict(cc_array.reshape(-1, 30, 1))[0][0]
            ffnn_prediction = ffnn_model.predict(cc_array)[0][0]
            if cnn_prediction >= 0.5 or ffnn_prediction >= 0.5:
                messagebox.showwarning("Prediction", "This credit card is fraudulent.")
            else:
                messagebox.showinfo("Prediction", "This credit card is not fraudulent.")

    predict_button = tk.Button(window, text="Predict", command=predict_fraud)
    predict_button.pack()

    # Run the window
    window.mainloop()

# Call the GUI function
gui()

