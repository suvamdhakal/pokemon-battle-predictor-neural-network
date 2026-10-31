import os                    
import logging 

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"        # Supress TensorFlow's INFO, WARNING, and ERROR logs

logging.getLogger("tensorflow").setLevel(logging.ERROR)

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_pokemon_data():

    rng = np.random.default_rng(7)

    X = rng.uniform(
        low=[40, 40, 40, 55],
        high=[151, 151, 151, 146],
        size=(600, 4)
    )

    attack = X[:, 0]
    defense = X[:, 1]
    speed = X[:, 2]
    opponent_power = X[:, 3]

    battle_score = (
        0.35 * attack
        + 0.30 * defense
        + 0.35 * speed
        + 0.003 * (attack - 95) * (speed - 95)
    )

    Y = (battle_score > opponent_power).astype(int)

    return X, Y.reshape(-1, 1)

X,Y = load_pokemon_data()
print(f"X.shape: {X.shape},Y.shape: {Y.shape}\n")  #dimensions of X and Y
print(f"X[:5]:\n{X[:5]},\n\nY[:5]:\n{Y[:5]}\n")      #first 5 data from X and Y

m = Y.shape[0]
win_count = 0
for i in range(m):
    if Y[i][0] == 1:
        win_count += 1
loss_count = m - win_count
print(f"Total number of wins = {win_count}")
print(f"Total number of losses = {loss_count}")

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, 
    test_size = 0.2, 
    random_state = 42, 
    stratify = Y
)

norm_l = tf.keras.layers.Normalization(axis = -1)
norm_l.adapt(X_train)

model = tf.keras.Sequential([
    norm_l,
    tf.keras.layers.Dense(5, activation = 'relu'),
    tf.keras.layers.Dense(3, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])

model.compile(
    optimizer = 'Adam',
    loss = 'binary_crossentropy'
)

history = model.fit(X_train,Y_train, epochs = 100)
loss = history.history['loss']

plt.plot(range(1, len(loss) + 1), loss)     #loss vs epoch graph
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss vs Epochs')
plt.show()

prediction_train = model.predict(X_train)
prediction_test = model.predict(X_test)

label_train = (prediction_train >=0.5).astype(int)
label_test = (prediction_test >=0.5).astype(int)

accuracy_train = np.mean(label_train == Y_train)
accuracy_test = np.mean(label_test == Y_test)

print(f"\nTrain accuracy = {accuracy_train * 100:.2f}%")
print(f"Test accuracy = {accuracy_test * 100:.2f}%\n")


W1, b1 = model.layers[1].get_weights()
W2, b2 = model.layers[2].get_weights()
W3, b3 = model.layers[3].get_weights()

print(f"\nWeight matrix of layer 1:\n{W1}\nBias vector of layer 1:\n{b1}\n\n\n")
print(f"Weight matrix of layer 2:\n{W2}\nBias vector of layer 2:\n{b2}\n\n\n")
print(f"Weight matrix of layer 3:\n{W3}\nBias vector of layer 3:\n{b3}\n\n\n")

print("Enter the following data:\n")
a = int(input("Attack: "))
b = int(input("Defence: "))
c = int(input("Speed: "))
d = int(input("Opponent Power: "))

new_inp = np.array([[a,b,c,d]])
new_prediction = model.predict(new_inp)
new_label = (new_prediction >= 0.5).astype(int)
if new_label[0][0] == 1:
    print("Your pokemon wins the fight")
else:
    print("Your pokemon loses the fight. Train it well for next time.")


