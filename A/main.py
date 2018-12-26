import keras
import grapher
import CNN
import linear
import time

# Load data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Build and compile networks
linearModel = linear.Linear(x_train, y_train, x_test, y_test)
CNNModel = CNN.CNN(x_train, y_train, x_test, y_test)

# Time linear
start = time.time()
linearModel.train()
linearTime=time.time()

# Time CNN
CNNModel.train()
CNNTime = time.time()

# Graph results over epochs
grapher.plotResults(linearModel.history,CNNModel.history)


# Print metrics
loss_and_metrics = linearModel.model.evaluate(linearModel.x_test, linearModel.y_test, verbose=2)
loss_and_metrics_Neural = CNNModel.model.evaluate(CNNModel.x_test, CNNModel.y_test, verbose=2)

print("Linear:")
print("Test Loss", loss_and_metrics[0])
print("Test Accuracy", loss_and_metrics[1])
print("Linear Time", linearTime-start)

print("Neural:")
print("Test Loss", loss_and_metrics_Neural[0])
print("Test Accuracy", loss_and_metrics_Neural[1])
print("CNN Time", CNNTime-start)

