from pandas import read_table
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model, neural_network
from sklearn.metrics import classification_report, confusion_matrix

def download_data():

    frame = read_table("spiral.csv", encoding='utf-8', sep=',', skipinitialspace=True, header=0, names=['x', 'y', 'classy'])

    return frame


def plotSpiral(results):

    zeroes = results.loc[results['classy'] == 0]
    ones = results.loc[results['classy'] == 1]
    twos = results.loc[results['classy'] == 2]

    fig = plt.figure(figsize=(6, 6))
    fig.canvas.set_window_title('Classifying Spiral ')
    
    plt.scatter(zeroes.x, zeroes.y, label='0')
    plt.scatter(ones.x,ones.y, label='1')
    plt.scatter(twos.x,twos.y, label='2')

    plt.title('Spiral Dataset')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc='lower left')

    plt.savefig("spiral.png", bbox_inches='tight')
    plt.tight_layout()
    plt.show()
    plt.close()


# =====================================================================

def visualize_classifier(classifier, X, y):  

   
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    mesh_step_size = 0.01

    x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size),np.arange(min_y, max_y, mesh_step_size))
    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()]) 
    output = output.reshape(x_vals.shape)
    plt.figure()  
    plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray)

    plt.scatter(X[:, 0], X[:, 1], c=y, s=75, edgecolors='black',linewidth=1, cmap=plt.cm.Paired)
    plt.xlim(x_vals.min(), x_vals.max())
    plt.ylim(y_vals.min(), y_vals.max()) 
  #  plt.xticks((np.arange(int(X[:, 0].min() - 1), int(X[:, 0].max() + 1),1.0)))
  #  plt.yticks((np.arange(int(X[:, 1].min() - 1), int(X[:, 1].max() + 1),1.0)))
    plt.show()

# =====================================================================


if __name__ == '__main__':
   
    print("Loading data from Spiral.csv")
    frame = download_data()
    
    # Display spiral
    print("Plotting Spiral")
    plotSpiral(frame)

    arr = np.array(frame, dtype=np.float)
    data = np.array(arr)
    X,y = data[:,:-1], data[:,-1]

    # Use 80% of the data for training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler = StandardScaler()
    scaler.fit(X)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    sgd = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
    sgd.fit(X_train, y_train)
    sgd_predictions = sgd.predict(X_test)

    mlp = neural_network.MLPClassifier(hidden_layer_sizes=(72));
    mlp.fit(X_train,y_train)
    mlp_predictions = mlp.predict(X_test)


    print("SGD Confusion Matrix")

    print(confusion_matrix(y_test, sgd_predictions))
    print(classification_report(y_test, sgd_predictions))

    print("MLP Confusion Matrix")
    print(confusion_matrix(y_test, mlp_predictions))
    print(classification_report(y_test, mlp_predictions))

    visualize_classifier(sgd, X_test, y_test)
    visualize_classifier(mlp,X_test,y_test)
