import matplotlib.pyplot as plt

def plot_results(history, neuralHistory):

    plt.subplot(2, 1, 1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.plot(neuralHistory.history['acc'])
    plt.plot(neuralHistory.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['linear train', 'linear test', 'CNN train', 'CNN test'], loc='lower right')

    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.plot(neuralHistory.history['loss'])
    plt.plot(neuralHistory.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['linear train', 'linear test', 'CNN train', 'CNN test'], loc='upper right')

    plt.tight_layout()
    plt.show()
    return
