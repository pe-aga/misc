'''A sample application showing how can we (de)serialize ML models to/from the disk.'''
from sklearn.neural_network import MLPClassifier
import pickle

SERIALIZED_FILE_PATH = "/Users/neo/Documents/workspace/serialized_model"

def generate_and_train_model():
  '''Generate and train a MPLClassifier model with dummy data.'''
  X = [[0., 0.], [1., 1.]]
  y = [0, 1]

  neural_network = MLPClassifier(solver='lbfgs', alpha=1e-5,
                                 hidden_layer_sizes=(5, 2), random_state=1)
  neural_network.fit(X, y) 
  return neural_network

def serialize_model(file_path, model):
  '''Serialize the model to the file.'''
  with open(file_path, 'wb') as f:
    pickle.dump(model, f)

def deserialize_model(file_path):
  '''Deserialize the model from the file.'''
  with open(file_path, 'rb') as f:
    return pickle.load(f)

neural_network = generate_and_train_model()
print("Result of a regular model")
print(neural_network.predict([[2., 2.], [-1., -2.]]))

serialize_model(SERIALIZED_FILE_PATH, neural_network)

deserialized_model = deserialize_model(SERIALIZED_FILE_PATH)
print("Result of a deserialized model")
print(deserialized_model.predict([[2., 2.], [-1., -2.]]))
