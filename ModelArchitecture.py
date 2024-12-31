from visualkeras import layered_view
import keras
import matplotlib.pyplot as plt

model = keras.models.load_model('model.keras')
view = layered_view(model, legend=True, max_xy=300)
view.show()