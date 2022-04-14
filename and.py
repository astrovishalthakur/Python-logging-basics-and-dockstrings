# import pandas as pd
# from utils.all_utils import prepare_data, save_plot
# from utils.model import Perceptron

# AND = {
#     "x1": [0, 0, 1, 1],
#     "x2": [0, 1, 0, 1],
#     "y": [0, 0, 0, 1]
# }

# df_AND = pd.DataFrame(AND)

# X, y = prepare_data(df_AND)


# ETA = 0.1
# EPOCHS = 10

# model_and = Perceptron(eta = ETA, epochs = EPOCHS)
# model_and.fit(X, y)

# model_and.total_loss()

# model_and.save(filename="and.model", model_dir="model_and")

# save_plot(df_AND, model_and, filename="and.png")



   
from utils.all_utils import prepare_data, save_plot
from utils.model import Perceptron
import pandas as pd


def main(data, modelName, plotName, eta, epochs):
    df_AND = pd.DataFrame(data)
    X, y = prepare_data(df_AND)

    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X, y)

    _ = model.total_loss()

    model.save(filename=modelName, model_dir="model")
    save_plot(df_AND, model, filename=plotName)

if __name__ == "__main__":
    AND = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y" : [0,0,0,1]
    }
    ETA = 0.3
    EPOCHS = 10
    main(data=AND, modelName="and.model", plotName="and.png", eta=ETA, epochs=EPOCHS)

