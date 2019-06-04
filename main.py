from code import *

if __name__ == '__main__':
    images_y, images_n = import_data()
    images_y, images_n = preparation(images_y, images_n)
    X_train, X_test, y_train, y_test = dataframe_preparation(images_y, images_n)
    model = neural_network_architecture()
    model = fit_neural_network(model, X_train, X_test, y_train, y_test)
    score = evaluate_model(model, X_test, y_test)