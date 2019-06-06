from code import *

if __name__ == '__main__':
    images_y, images_n = import_data()
    images_y, images_n = preparation(images_y, images_n)
    X_train, X_test, y_train, y_test, img_rows, img_cols = dataframe_preparation(images_y, images_n)
    model, X_train, X_test, y_train, y_test = neural_network_architecture(X_train, X_test, y_train, y_test, img_rows, img_cols)
    model = fit_neural_network(model, X_train, X_test, y_train, y_test)
    result = evaluate_model(model, X_test, y_test)