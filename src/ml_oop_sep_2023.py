import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.pipeline import Pipeline
from joblib import dump
import sys

class MachineLearningModel:
    def __init__(self):
        self.data = None
        self.independ_var = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()

    def load_data(self, file_name=None):
        if file_name is None:
            file_name = input('Enter the CSV file name: ')
        file_name = file_name + '.csv'
        self.data = pd.read_csv(file_name)

    def choose_independent_variable(self):
        while True:
            self.independ_var = input('Choose an independent variable: ')
            if self.independ_var in self.data.columns:
                print(f'You have chosen "{self.independ_var}" as the independent variable.')
                break
            else:
                print('Invalid column name. Please try again.')

    def handle_missing_values(self):
        if self.data.isnull().values.any():
            non_names = self.data.columns[self.data.isnull().any()].tolist()
            print(f'These columns have missing values: {non_names}')
            print('Please fill in missing values and return to the app.')
            sys.exit()

    def convert_categorical_to_dummy(self):
        str_list = list(self.data.select_dtypes(include=['object']).columns)
        if len(str_list) > 0:
            print(f'Columns with string data type that need to create dummies: {str_list}')
            create_dummy = input('Do you want to create dummies? (Y: yes, N: no)')
            if create_dummy.upper() == 'Y':
                self.data = pd.get_dummies(self.data, columns=str_list, drop_first=True)
            elif create_dummy.upper() == 'N':
                print('Please convert string type data to dummies yourself and return to the app.')
                sys.exit()
    


    def split_data(self):
        X = self.data.drop([self.independ_var], axis=1)
        y = self.data[self.independ_var]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    def scale_data(self):
        self.scaler.fit(self.X_train)
        self.X_train = self.scaler.transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

class RegressionModel(MachineLearningModel):
    def __init__(self):
        super().__init__()

    def train_models(self):
        # Linear Regression
        model_lir = LinearRegression()
        model_lir.fit(self.X_train, self.y_train)
        self.save_model(model_lir, 'Linear_Regression')

        # Lasso
        lasso_model_cv = LassoCV(eps=0.1, n_alphas=100, cv=10)
        lasso_model_cv.fit(self.X_train, self.y_train)
        self.save_model(lasso_model_cv, 'Lasso_Regression')

        # Ridge CV
        ridge_model_CV = RidgeCV(alphas=(0.1, 0.5, 1, 5, 10), scoring='neg_mean_squared_error')
        ridge_model_CV.fit(self.X_train, self.y_train)
        self.save_model(ridge_model_CV, 'Ridge_Regression')

        # Elastic Net
        elastic_model_cv = ElasticNetCV(l1_ratio=[0.001, 0.005, 0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1], tol=0.0001, max_iter=1_000_000)
        elastic_model_cv.fit(self.X_train, self.y_train)
        self.save_model(elastic_model_cv, 'Elastic_Net')

        # Support Vector Machine (SVM)
        base_svr_model = SVR()
        param_grid = {'C': [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 100],
                      'kernel': ['linear', 'poly', 'rbf'],
                      'degree': [2, 3, 4],
                      'epsilon': [0, 0.01, 0.1, 0.5, 1, 2],
                      'gamma': ['scale', 'auto']}
                      #'cv': 3 } # Adjust the number of splits (e.g., to 3)}
        
        grid_SVR = GridSearchCV(estimator=base_svr_model, param_grid=param_grid, cv=3)
        grid_SVR.fit(self.X_train, self.y_train)
        self.save_model(grid_SVR.best_estimator_, 'SVM')

        # Artificial Neural Network (ANN) for Regression
        ann_regressor = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        ann_regressor.fit(self.X_train, self.y_train)
        self.save_model(ann_regressor, 'ANN_Regression')

    def save_model(self, model, name):
        dump(model, f'{name}_predictor.joblib')
        print(f'Your final prediction file for {name} is ready as ({name}_predictor.joblib) in the directory.')

    def train(self):
        self.choose_independent_variable()
        self.handle_missing_values()
        self.convert_categorical_to_dummy()
        self.split_data()
        self.scale_data()
        self.train_models()

class ClassificationModel(MachineLearningModel):
    def __init__(self):
        super().__init__()

    def train_models(self):
        # Logistic Regression
        log_model_cv = LogisticRegression(solver='saga', multi_class='auto', max_iter=5000)
        penalty = ['l1', 'l2']
        C = np.linspace(0.1, 4, 10)
        param_grid_log = {'C': C, 'penalty': penalty}
        grid_model_cv = GridSearchCV(log_model_cv, param_grid=param_grid_log)
        grid_model_cv.fit(self.X_train, self.y_train)
        self.save_model(grid_model_cv, 'Logistic_Regression')

        # K-Nearest Neighbors (KNN)
        knn = KNeighborsClassifier()
        operations = [('scaler', self.scaler), ('knn', knn)]
        pipe = Pipeline(operations)
        k_values = list(range(1, 30))
        param_grid_knn = {'knn__n_neighbors': k_values}
        knn_model_cv = GridSearchCV(estimator=pipe, param_grid=param_grid_knn, scoring='accuracy')
        knn_model_cv.fit(self.X_train, self.y_train)
        self.save_model(knn_model_cv, 'KNN')

        # Support Vector Classifier (SVC)
        svc = SVC()
        param_grid_svc = {'C': [0.001, 0.005, 0.01, 0.05, 1, 5, 10, 50, 100, 500, 1000, 5000]}
        param_grid_svc = {'C': [0.001, 0.005, 0.01, 0.05, 1, 5, 10, 50, 100, 500, 1000, 5000], 'kernel': ['linear', 'rbf']}
        grid_svc = GridSearchCV(svc, param_grid=param_grid_svc)
        grid_svc.fit(self.X_train, self.y_train)
        self.save_model(grid_svc, 'SVC')

        # Artificial Neural Network (ANN) for Classification
        ann_classifier = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        ann_classifier.fit(self.X_train, self.y_train)
        self.save_model(ann_classifier, 'ANN_Classification')

    def save_model(self, model, name):
        dump(model, f'{name}_predictor.joblib')
        print(f'Your final prediction file for {name} is ready as ({name}_predictor.joblib) in the directory.')

    def train(self):
        self.choose_independent_variable()
        self.handle_missing_values()
        self.convert_categorical_to_dummy()
        self.split_data()
        self.scale_data()
        self.train_models()


if __name__ == "__main__":
    while True:
        response = input('Hello, please tell us which kind of machine learning method you need (R: regression, C: classifier): ')
        if response.upper() == 'R':
            regression_model = RegressionModel()
            regression_model.load_data(input('What is the name of your CSV file? Make sure you have uploaded it: '))
            regression_model.train()
            break
        elif response.upper() == 'C':
            classification_model = ClassificationModel()
            classification_model.load_data(input('What is the name of your CSV file? Make sure you have uploaded it: '))
            classification_model.train()
            break