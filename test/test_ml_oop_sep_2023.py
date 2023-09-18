import sys
import os
#print(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import unittest
import pandas as pd
from unittest.mock import patch  
from ml_oop_sep_2023 import MachineLearningModel, RegressionModel, ClassificationModel



class TestMachineLearningModel(unittest.TestCase):
    def setUp(self):
        # Create an instance of MachineLearningModel
        self.model = MachineLearningModel()
    
    @patch('builtins.input', side_effect=['mpg'])  # Mock user input
    def test_load_data(self, mock_input):
        try:
            self.model.load_data()
        except FileNotFoundError:
            self.fail("File 'mpg.csv' not found.")
        self.assertIsNotNone(self.model.data)

    def test_choose_independent_variable(self):
        # Assuming you have sample data loaded, provide a valid independent variable name
        self.model.data = pd.DataFrame({
            'X1': [1, 2, 3, 4, 5],
            'y': [3, 4, 5, 6, 7]
        })
        with patch('builtins.input', return_value='X1'):
            self.model.choose_independent_variable()
        self.assertIsNotNone(self.model.independ_var)

    def test_handle_missing_values_no_missing(self):
        # Test when there are no missing values in the data
        self.model.data = pd.DataFrame({
            'X1': [1, 2, 3, 4, 5],
            'y': [3, 4, 5, 6, 7]
        })
        self.assertIsNone(self.model.handle_missing_values())

    def test_handle_missing_values_with_missing(self):
        # Test when there are missing values in the data
        self.model.data = pd.DataFrame({
            'X1': [1, 2, 3, 4, 5],
            'y': [3, 4, None, 6, 7]
        })
        with self.assertRaises(SystemExit):
            self.model.handle_missing_values()

    
    def test_convert_categorical_to_dummy(self):
        self.model.data = pd.DataFrame({
            'X1': [1, 2, 3, 4, 5],
            'cat_col': ['A', 'B', 'C', 'B', 'C']
        })
        with patch('builtins.input', return_value='Y'):
            self.model.convert_categorical_to_dummy()
        self.assertNotIn('cat_col_A', self.model.data.columns)  # Expect 'cat_col_A' to be dropped
        self.assertIn('cat_col_B', self.model.data.columns)
        self.assertIn('cat_col_C', self.model.data.columns)


    def test_split_data(self):
        # Test the split_data method
        self.model.data = pd.DataFrame({
            'X1': [1, 2, 3, 4, 5],
            'y': [3, 4, 5, 6, 7]
        })
        self.model.independ_var = 'y'
        self.model.split_data()
        self.assertIsNotNone(self.model.X_train)
        self.assertIsNotNone(self.model.X_test)
        self.assertIsNotNone(self.model.y_train)
        self.assertIsNotNone(self.model.y_test)

    def test_scale_data(self):
        # Test the scale_data method
        self.model.data = pd.DataFrame({
            'X1': [1, 2, 3, 4, 5],
            'y': [3, 4, 5, 6, 7]
        })
        self.model.independ_var = 'y'
        self.model.split_data()
        self.model.scale_data()
        self.assertIsNotNone(self.model.X_train)
        self.assertIsNotNone(self.model.X_test)

    # Add more test cases for other methods in MachineLearningModel

class TestRegressionModel(unittest.TestCase):
    def setUp(self):
        # Create an instance of RegressionModel
        self.model = RegressionModel()

    @patch('builtins.input', side_effect=['mpg'])  # Mock user input
    def test_load_data(self, mock_input):
        self.model.load_data()
        self.assertIsNotNone(self.model.data)

    def test_choose_independent_variable(self):
        # Assuming you have sample data loaded, provide a valid independent variable name
        self.model.data = pd.DataFrame({
            'X1': [1, 2, 3, 4, 5],
            'y': [3, 4, 5, 6, 7]
        })
        with patch('builtins.input', return_value='X1'):
            self.model.choose_independent_variable()
        self.assertIsNotNone(self.model.independ_var)
    

    def test_train_models(self):
        # Test the train_models method for RegressionModel
        self.model.data = pd.DataFrame({
            'X1': list(range(1, 101)),  # 100 data points
            'y': [i * 2 + 5 for i in range(1, 101)]  # Linear relation as an example
        })
        self.model.independ_var = 'y'
        self.model.split_data()
        self.model.scale_data()
        self.model.train_models()
        # Add assertions to check the trained models or saved files as needed

    

    def test_save_model(self):
                # Test the save_model method for RegressionModel
        # Assuming you have a trained model to save
        dummy_model = 'Dummy Model'
        self.model.save_model(dummy_model, 'Dummy_Model')
        # Add assertions to check if the model is saved correctly

    # Add more test cases for other methods in RegressionModel

class TestClassificationModel(unittest.TestCase):
    def setUp(self):
        # Create an instance of ClassificationModel
        self.model = ClassificationModel()

    @patch('builtins.input', side_effect=['mpg'])  # Mock user input
    def test_load_data(self, mock_input):
        self.model.load_data()
        self.assertIsNotNone(self.model.data)

    def test_choose_independent_variable(self):
        # Assuming you have sample data loaded, provide a valid independent variable name
        self.model.data = pd.DataFrame({
            'X1': [1, 2, 3, 4, 5],
            'y': ['A', 'B', 'A', 'C', 'B'] })
        with patch('builtins.input', return_value='X1'):
            self.model.choose_independent_variable()
        self.assertIsNotNone(self.model.independ_var)

    def test_train_models(self):
        # Test the train_models method for ClassificationModel
        self.model.data = pd.DataFrame({
            'X1': list(range(1, 101)),  # 100 data points
            'y': ['A'] * 50 + ['B'] * 30 + ['C'] * 20  # Three classes in a total of 100 points
        })
        self.model.independ_var = 'y'
        self.model.split_data()
        self.model.scale_data()
        self.model.train_models()
        # Add assertions to check the trained models or saved files as needed


    def test_save_model(self):
        # Test the save_model method for ClassificationModel
        # Assuming you have a trained model to save
        dummy_model = 'Dummy Model'
        self.model.save_model(dummy_model, 'Dummy_Model')
        # Add assertions to check if the model is saved correctly

    # Add more test cases for other methods in ClassificationModel

    
    
    





if __name__ == '__main__':
    unittest.main()
    
