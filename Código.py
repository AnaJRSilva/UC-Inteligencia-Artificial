# Importação de bibliotecas
import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from prophet import Prophet

# Definição da classe SeaLevelRegressor
class SeaLevelRegressor:
    def __init__(self, data_path):
        # Leitura do conjunto de dados
        self.df = pd.read_csv(data_path)

    def explore_data(self): # Exploração inicial dos dados
        print(self.df.head())
        print(self.df.shape)
        print(self.df.describe())
        print(self.df.info())
        print(self.df.isnull().sum())
        self.df.drop_duplicates()
        print(self.df.shape)
        print('\n')

    def plot_column(self, col): # Visualização de dispersão da coluna ao longo dos anos
        sns.scatterplot(data=self.df, x='Year', y=col)
        plt.title(f'Scatter Plot - {col} over Years')
        plt.xlabel('Year')
        plt.ylabel(col)
        plt.show() 

    def preprocess_data(self): # Pré-processamento dos dados
        X = self.df.drop("GMSL_noGIA", axis=1)
        Y = self.df["GMSL_noGIA"]

        # Divisão do conjunto de dados
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        # Normalização dos dados
        min_max_scaler = MinMaxScaler()
        X_train = min_max_scaler.fit_transform(X_train)
        X_val = min_max_scaler.transform(X_val)
        X_test = min_max_scaler.transform(X_test)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def knn_regression(self, X_train, X_test, y_train, y_test): # Regressão com o algoritmo KNN
        knn_model = KNeighborsRegressor(n_neighbors=5)
        knn_model.fit(X_train, y_train)
        predictions_knn = knn_model.predict(X_test)
        mse_knn = mean_squared_error(y_test, predictions_knn)
        r2_knn = r2_score(y_test, predictions_knn)
        print("KNN | Mean Squared Error:", mse_knn)
        print("KNN | R2 Score:", r2_knn)

    def linear_regression(self, X_train, X_test, y_train, y_test): # Regressão Linear
        model = LinearRegression()
        reg = model.fit(X_train, y_train)
        predictions_reg = reg.predict(X_test)
        mse_reg = mean_squared_error(y_test, predictions_reg)
        r2_reg = r2_score(y_test, predictions_reg)
        print("Linear Regression | Mean Squared Error:", mse_reg)
        print("Linear Regression | R2 Score:", r2_reg)

    def mlp_regression(self, X_train, X_test, y_train, y_test): # Regressão com MLP
        model = MLPRegressor(random_state=42, max_iter=1000, hidden_layer_sizes=(100, 50, 25),
                             alpha=0.1, activation='relu', learning_rate='adaptive')
        reg = model.fit(X_train, y_train)
        predictions_mlp = reg.predict(X_test)
        mse_mlp = mean_squared_error(y_test, predictions_mlp)
        r2_mlp = r2_score(y_test, predictions_mlp)
        print("MLP Regression | Mean Squared Error:", mse_mlp)
        print("MLP Regression | R2 Score:", r2_mlp)
    
    def prediction_prophet(self): # Aplicação do Phophet (Fonte: https://www.kaggle.com/code/aranpandey/global-sea-level-eda-predictions)
        
        # Preparação dos dados para o modelo Prophet
        df2 = self.df[['Year', 'GMSL_noGIA']].copy()
        df2.columns = ['ds', 'y']

        # Inicialização e treinamento do modelo Prophet
        model1 = Prophet(yearly_seasonality=True)
        model1.fit(df2)

        # Geração de dados futuros para predição
        future = model1.make_future_dataframe(periods=60, freq='Y')

        # Predição utilizando o modelo Prophet
        forecast = model1.predict(future)

        # Visualização dos últimos 10 resultados previstos
        print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))

        # Visualização gráfica do resultado da predição
        model1.plot(forecast)
        plt.xlabel('Year')
        plt.ylabel('GSML no GIA')
        plt.show()

if __name__ == "__main__":
    data_path = r'C:\Users\Ana Júlia\OneDrive\Documentos\trabalhoIA\sealevel.csv'
    sea_level_regressor = SeaLevelRegressor(data_path)
    sea_level_regressor.explore_data()
   
    sea_level_regressor.plot_column('TotalWeightedObservations')
    sea_level_regressor.plot_column('GMSL_noGIA')
    sea_level_regressor.plot_column('StdDevGMSL_noGIA')
    sea_level_regressor.plot_column('SmoothedGSML_noGIA')
    sea_level_regressor.plot_column('GMSL_GIA')
    sea_level_regressor.plot_column('StdDevGMSL_GIA')
    sea_level_regressor.plot_column('SmoothedGSML_GIA')
    sea_level_regressor.plot_column('SmoothedGSML_GIA_sigremoved')

    X_train, X_val, X_test, y_train, y_val, y_test = sea_level_regressor.preprocess_data()

    print(f"Tamanho do Conjunto de Treinamento: {len(X_train)}")
    print(f"Tamanho do Conjunto de Validação: {len(X_val)}")
    print(f"Tamanho do Conjunto de Teste: {len(X_test)}\n")

    # Avaliação do KNN Regression
    start_time = time.time()
    sea_level_regressor.knn_regression(X_train, X_test, y_train, y_test)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Tempo de Execução do KNN Regression: {execution_time} segundos\n")
    
    # Avaliação da Linear Regression
    start_time = time.time()
    sea_level_regressor.linear_regression(X_train, X_test, y_train, y_test)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Tempo de Execução da Linear Regression: {execution_time} segundos\n")

    # Avaliação da MLP Regression
    start_time = time.time()
    sea_level_regressor.mlp_regression(X_train, X_test, y_train, y_test)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Tempo de Execução da MLP Regression: {execution_time} segundos\n")

    # Avaliação da Prophet Prediction
    start_time = time.time()
    sea_level_regressor.prediction_prophet()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Tempo de Execução da Prophet Prediction: {execution_time} segundos\n")
