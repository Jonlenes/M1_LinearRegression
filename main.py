from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score as acc, mean_squared_error
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

import numpy as np
import pandas as p

from preprocess_dataset import preprocess, path_dataset
from LR_v1 import LR
from LR_v2 import LR_NEW

columns_type_number = ["n_tokens_title", "n_tokens_content", "num_hrefs", "num_self_hrefs", "num_imgs", "num_videos",
                       "average_token_length", "num_keywords", "kw_min_min", "kw_max_min", "kw_avg_min", "kw_min_max",
                       "kw_max_max", "kw_avg_max", "kw_min_avg", "kw_max_avg", "kw_avg_avg", "self_reference_min_shares",
                       "self_reference_max_shares", "self_reference_avg_sharess"]


def rescaling(x):
    for j in range(x.shape[1]):
        arr = x[:, j]
        min_val = np.min(arr)
        max_val = np.max(arr)
        x[:, j] = (x[:, j] - min_val) / (max_val - min_val)
    return x


def mean_normalisation(x):
    for j in range(x.shape[1]):
        arr = x[:, j]
        min_val = np.min(arr)
        max_val = np.max(arr)
        mean = np.mean(arr)
        x[:, j] = (x[:, j] - mean) / (max_val - min_val)
    return x


def transpose_vector(arr):
    return np.asarray(np.matrix(arr).transpose())


# Model Evaluation - RMSE
def rmse(Y, Y_pred):
    Y_pred = np.asarray(np.matrix(Y_pred).transpose())
    rmse = np.sqrt(sum((Y - Y_pred) ** 2) / len(Y))
    return rmse


# Model Evaluation - R2 Score
def r2_score(Y, Y_pred):
    Y_pred = np.asarray(np.matrix(Y_pred).transpose())
    mean_y = np.mean(Y)
    ss_tot = sum((Y - mean_y) ** 2)
    ss_res = sum((Y - Y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


def main(algo_scaling=2, algo_reg=0, algo_select=0):
    """
    Executa o features scaling, features select, trainamento e validação

    Parameters
    ----------
    algo_scaling : int
        0, 1, 2 - rescaling, mean_normalisation, scale
    algo_reg     : int
        0, 1 - Implementação propria, Sklearn
    algo_select  : int
        Algoritmo utilizado para realizada de feature select
        0, 1, 2 - SelectKBest, PCA (Não faz muito sentido usar 
        PCA neste problema, no entanto ele foi testado assim mesmo), RFE

    """

    func_scaling = [rescaling, 
                    mean_normalisation, 
                    scale][algo_scaling]

    model = [LR_NEW(), 
            LinearRegression()][algo_reg]


    min_custo, min_error, max_acc = 10**10, 10**10, -1
    id_min_custo, id_min_error, id_max_acc = 1, 1, 1


    # Testando a utilização de diversos valores diferentes para a quantidade de features
    for i in range(1, 59):
        fs = [SelectKBest(score_func=chi2, k=i),
            PCA(n_components=i),
            RFE(model, i)][algo_select]
        
        x_train, _, y_train, _ = preprocess()

        x_train = func_scaling(x_train)
        fs = fs.fit(x_train, y_train.ravel())
        x_train = fs.transform(x_train)

        # Split train e validação 
        x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.2, random_state=0)

        print("Inicio do treinamento.")
        cost = model.fit(x_train, y_train)

        if cost < min_custo:
            min_custo = cost
            id_min_custo = i


        y_pred_train = model.predict(x_train)
        y_pred_validation = model.predict(x_validation)

        r = rmse(y_validation, y_pred_validation)
        if r < min_error:
            min_error = r
            id_min_error = i

        print("mean_squared_error train:", rmse(y_train, y_pred_train))
        print("mean_squared_error validation:", r)

        a = acc(y_validation, y_pred_validation)
        if a > max_acc:
            max_acc = a
            id_max_acc = i

        print("accuracy_score train:", acc(y_train, y_pred_train))
        print("accuracy_score validation:", a)

    print(id_min_custo, id_min_error, id_max_acc)


if __name__ == '__main__':
    main()
