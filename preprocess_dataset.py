import pandas


# endereço do dataset
path_dataset = "."


def load_csv_file(file_name):
    return pandas.read_csv(path_dataset + file_name)


def binarize_shares(df):
    # atualizando o valor do campo 'shares' baseado no seu valor antigo
    #   se 'shares' >= 1400 --->    [shares] = 1
    #   se 'shares' < 1400  --->    [shares] = 0
    greater_1400 = df['shares'] >= 1400
    less_1400 = df['shares'] < 1400

    df.loc[greater_1400, 'shares'] = 1
    df.loc[less_1400, 'shares'] = 0


def preprocess(perct_dataset_train=1.0, perct_dataset_test=1.0):

    # carregando os dataframes com os arquivos csv
    df_train = load_csv_file("train2.csv")
    df_test = load_csv_file("test.csv")
    df_test_target = load_csv_file("test_target.csv")

    df_train = df_train[df_train.average_token_length < 1000]
    df_train = df_train[df_train.kw_avg_min < 10000]
    # df_train = df_train[df_train.kw_min_max < 82000]


    # binarizando o campo 'shares'
    # binarize_shares(df_train)
    # binarize_shares(df_test_target)

    # definindo linhas e colunas a serem consideradas no train
    columns_train = list(df_train.columns[2: len(df_train.columns) - 1])  # colunas que serão consideradas no treino
    row_count_train = int(df_train.shape[0] * perct_dataset_train)  # quatidade de linhas conforme percentual passado

    # pegando a carasteristicas do train
    features_train = df_train[columns_train][:row_count_train]

    # definindo linhas e colunas a serem consideradas no test
    columns_test = list(df_test.columns[2:])  # colunas que serão consideradas no teste
    row_count_test = int(df_test.shape[0] * perct_dataset_test)  # quatidade de linhas conforme percentual passado

    # pegando a carasteristicas do test
    features_test = df_test[columns_test][:row_count_test]

    # pegando os targets/labes
    labels_train = df_train[['shares']][:row_count_train]
    labels_test = df_test_target[['shares']][:row_count_test]

    return features_train.values, features_test.values, labels_train.values, labels_test.values

