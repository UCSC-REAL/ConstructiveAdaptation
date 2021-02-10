df = pd.read_csv('data/credit_processed.csv')

y_0, x_0 = resample(df[df['NoDefaultNextMonth'] == 0.].iloc[:, 0].values, df[df['NoDefaultNextMonth'] == 0.].iloc[:, 1:].values, n_samples=10000, random_state=1234)
y_1, x_1 = resample(df[df['NoDefaultNextMonth'] == 1.].iloc[:, 0].values, df[df['NoDefaultNextMonth'] == 1.].iloc[:, 1:].values, n_samples=10000, random_state=4321)
x = np.concatenate((x_0, x_1))
y = np.concatenate((y_0-1, y_1))

x = preprocessing.StandardScaler().fit_transform(x)
improvement_features = [6, 14, 15]
x_I = x[:, improvement_features]
manipulated_features = [7, 8, 9, 10, 11, 12, 13]
x_M = x[:, manipulated_features]
unactionable_features = [0, 1, 2, 3, 4, 5, 16]
x_N = x[:, unactionable_features]
x = np.concatenate((x_I, x_M, x_N), axis=1)
N_I, N_M = 3, 7
