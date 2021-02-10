df = pd.read_csv('data/spambase_data.csv')
x = df.iloc[:, 1:].values
y = df.Spam.values
x = preprocessing.StandardScaler().fit_transform(x)
manipulated_features = [15, 22, 27, 36, 44, 45, 48, 49, 50, 51, 52, 53, 54, 55, 56]
improve_features = list(set(range(57)) - set(manipulated_features))

N_I = 42
N_M = 15

y_0, x_0 = resample(y[y==0], x[y==0], n_samples=5000, random_state=426)
y_1, x_1 = resample(y[y==1], x[y==1], n_samples=5000, random_state=426)
x = np.concatenate((x_0, x_1))
y = np.concatenate((y_0-1, y_1))

x_I = x[:, :N_I]
x_M = x[:, N_I:]
