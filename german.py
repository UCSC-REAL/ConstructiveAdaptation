df = pd.read_csv('data/german_processed.csv')
improvement_features = [8, 10, 11, 13,14,15,16,17, 18, 19, 20, 21, 22, 23, 26]
manipulated_features = [5,6,7,12]
unaction_features = [1, 2, 3, 4, 9, 24, 25, 27, 28, 29]
N_I = 15
N_M = 4

df.loc[:, 'PurposeOfLoan'] = preprocessing.LabelEncoder().fit_transform(df.PurposeOfLoan)
y = df.GoodCustomer.values
x = df.values
x = np.concatenate((x[:, improvement_features], x[:, manipulated_features], x[:, unaction_features]), axis=1)
x = preprocessing.StandardScaler().fit_transform(x)
x_0, y_0 = resample(x[y == -1], y[y==-1], n_samples=1000, random_state=426)
x_1, y_1 = resample(x[y == 1], y[y==1], n_samples=1000, random_state=426)
x = np.concatenate((x_0, x_1), axis = 0)
y = np.concatenate((y_0, y_1))

x_I = x[:, :N_I]
x_M = x[:, N_I:N_I+N_M]
x_N = x[:, N_I+N_M:]