from responsibly.dataset import AdultDataset

df = AdultDataset().df
unaction_features = ['age', 'race', 'sex', 'native_country', 'relationship', 'workclass', 'marital_status']
unaction_encoder = preprocessing.OrdinalEncoder().fit(df[unaction_features[1:]])
x_N = np.concatenate((np.expand_dims(df.age.values, axis=1), unaction_encoder.transform(df[unaction_features[1:]])), axis=1)

improvement_features = ['education', 'education-num']
improvement_encoder = preprocessing.OrdinalEncoder().fit(df[improvement_features[0]].values.reshape(-1, 1))
x_I = np.concatenate((improvement_encoder.transform(df[improvement_features[0]].values.reshape(-1,1)), df['education-num'].values.reshape(-1, 1)), axis=1)

manipulated_features = ['hours_per_week', 'capital_gain', 'capital_loss']
# manipulated_features = ['hours_per_week', 'capital_gain', 'capital_loss', 'age', 'race', 'sex', 'native_country', 'relationship', 'workclass', 'marital_status']
x_M = df[manipulated_features].values

y = preprocessing.LabelEncoder().fit_transform(df['income_per_year'])
y = y * 2 - 1
x = np.concatenate((x_I, x_M, x_N), axis=1)
N_I, N_M = 2, 3

x_0, y_0 = resample(x[y == -1], y[y==-1], n_samples=10000, random_state=426)
x_1, y_1 = resample(x[y == 1], y[y==1], n_samples=10000, random_state=426)
x = np.concatenate((x_0, x_1), axis = 0)
y = np.concatenate((y_0, y_1))
x = preprocessing.StandardScaler().fit_transform(x)
x_I = x[:, :N_I]
x_M = x[:, N_I:N_I+N_M]
x_N = x[:, N_I+N_M:]
