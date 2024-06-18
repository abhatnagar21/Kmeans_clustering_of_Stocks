import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Define the dictionary of companies and their tickers
companies_dict = {
    'Amazon': 'AMZN',
    'Apple': 'AAPL',
    'Walgreen': 'WBA',
    'Northrop Grumman': 'NOC',
    'Boeing': 'BA',
    'Lockheed Martin': 'LMT',
    'McDonalds': 'MCD',
    'Intel': 'INTC',
    'IBM': 'IBM',
    'Texas Instruments': 'TXN',
    'MasterCard': 'MA',
    'Microsoft': 'MSFT',
    'General Electrics': 'GE',
    'American Express': 'AXP',
    'Pepsi': 'PEP',
    'Coca Cola': 'KO',
    'Johnson & Johnson': 'JNJ',
    'Toyota': 'TM',
    'Honda': 'HMC',
    'Exxon': 'XOM',
    'Chevron': 'CVX',
    'Valero Energy': 'VLO',
    'Ford': 'F',
    'Bank of America': 'BAC'
}

start_date = '2015-04-25'
end_date = '2020-04-25'

# Fetch data from Yahoo Finance using yfinance
data = yf.download(list(companies_dict.values()), start=start_date, end=end_date)

# Access the stock price data
df_open = data['Open']
df_close = data['Close']
df_volume = data['Volume']

# Display the dataframe
print(df_open.head())
print(df_open.isna().sum())

# Process the data
stock_open = np.array(df_open).T
stock_close = np.array(df_close).T
movements = stock_close - stock_open

for i, company in enumerate(companies_dict.keys()):
    print('company:{}, Change:{}'.format(company, movements[i]))

# Plotting
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.title('Company: Amazon', fontsize=20)
plt.xticks(fontsize=10)
plt.yticks(fontsize=20)
plt.xlabel('Date', fontsize=15)
plt.ylabel('Opening price', fontsize=15)
plt.plot(df_open['AMZN'])

plt.subplot(1, 2, 2)
plt.title('Company: Apple', fontsize=20)
plt.xticks(fontsize=10)
plt.yticks(fontsize=20)
plt.xlabel('Date', fontsize=15)
plt.ylabel('Opening price', fontsize=15)
plt.plot(df_open['AAPL'])

plt.figure(figsize=(20, 10))
plt.title('Company: Amazon', fontsize=20)
plt.xticks(fontsize=10)
plt.yticks(fontsize=20)
plt.xlabel('Date', fontsize=20)
plt.ylabel('Price', fontsize=20)
plt.plot(df_open['AMZN'].iloc[0:30], label='Open')
plt.plot(df_close['AMZN'].iloc[0:30], label='Close')
plt.legend(loc='upper left', frameon=False, framealpha=1, prop={'size': 22})

plt.figure(figsize=(20, 8))
plt.title('Company: Amazon', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=20)
plt.xlabel('Date', fontsize=20)
plt.ylabel('Movement', fontsize=20)
plt.plot(movements[0][0:30])

plt.figure(figsize=(20, 10))
plt.title('Company: Amazon', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=20)
plt.xlabel('Date', fontsize=20)
plt.ylabel('Volume', fontsize=20)
plt.plot(df_volume['AMZN'], label='Volume')

plt.figure(figsize=(20, 8))
ax1 = plt.subplot(1, 2, 1)
plt.title('Company: Amazon', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=20)
plt.xlabel('Date', fontsize=20)
plt.ylabel('Movement', fontsize=20)
plt.plot(movements[0])

plt.subplot(1, 2, 2, sharey=ax1)
plt.title('Company: Apple', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=20)
plt.xlabel('Date', fontsize=20)
plt.ylabel('Movement', fontsize=20)
plt.plot(movements[1])

# Normalize the movements
normalizer = Normalizer()
norm_movements = normalizer.fit_transform(movements)

print(norm_movements.min())
print(norm_movements.max())
print(norm_movements.mean())

# Clustering
kmeans = KMeans(n_clusters=10, max_iter=1000)
pipeline = make_pipeline(normalizer, kmeans)
pipeline.fit(movements)
labels = pipeline.predict(movements)

df1 = pd.DataFrame({'labels': labels, 'companies': list(companies_dict.keys())}).sort_values(by=['labels'], axis=0)
print(df1)

# PCA for dimensionality reduction
reduced_data = PCA(n_components=2).fit_transform(norm_movements)

# Plotting decision boundary
h = 0.01
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 10))
plt.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()), cmap=plt.cm.Paired, aspect='auto', origin='lower')
plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=5)
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=169, linewidths=3, color='w', zorder=10)
plt.title('K-Means clustering on stock market movements (PCA-Reduced data)')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()
