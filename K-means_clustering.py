from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#sample data:400개,중심 4개, 2차원 벡터 데이터 set 생성
X,y_true = make_blobs(n_samples=400, centers=4 , n_features= 2, cluster_std=1.2)
plt.scatter(X[:,0] , X[:,1] , s=20)
plt.show()

#4개로 나누고, 초기화 값은 random으로 kmean 시작
kmeans = KMeans(n_clusters=4, init='k-means++')
#X를 kmeans에 넣기
kmeans.fit(X)
#센터값 추출
center = kmeans.cluster_centers_
#새 클러스터 출력
y_kmeans = kmeans.fit_predict(X)
plt.scatter(X[:,0] , X[:,1] , c=y_kmeans , s=20 , cmap='viridis')
plt.scatter(center[:,0] , center[:,1] , c='black' , s=100)
plt.show()