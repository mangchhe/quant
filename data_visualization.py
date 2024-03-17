from seaborn import load_dataset
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

'''
https://matplotlib.org/
https://seaborn.pydata.org/index.html
'''

# Title: 그래프의 구성 요소
'''
Figure : 그림 전체
Axes: 좌표축
Legend: 범례
Spines : 윤곽선
'''

df = load_dataset('penguins')

# Title: matplotlib 이용한 시각화

# scatter
# plt.scatter(df['flipper_length_mm'], df['body_mass_g'])
# plt.show()

# bar
# df_group = df.groupby('species')['body_mass_g'].mean().reset_index()
# print(df_group)
# plt.bar(x=df_group['species'], height=df_group['body_mass_g'])
# plt.show()

# hist
# plt.rc('font', family='Malgun Gothic')
# plt.hist(df['body_mass_g'], bins=30)
# plt.xlabel('Body Mass')
# plt.ylabel('Count')
# plt.title('Penguin weight distribution')
# plt.show()

# plot
# df_unrate = pd.read_csv(
#     'https://research.stlouisfed.org/fred2/series/UNRATE/downloaddata/UNRATE.csv'
# )
# df_unrate['DATE'] = pd.to_datetime(df_unrate['DATE'])
# plt.plot(df_unrate['DATE'], df_unrate['VALUE'])
# plt.show()

# fig, axes = plt.subplots(2, 1, figsize=(10, 6)) # 2행 1열, 가로 세로 10 6

# axes[0].scatter(df['flipper_length_mm'], df['body_mass_g'])
# axes[0].set_xlabel('날개 길이(mm)')
# axes[0].set_ylabel('몸무게(g)')
# axes[0].set_title('날개와 몸무게 간의 관계')

# axes[1].hist(df['body_mass_g'], bins=30)
# axes[1].set_xlabel('Body Mass')
# axes[1].set_ylabel('Count')
# axes[1].set_title('펭귄의 몸무게 분포')

# plt.subplots_adjust(left=.1,
#                     right=.95,
#                     bottom=.1,
#                     top=.95,
#                     wspace=.5,
#                     hspace=.5)

# plt.show()

# fig, axes = plt.subplots(figsize=(10, 6))

# plt.subplot(2, 1, 1)
# plt.scatter(df['flipper_length_mm'], df['body_mass_g'])
# plt.xlabel('날개 길이(mm)')
# plt.ylabel('몸무게(g)')
# plt.title('날개와 몸무게 간의 관계')

# plt.subplot(2, 1, 2)
# plt.hist(df['body_mass_g'], bins=30)
# plt.xlabel('Body Mass')
# plt.ylabel('Count')
# plt.title('펭귄의 몸무게 분포')

# plt.subplots_adjust(left=.1,
#                     right=.95,
#                     bottom=.1,
#                     top=.95,
#                     wspace=.5,
#                     hspace=.5)

# plt.show()

# Title: pandas 이용한 시각화

df = load_dataset('diamonds')
print(df.head())

plt.rc('font', family='Malgun Gothic')
# df.plot.scatter(x='carat', y='price', figsize=(10, 6), title='캐럿과 가격 간의 관계')
# plt.show()

# df.plot.scatter(x='carat', y='price', c='cut', cmap='Set2', figsize=(10, 6)) # c = 색 구분 열, cmap = 파레트
# plt.show()

# df['price'].plot.hist(figsize=(10, 6), bins=20)
# plt.show()

# df.groupby('color')['carat'].mean().plot.bar(figsize=(10, 6))
# plt.show()

# Title: seaborn 이용한 시각화

df = load_dataset('titanic')
print(df.head())

# sns.scatterplot(data=df, x='age', y='fare')
# plt.show()

# sns.scatterplot(data=df, x='age', y='fare', hue='class', style='class')
# plt.show()

df_pivot = df.pivot_table(index='class',
               columns='sex',
               values='survived',
               aggfunc='mean')
print(df_pivot)
'''
sex       female      male
class                     
First   0.968085  0.368852
Second  0.921053  0.157407
Third   0.500000  0.135447
'''

# sns.heatmap(df_pivot, annot=True, cmap='coolwarm') # annot : 데이터에 값을 표시할지 말지, cmap : 팔레트 종류 (높을 수록 붉은색)
# plt.show()

'''
seaborn
figure-level : matplotlib과 별개로 seaborn의 figure를 만들어 그곳에 그래프를 나타낸다. figure-level 함수를 사용할 경우 facetgrid를 통해 레이아웃을 변경하고 여러 개의 그래프를 나타낼 수 있다.
axes-level : matplotlib의 axes에 그래프를 나타낸다.

figure-levle / axes-level
replot - scatterplot, lineplot
displot - histplot, kdeplot, ecdfplot, rugplot
catplot - stripplot, swarmplot, boxplot, violinplot, pointrplot, barplot
'''

# sns.displot(data=df, x='age', hue='class', kind='hist', alpha=.3)
# plt.show()

# sns.displot(data=df, x='age', col='class')
# plt.show()

# sns.displot(data=df, x='age', col='class', row='sex', kind='hist')
# plt.show()

g, axes = plt.subplots(2, 1, figsize=(8, 6))

sns.histplot(data=df, x='age', hue='class', ax=axes[0])
sns.barplot(data=df, x='class', y='age', ax=axes[1])

axes[0].set_title('클래스별 나이 분포도')
axes[1].set_title('클래스별 평균 나이')

g.tight_layout()
plt.show()