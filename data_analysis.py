import pandas as pd

# 시리즈
'''
데이터가 순차적으로 나열된 1차원 배열
index, value은 일대응 대응 관계
'''

dict_data = {'a':1,'b':2,'c':3}
series = pd.Series(dict_data)

print(series)
'''
a    1
b    2
c    3
dtype: int64
'''
print(type(series))
'''
<class 'pandas.core.series.Series'>
'''
print(series.index)
'''
Index(['a', 'b', 'c'], dtype='object')
'''
print(series.values)
'''
[1 2 3]
'''

list_data = ['a', 'b', 'c']
series = pd.Series(list_data)

print(series)
'''
0    a
1    b
2    c
'''

series = pd.Series(dict_data, index=['i1', 'i2', 'i3'])

print(series)
'''
i1   NaN
i2   NaN
i3   NaN
'''

series = pd.Series(list_data, index=['i1', 'i2', 'i3'])

print(series)
'''
i1    a
i2    b
i3    c
'''

print(series['i1'])
print(series[['i1', 'i2']])
# print(series[0]) # deprecated
# print(series[[0, 1]]) # deprecated

a = pd.Series([1, 2, 3])
b = pd.Series([2, 3, 4])
print(a + b)
'''
0    3
1    5
2    7
'''

print(a * b)
'''
0     2
1     6
2    12
'''

# 데이터 프레임
'''
시리즈 1차원 배열이면 데이터 프레임은 2차원 배열

데이터프레임의 각 열은 시리즈 객체, 이러한 시리즈가 모여 데이터 프레임을 구성
'''

dict_data = {'col1': [1, 2, 3], 'col2': [2, 3, 4], 'col3': [3, 4, 5]}
df = pd.DataFrame(dict_data)
'''
   col1  col2  col3
0     1     2     3
1     2     3     4
2     3     4     5
'''

print(df)
print(type(df))

df = pd.DataFrame([[1, 2, 3], [2, 3, 4],[3, 4, 5]])

print(df)
'''
   0  1  2
0  1  2  3
1  2  3  4
2  3  4  5
'''

df = pd.DataFrame([[1, 2, 3], [2, 3, 4],[3, 4, 5]],
                  index=['idx1', 'idx2', 'idx3'],
                  columns=['col1','col2','col3'])

print(df)
'''
      col1  col2  col3
idx1     1     2     3
idx2     2     3     4
idx3     3     4     5
'''

df.index = [5, 4, 3]
df.columns = [3, 4, 5]

print(df)
'''
   3  4  5
5  1  2  3
4  2  3  4
3  3  4  5
'''

df.rename(index={5:55}, inplace=True)
df.rename(columns={3:33}, inplace=True)

print(df)
'''
    33  4   5 
55   1   2   3
4    2   3   4
3    3   4   5
'''

df.drop(55, axis=0, inplace=True)
df.drop(33, axis=1, inplace=True)

print(df)

'''
   4  5
4  3  4
3  4  5
'''


df = pd.DataFrame([[1, 2, 3], [2, 3, 4],[3, 4, 5]],
                  index=['idx1', 'idx2', 'idx3'],
                  columns=['col1','col2','col3'])

print(df['col1'])
'''
idx1    1
idx2    2
idx3    3
Name: col1, dtype: int64
'''
print(df.loc['idx1'])
'''
col1    1
col2    2
col3    3
Name: idx1, dtype: int64
'''
print(df.loc[['idx1','idx3']])
'''
      col1  col2  col3
idx1     1     2     3
idx3     3     4     5
'''
print(df.loc['idx1':'idx3'])
'''
      col1  col2  col3
idx1     1     2     3
idx2     2     3     4
idx3     3     4     5
'''
print(df.iloc[1])
'''
col1    2
col2    3
col3    4
Name: idx2, dtype: int64
'''
print(df.loc[['idx1','idx3'], ['col1', 'col3']])
'''
      col1  col3
idx1     1     3
idx3     3     5
'''
print(df.iloc[1, 1])
'''
3
'''

# 데이터 불러오기 및 저장하기
'''
csv
read_csv(), to_csv()

excel
read_excel(), to_excel()

sql
read_sql(), to_sql()

html
read_html() to_html()

json
read_json(), to_json()

hdf5
read_hdf(), to_hdf()
'''

# CSV 파일을 읽어 데이터 프레임으로 변환
data_csv = pd.read_csv('kospi.csv')

print(data_csv)
'''
           Date    Close   Ret
0    2020-01-02  2175.17 -1.02
1    2020-01-03  2176.46  0.06
2    2020-01-06  2155.07 -0.98
3    2020-01-07  2175.54  0.95
4    2020-01-08  2151.31 -1.11
..          ...      ...   ...
243  2020-12-23  2759.82  0.96
244  2020-12-24  2806.86  1.70
245  2020-12-28  2808.60  0.06
246  2020-12-29  2820.51  0.42
247  2020-12-30  2873.47  1.88

[248 rows x 3 columns]
'''

data_csv.to_csv('changed_kospi.csv')

# CSV 파일을 읽어 데이터 프레임으로 변환
data_excel = pd.read_excel('kospi.xlsx')
'''
          Date    Close   Ret
0   2020-01-02  2175.17 -1.02
1   2020-01-03  2176.46  0.06
2   2020-01-06  2155.07 -0.98
3   2020-01-07  2175.54  0.95
4   2020-01-08  2151.31 -1.11
..         ...      ...   ...
243 2020-12-23  2759.82  0.96
244 2020-12-24  2806.86  1.70
245 2020-12-28  2808.60  0.06
246 2020-12-29  2820.51  0.42
247 2020-12-30  2873.47  1.88

[248 rows x 3 columns]
'''

print(data_excel)

data_excel.to_excel('changed_kospi.xlsx')