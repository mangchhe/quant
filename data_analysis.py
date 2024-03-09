import pandas as pd

# Title: 시리즈
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

# Title: 데이터 프레임
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

# Title: 데이터 불러오기 및 저장하기
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

# Title: 데이터 요약 정보 및 통곗값 살펴보기

import seaborn as sns

df = sns.load_dataset('titanic')

print(df.head(5))
'''
   survived  pclass     sex   age  sibsp  parch     fare embarked  class    who  adult_male deck  embark_town alive  alone
0         0       3    male  22.0      1      0   7.2500        S  Third    man        True  NaN  Southampton    no  False
1         1       1  female  38.0      1      0  71.2833        C  First  woman       False    C    Cherbourg   yes  False
2         1       3  female  26.0      0      0   7.9250        S  Third  woman       False  NaN  Southampton   yes   True
3         1       1  female  35.0      1      0  53.1000        S  First  woman       False    C  Southampton   yes  False
4         0       3    male  35.0      0      0   8.0500        S  Third    man        True  NaN  Southampton    no   True
'''
print(df.tail()) # 따로 파라미터 값이 없으면 df 출력한느 것과 동일
'''
     survived  pclass     sex   age  sibsp  parch   fare embarked   class    who  adult_male deck  embark_town alive  alone
886         0       2    male  27.0      0      0  13.00        S  Second    man        True  NaN  Southampton    no   True
887         1       1  female  19.0      0      0  30.00        S   First  woman       False    B  Southampton   yes   True
888         0       3  female   NaN      1      2  23.45        S   Third  woman       False  NaN  Southampton    no  False
889         1       1    male  26.0      0      0  30.00        C   First    man        True    C    Cherbourg   yes   True
890         0       3    male  32.0      0      0   7.75        Q   Third    man        True  NaN   Queenstown    no   True
'''
print(df.shape)
'''
(891, 15)
'''
print(df.info())
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 15 columns):
 #   Column       Non-Null Count  Dtype   
---  ------       --------------  -----   
 0   survived     891 non-null    int64   
 1   pclass       891 non-null    int64   
 2   sex          891 non-null    object  
 3   age          714 non-null    float64 
 4   sibsp        891 non-null    int64   
 5   parch        891 non-null    int64   
 6   fare         891 non-null    float64 
 7   embarked     889 non-null    object  
 8   class        891 non-null    category
 9   who          891 non-null    object  
 10  adult_male   891 non-null    bool    
 11  deck         203 non-null    category
 12  embark_town  889 non-null    object  
 13  alive        891 non-null    object  
 14  alone        891 non-null    bool    
dtypes: bool(2), category(2), float64(2), int64(4), object(5)
memory usage: 80.7+ KB
'''
print(df['sex'].value_counts())
'''
sex
male      577
female    314
'''
print(df[['sex', 'survived']].value_counts())
'''
sex     survived
male    0           468
female  1           233
male    1           109
female  0            81
'''
print(df[['sex', 'survived']].value_counts().sort_index())
'''
sex     survived
female  0            81
        1           233
male    0           468
        1           109
'''
print(df['survived'].mean())
'''
0.3838383838383838
'''
print(df[['survived', 'age']].mean())
'''
survived     0.383838
age         29.699118
'''
print(df['fare'].min())
print(df['fare'].max())
print(df['fare'].mean()) # 산술평균
print(df['fare'].median()) # 중위수
'''
0.0
512.3292
32.204207968574636
14.4542
'''

# Title: 걸측치 처리하기
'''
df.info() 했을 때 Non-Null Count가 나오고 전체 개수에 해당 개수를 빼면 결측치가 나오게 된다.
'''

# isnull() 결측치면 True, 아니면 False / notnull() 결측치면 False, 아니면 True
print(df.head().isnull())
'''
   survived  pclass    sex    age  sibsp  parch   fare  embarked  class    who  adult_male   deck  embark_town  alive  alone
0     False   False  False  False  False  False  False     False  False  False       False   True        False  False  False
1     False   False  False  False  False  False  False     False  False  False       False  False        False  False  False
2     False   False  False  False  False  False  False     False  False  False       False   True        False  False  False
3     False   False  False  False  False  False  False     False  False  False       False  False        False  False  False
4     False   False  False  False  False  False  False     False  False  False       False   True        False  False  False
'''
# dropna() 결측치가 있는 행과 열을 삭제
print(df.dropna().head().isnull())
'''
    survived  pclass    sex    age  sibsp  parch   fare  embarked  class    who  adult_male   deck  embark_town  alive  alone
1      False   False  False  False  False  False  False     False  False  False       False  False        False  False  False
3      False   False  False  False  False  False  False     False  False  False       False  False        False  False  False
6      False   False  False  False  False  False  False     False  False  False       False  False        False  False  False
10     False   False  False  False  False  False  False     False  False  False       False  False        False  False  False
11     False   False  False  False  False  False  False     False  False  False       False  False        False  False  False
'''
print(df.dropna(subset=['age'], axis=0))
'''
     survived  pclass     sex   age  sibsp  parch     fare embarked   class    who  adult_male deck  embark_town alive  alone
0           0       3    male  22.0      1      0   7.2500        S   Third    man        True  NaN  Southampton    no  False
1           1       1  female  38.0      1      0  71.2833        C   First  woman       False    C    Cherbourg   yes  False
2           1       3  female  26.0      0      0   7.9250        S   Third  woman       False  NaN  Southampton   yes   True
3           1       1  female  35.0      1      0  53.1000        S   First  woman       False    C  Southampton   yes  False
4           0       3    male  35.0      0      0   8.0500        S   Third    man        True  NaN  Southampton    no   True
..        ...     ...     ...   ...    ...    ...      ...      ...     ...    ...         ...  ...          ...   ...    ...
885         0       3  female  39.0      0      5  29.1250        Q   Third  woman       False  NaN   Queenstown    no  False
886         0       2    male  27.0      0      0  13.0000        S  Second    man        True  NaN  Southampton    no   True
887         1       1  female  19.0      0      0  30.0000        S   First  woman       False    B  Southampton   yes   True
889         1       1    male  26.0      0      0  30.0000        C   First    man        True    C    Cherbourg   yes   True
890         0       3    male  32.0      0      0   7.7500        Q   Third    man        True  NaN   Queenstown    no   True

[714 rows x 15 columns
'''
print(df.dropna(axis=1))
'''
     survived  pclass     sex  sibsp  parch     fare   class    who  adult_male alive  alone
0           0       3    male      1      0   7.2500   Third    man        True    no  False
1           1       1  female      1      0  71.2833   First  woman       False   yes  False
2           1       3  female      0      0   7.9250   Third  woman       False   yes   True
3           1       1  female      1      0  53.1000   First  woman       False   yes  False
4           0       3    male      0      0   8.0500   Third    man        True    no   True
..        ...     ...     ...    ...    ...      ...     ...    ...         ...   ...    ...
886         0       2    male      0      0  13.0000  Second    man        True    no   True
887         1       1  female      0      0  30.0000   First  woman       False   yes   True
888         0       3  female      1      2  23.4500   Third  woman       False    no  False
889         1       1    male      0      0  30.0000   First    man        True   yes   True
890         0       3    male      0      0   7.7500   Third    man        True    no   True

[891 rows x 11 columns]
'''
print(df.dropna(axis=1, thresh=300)) # 결측치 300개 이상 갖는 열을 삭제
'''
     survived  pclass     sex   age  sibsp  parch     fare embarked   class    who  adult_male  embark_town alive  alone
0           0       3    male  22.0      1      0   7.2500        S   Third    man        True  Southampton    no  False
1           1       1  female  38.0      1      0  71.2833        C   First  woman       False    Cherbourg   yes  False
2           1       3  female  26.0      0      0   7.9250        S   Third  woman       False  Southampton   yes   True
3           1       1  female  35.0      1      0  53.1000        S   First  woman       False  Southampton   yes  False
4           0       3    male  35.0      0      0   8.0500        S   Third    man        True  Southampton    no   True
..        ...     ...     ...   ...    ...    ...      ...      ...     ...    ...         ...          ...   ...    ...
886         0       2    male  27.0      0      0  13.0000        S  Second    man        True  Southampton    no   True
887         1       1  female  19.0      0      0  30.0000        S   First  woman       False  Southampton   yes   True
888         0       3  female   NaN      1      2  23.4500        S   Third  woman       False  Southampton    no  False
889         1       1    male  26.0      0      0  30.0000        C   First    man        True    Cherbourg   yes   True
890         0       3    male  32.0      0      0   7.7500        Q   Third    man        True   Queenstown    no   True

[891 rows x 14 columns]
'''
df_2 = df.copy()
print(df_2['age'].head(6))
'''
0    22.0
1    38.0
2    26.0
3    35.0
4    35.0
5     NaN
'''
mean_age = df_2['age'].mean()
df_2['age'].fillna(mean_age, inplace=True) # 결측치를 해당 인자 값으로 대체한다.
print(df_2['age'].head(6))
'''
0    22.000000
1    38.000000
2    26.000000
3    35.000000
4    35.000000
5    29.699118
'''
# 서로 이웃하고 있는 데이터끼리는 유사성을 가질 가능성이 높다. (시계열 데이터는 더욱 그러하다.)
df_2['deck_ffill'] = df_2['deck'].fillna(method='ffill') # 결측치가 있을 경우 결측치 이전에 유효하는 값으로 변경
df_2['deck_bfill'] = df_2['deck'].fillna(method='bfill') # 결측치가 있을 경우 결측치 이후에 유효하는 값으로 변경

print(df_2[['deck', 'deck_ffill', 'deck_bfill']].head(12))
'''
   deck deck_ffill deck_bfill
0   NaN        NaN          C
1     C          C          C
2   NaN          C          C
3     C          C          C
4   NaN          C          E
5   NaN          C          E
6     E          E          E
7   NaN          E          G
8   NaN          E          G
9   NaN          E          G
10    G          G          G
11    C          C          C
'''

# Title: 인덱스 다루기

df = sns.load_dataset('mpg')

print(df.head())
'''
    mpg  cylinders  displacement  horsepower  weight  acceleration  model_year origin                       name
0  18.0          8         307.0       130.0    3504          12.0          70    usa  chevrolet chevelle malibu
1  15.0          8         350.0       165.0    3693          11.5          70    usa          buick skylark 320
2  18.0          8         318.0       150.0    3436          11.0          70    usa         plymouth satellite
3  16.0          8         304.0       150.0    3433          12.0          70    usa              amc rebel sst
4  17.0          8         302.0       140.0    3449          10.5          70    usa                ford torino
'''

df.set_index('name', inplace=True)

print(df.sort_index().head())
'''
                          mpg  cylinders  displacement  horsepower  weight  acceleration  model_year origin
name                                                                                                       
amc ambassador brougham  13.0          8         360.0       175.0    3821          11.0          73    usa
amc ambassador dpl       15.0          8         390.0       190.0    3850           8.5          70    usa
amc ambassador sst       17.0          8         304.0       150.0    3672          11.5          72    usa
amc concord              24.3          4         151.0        90.0    3003          20.1          80    usa
amc concord              19.4          6         232.0        90.0    3210          17.2          78    usa
'''

df.sort_index(inplace=True, ascending=False)

print(df.head())
'''
                       mpg  cylinders  displacement  horsepower  weight  acceleration  model_year  origin
name                                                                                                     
vw rabbit custom      31.9          4          89.0        71.0    1925          14.0          79  europe
vw rabbit c (diesel)  44.3          4          90.0        48.0    2085          21.7          80  europe
vw rabbit             29.0          4          90.0        70.0    1937          14.2          76  europe
vw rabbit             41.5          4          98.0        76.0    2144          14.7          80  europe
vw pickup             44.0          4          97.0        52.0    2130          24.6          82  europe
'''

df.reset_index(inplace=True)

print(df.head())
'''
                   name   mpg  cylinders  displacement  horsepower  weight  acceleration  model_year  origin
0      vw rabbit custom  31.9          4          89.0        71.0    1925          14.0          79  europe
1  vw rabbit c (diesel)  44.3          4          90.0        48.0    2085          21.7          80  europe
2             vw rabbit  29.0          4          90.0        70.0    1937          14.2          76  europe
3             vw rabbit  41.5          4          98.0        76.0    2144          14.7          80  europe
4             vw pickup  44.0          4          97.0        52.0    2130          24.6          82  europe
'''

# Title: 필터링
'''
크게 불리언 인덱싱과 isin() 메서드를 사용
'''

df = sns.load_dataset('mpg')

print(df.tail(10))
print(df['cylinders'].unique())
'''
[8 4 6 3 5]
'''

filtered_bool = df['cylinders'] == 4

print(filtered_bool.tail(10))
'''
388     True
389    False
390     True
391     True
392     True
393     True
394     True
395     True
396     True
397     True
'''

print(df.loc[filtered_bool, ])
'''
      mpg  cylinders  displacement  horsepower  weight  acceleration  model_year  origin                          name
14   24.0          4         113.0        95.0    2372          15.0          70   japan         toyota corona mark ii
18   27.0          4          97.0        88.0    2130          14.5          70   japan                  datsun pl510
19   26.0          4          97.0        46.0    1835          20.5          70  europe  volkswagen 1131 deluxe sedan
20   25.0          4         110.0        87.0    2672          17.5          70  europe                   peugeot 504
21   24.0          4         107.0        90.0    2430          14.5          70  europe                   audi 100 ls
..    ...        ...           ...         ...     ...           ...         ...     ...                           ...
393  27.0          4         140.0        86.0    2790          15.6          82     usa               ford mustang gl
394  44.0          4          97.0        52.0    2130          24.6          82  europe                     vw pickup
395  32.0          4         135.0        84.0    2295          11.6          82     usa                 dodge rampage
396  28.0          4         120.0        79.0    2625          18.6          82     usa                   ford ranger
397  31.0          4         119.0        82.0    2720          19.4          82     usa                    chevy s-10

[204 rows x 9 columns]
'''

filtered_bool = (df['cylinders'] == 4) & (df['horsepower'] >= 100)

print(df.loc[filtered_bool, ['cylinders', 'horsepower', 'name']])
'''
     cylinders  horsepower              name
23           4       113.0          bmw 2002
76           4       112.0   volvo 145e (sw)
120          4       112.0       volvo 144ea
122          4       110.0         saab 99le
180          4       115.0         saab 99le
207          4       102.0         volvo 245
242          4       110.0          bmw 320i
271          4       105.0  plymouth sapporo
276          4       115.0        saab 99gle
323          4       105.0        dodge colt
357          4       100.0      datsun 200sx
'''

filtered_isin = df['name'].isin(
    ['ford maverick', 'ford mustang ii', 'chevrolet impala']
)

print(df.loc[filtered_isin, ])
'''
      mpg  cylinders  displacement  horsepower  weight  acceleration  model_year origin              name
6    14.0          8         454.0       220.0    4354           9.0          70    usa  chevrolet impala
17   21.0          6         200.0        85.0    2587          16.0          70    usa     ford maverick
38   14.0          8         350.0       165.0    4209          12.0          71    usa  chevrolet impala
62   13.0          8         350.0       165.0    4274          12.0          72    usa  chevrolet impala
100  18.0          6         250.0        88.0    3021          16.5          73    usa     ford maverick
103  11.0          8         400.0       150.0    4997          14.0          73    usa  chevrolet impala
126  21.0          6         200.0         NaN    2875          17.0          74    usa     ford maverick
155  15.0          6         250.0        72.0    3158          19.5          75    usa     ford maverick
166  13.0          8         302.0       129.0    3169          12.0          75    usa   ford mustang ii
193  24.0          6         200.0        81.0    3012          17.6          76    usa     ford maverick
'''

print(df.loc[filtered_isin, ].sort_values('horsepower', ascending=False))

# Title: 새로운 열 만들기

import numpy as np

df['ratio'] = (df['mpg'] / df['weight']) * 100
print(df.head())
'''
    mpg  cylinders  displacement  horsepower  weight  acceleration  model_year origin                       name     ratio
0  18.0          8         307.0       130.0    3504          12.0          70    usa  chevrolet chevelle malibu  0.513699
1  15.0          8         350.0       165.0    3693          11.5          70    usa          buick skylark 320  0.406174
'''

num = pd.Series([-2, -1, 1, 2])
print(np.where(num >= 0))
print(np.where(num >= 0, '양수', '음수'))
'''
(array([2, 3]),)
['음수' '음수' '양수' '양수']
'''

df['horse_power_div'] = np.where(
    df['horsepower'] < 100, '100 미만',
    np.where((df['horsepower'] >= 100) & (df['horsepower'] < 200), '100 이상',
             np.where(df['horsepower'] >= 200, '200 이상', '기타'))
)
print(df)
'''
      mpg  cylinders  displacement  horsepower  weight  acceleration  model_year  origin                       name     ratio horse_power_div
0    18.0          8         307.0       130.0    3504          12.0          70     usa  chevrolet chevelle malibu  0.513699          100 이상
..    ...        ...           ...         ...     ...           ...         ...     ...                        ...       ...             ...
397  31.0          4         119.0        82.0    2720          19.4          82     usa                 chevy s-10  1.139706          100 미만
'''

# Title: 데이터 프레임 합치기

df1 = pd.DataFrame({
    "A":["A0", "A1"],
    "B":["B0", "B1"]
}, index=[0, 1])

df2 = pd.DataFrame({
    "A":["A2", "A3"],
    "B":["B2", "B3"]
}, index=[2, 3])

df3 = pd.DataFrame({
    "A":["A4", "A5"],
    "B":["B4", "B5"]
}, index=[4, 5])

print(pd.concat([df1, df2, df3]))
'''
    A   B
0  A0  B0
1  A1  B1
2  A2  B2
3  A3  B3
4  A4  B4
5  A5  B5
'''

df4 = pd.DataFrame({
    "A":["A4", "A5"],
}, index=[1, 3])
print(pd.concat([df1, df4]))
'''
    A    B
0  A0   B0
1  A1   B1
1  A4  NaN
3  A5  NaN
'''

print(pd.concat([df1, df4], ignore_index=True)) # 기존 인덱스 무시
'''
    A    B
0  A0   B0
1  A1   B1
2  A4  NaN
3  A5  NaN
'''

print(pd.concat([df1, df4], axis=1)) # 행이 아닌 열 기준으로 합치기
'''
     A    B    A
0   A0   B0  NaN
1   A1   B1   A4
3  NaN  NaN   A5
'''

print(pd.concat([df1, df4], axis=1, join='inner')) # 행이 아닌 열 기준으로 합치기. NaN 이 있는 행은 제외
'''
    A   B   A
1  A1  B1  A4
'''

left = pd.DataFrame({
    "key":["K0", "K1"],
    "A":["A0", "A1"],
    "B":["B0", "B1"]
})

right = pd.DataFrame({
    "key":["K0", "K2"],
    "C":["A2", "A3"],
    "D":["B2", "B3"]
})

print(pd.merge(left, right, on='key')) # default: inner join
'''
  key   A   B   C   D
0  K0  A0  B0  A2  B2
'''

print(pd.merge(left, right, on='key', how='left'))
'''
  key   A   B    C    D
0  K0  A0  B0   A2   B2
1  K1  A1  B1  NaN  NaN
'''

print(pd.merge(left, right, on='key', how='right'))
'''
  key    A    B   C   D
0  K0   A0   B0  A2  B2
1  K2  NaN  NaN  A3  B3
'''

print(pd.merge(left, right, on='key', how='outer'))
'''

  key    A    B    C    D
0  K0   A0   B0   A2   B2
1  K1   A1   B1  NaN  NaN
2  K2  NaN  NaN   A3   B3
'''

left = pd.DataFrame({
    "key_left":["K0", "K1"],
    "A":["A0", "A1"],
    "B":["B0", "B1"]
})

right = pd.DataFrame({
    "key_right":["K0", "K2"],
    "C":["A2", "A3"],
    "D":["B2", "B3"]
})

print(pd.merge(left, right, left_on='key_left', right_on='key_right', how='left'))
'''
  key_left   A   B key_right    C    D
0       K0  A0  B0        K0   A2   B2
1       K1  A1  B1       NaN  NaN  NaN
'''

print(left.merge(right, left_on='key_left', right_on='key_right', how='inner'))
'''
  key_left   A   B key_right   C   D
0       K0  A0  B0        K0  A2  B2
'''

left = pd.DataFrame({
    "A":["A0", "A1"],
    "B":["B0", "B1"]
})

right = pd.DataFrame({
    "C":["A2", "A3"],
    "D":["B2", "B3"]
})

print(left.join(right)) # 행 인덱스 기준으로 결합
'''
    A   B   C   D
0  A0  B0  A2  B2
1  A1  B1  A3  B3
'''

# Title: 데이터 재구조화

df = sns.load_dataset('penguins')
print(df)
'''
    species     island  bill_length_mm  bill_depth_mm  flipper_length_mm  body_mass_g     sex
0    Adelie  Torgersen            39.1           18.7              181.0       3750.0    Male
1    Adelie  Torgersen            39.5           17.4              186.0       3800.0  Female
2    Adelie  Torgersen            40.3           18.0              195.0       3250.0  Female
3    Adelie  Torgersen             NaN            NaN                NaN          NaN     NaN
4    Adelie  Torgersen            36.7           19.3              193.0       3450.0  Female
..      ...        ...             ...            ...                ...          ...     ...
339  Gentoo     Biscoe             NaN            NaN                NaN          NaN     NaN
340  Gentoo     Biscoe            46.8           14.3              215.0       4850.0  Female
341  Gentoo     Biscoe            50.4           15.7              222.0       5750.0    Male
342  Gentoo     Biscoe            45.2           14.8              212.0       5200.0  Female
343  Gentoo     Biscoe            49.9           16.1              213.0       5400.0    Male

[344 rows x 7 columns]
'''

print(df.melt(id_vars=['species', 'island'])) # id_vars 값 기준으로 나머지 열들은 variable에 넣고 value를 나영ㄹ
'''
     species     island        variable   value
0     Adelie  Torgersen  bill_length_mm    39.1
1     Adelie  Torgersen  bill_length_mm    39.5
2     Adelie  Torgersen  bill_length_mm    40.3
3     Adelie  Torgersen  bill_length_mm     NaN
4     Adelie  Torgersen  bill_length_mm    36.7
...      ...        ...             ...     ...
1715  Gentoo     Biscoe             sex     NaN
1716  Gentoo     Biscoe             sex  Female
1717  Gentoo     Biscoe             sex    Male
1718  Gentoo     Biscoe             sex  Female
1719  Gentoo     Biscoe             sex    Male
'''

# index: 행 인덱스, columns: 열 인덱스, values: 데이터 값, aggfunc: 데이터 집계 함수
print(df.pivot_table(
    index='species',
    columns='island',
    values='bill_length_mm',
    aggfunc='mean'
))
'''
island        Biscoe      Dream  Torgersen
species
Adelie     38.975000  38.501786   38.95098
Chinstrap        NaN  48.833824        NaN
Gentoo     47.504878        NaN        NaN
'''

print(df.pivot_table(
    index=['species', 'sex'],
    columns='island',
    values=['bill_length_mm', 'flipper_length_mm'],
    aggfunc=['mean', 'count']
))
'''
                           mean                                                                          count
                 bill_length_mm                       flipper_length_mm                         bill_length_mm                 flipper_length_mm
island                   Biscoe      Dream  Torgersen            Biscoe       Dream   Torgersen         Biscoe Dream Torgersen            Biscoe Dream Torgersen
species   sex
Adelie    Female      37.359091  36.911111  37.554167        187.181818  187.851852  188.291667           22.0  27.0      24.0              22.0  27.0      24.0
          Male        40.590909  40.071429  40.586957        190.409091  191.928571  194.913043           22.0  28.0      23.0              22.0  28.0      23.0
Chinstrap Female            NaN  46.573529        NaN               NaN  191.735294         NaN            NaN  34.0       NaN               NaN  34.0       NaN
          Male              NaN  51.094118        NaN               NaN  199.911765         NaN            NaN  34.0       NaN               NaN  34.0       NaN
Gentoo    Female      45.563793        NaN        NaN        212.706897         NaN         NaN           58.0   NaN       NaN              58.0   NaN       NaN
          Male        49.473770        NaN        NaN        221.540984         NaN         NaN           61.0   NaN       NaN              61.0   NaN       NaN
'''

print(df.pivot_table(
    index=['species', 'sex'],
    columns='island',
    values=['bill_length_mm', 'flipper_length_mm'],
    aggfunc=['mean', 'count']
).stack())
'''
                                     mean                            count
                           bill_length_mm flipper_length_mm bill_length_mm flipper_length_mm
species   sex    island
Adelie    Female Biscoe         37.359091        187.181818           22.0              22.0
                 Dream          36.911111        187.851852           27.0              27.0
                 Torgersen      37.554167        188.291667           24.0              24.0
          Male   Biscoe         40.590909        190.409091           22.0              22.0
                 Dream          40.071429        191.928571           28.0              28.0
                 Torgersen      40.586957        194.913043           23.0              23.0
Chinstrap Female Dream          46.573529        191.735294           34.0              34.0
          Male   Dream          51.094118        199.911765           34.0              34.0
Gentoo    Female Biscoe         45.563793        212.706897           58.0              58.0
          Male   Biscoe         49.473770        221.540984           61.0              61.0
'''

print(df.pivot_table(
    index=['species', 'sex'],
    columns='island',
    values=['bill_length_mm', 'flipper_length_mm'],
    aggfunc=['mean', 'count']
).stack().unstack())
'''
                           mean                                                                          count
                 bill_length_mm                       flipper_length_mm                         bill_length_mm                 flipper_length_mm
island                   Biscoe      Dream  Torgersen            Biscoe       Dream   Torgersen         Biscoe Dream Torgersen            Biscoe Dream Torgersen
species   sex
Adelie    Female      37.359091  36.911111  37.554167        187.181818  187.851852  188.291667           22.0  27.0      24.0              22.0  27.0      24.0
          Male        40.590909  40.071429  40.586957        190.409091  191.928571  194.913043           22.0  28.0      23.0              22.0  28.0      23.0
Chinstrap Female            NaN  46.573529        NaN               NaN  191.735294         NaN            NaN  34.0       NaN               NaN  34.0       NaN
          Male              NaN  51.094118        NaN               NaN  199.911765         NaN            NaN  34.0       NaN               NaN  34.0       NaN
Gentoo    Female      45.563793        NaN        NaN        212.706897         NaN         NaN           58.0   NaN       NaN              58.0   NaN       NaN
          Male        49.473770        NaN        NaN        221.540984         NaN         NaN           61.0   NaN       NaN              61.0   NaN       NaN
'''

# Title: 데이터 프레임에 함수 적용

df = sns.load_dataset('penguins')
print(df['bill_length_mm'].apply(np.sqrt))
'''
0      6.252999
         ...   
339         NaN
343    7.063993
'''

def mm_to_cm(num):
    return num / 10

print(df['bill_length_mm'].apply(mm_to_cm))
'''
0      3.91
       ... 
343    4.99
'''

def num_null(data):
    null_vec = pd.isnull(data)
    null_count = np.sum(null_vec)

    return null_count

print(df.apply(num_null))
'''
species               0
island                0
bill_length_mm        2
bill_depth_mm         2
flipper_length_mm     2
body_mass_g           2
sex                  11
'''

# 그룹 연산하기
'''
count 누락 값을 제외한 데이터 수
size 누락 값을 포함한 데이터 수
mean 평균
std 표준편차
var 분산
min 최솟값
max 최댓값
quantile(q=0.25) 백분위 25%
quantile(q=0.50) 백분위 50%
quantile(q=0.75) 백분위 75%
sum 전체 합
describe 데이터 수, 평균, 표준편차, 최솟값, 백분위수(25, 50, 75), 최댓값 반환
first 첫 번째 행 반환
last 마지막 행 반환
nth n번째 행 반환
'''

df_group = df.groupby(['species'])

for key, group in df_group:
    print(key)
    print(group.head(2))
'''
('Adelie',)
  species     island  bill_length_mm  bill_depth_mm  flipper_length_mm  body_mass_g     sex
0  Adelie  Torgersen            39.1           18.7              181.0       3750.0    Male
1  Adelie  Torgersen            39.5           17.4              186.0       3800.0  Female
('Chinstrap',)
       species island  bill_length_mm  bill_depth_mm  flipper_length_mm  body_mass_g     sex
152  Chinstrap  Dream            46.5           17.9              192.0       3500.0  Female
153  Chinstrap  Dream            50.0           19.5              196.0       3900.0    Male
('Gentoo',)
    species  island  bill_length_mm  bill_depth_mm  flipper_length_mm  body_mass_g     sex
220  Gentoo  Biscoe            46.1           13.2              211.0       4500.0  Female
221  Gentoo  Biscoe            50.0           16.3              230.0       5700.0    Male
'''

df_group = df.groupby(['species', 'sex', 'island'])

print(df_group.mean())
'''
                            bill_length_mm  bill_depth_mm  flipper_length_mm  body_mass_g
species   sex    island                                                                  
Adelie    Female Biscoe          37.359091      17.704545         187.181818  3369.318182
                 Dream           36.911111      17.618519         187.851852  3344.444444
                 Torgersen       37.554167      17.550000         188.291667  3395.833333
          Male   Biscoe          40.590909      19.036364         190.409091  4050.000000
                 Dream           40.071429      18.839286         191.928571  4045.535714
                 Torgersen       40.586957      19.391304         194.913043  4034.782609
Chinstrap Female Dream           46.573529      17.588235         191.735294  3527.205882
          Male   Dream           51.094118      19.252941         199.911765  3938.970588
Gentoo    Female Biscoe          45.563793      14.237931         212.706897  4679.741379
          Male   Biscoe          49.473770      15.718033         221.540984  5484.836066
'''

def min_max(x):
    return x.max() - x.min()

print(df.groupby(['species', 'sex', 'island']).agg(min_max))
'''
                            bill_length_mm  bill_depth_mm  flipper_length_mm  body_mass_g
species   sex    island                                                                  
Adelie    Female Biscoe                6.0            4.7               27.0       1050.0
                 Dream                10.1            3.8               24.0        800.0
                 Torgersen             7.6            3.4               20.0        900.0
          Male   Biscoe                8.0            3.9               23.0       1225.0
                 Dream                 7.8            4.2               30.0       1225.0
                 Torgersen            11.4            3.9               29.0       1375.0
Chinstrap Female Dream                17.1            3.0               24.0       1450.0
          Male   Dream                 7.3            3.3               25.0       1550.0
Gentoo    Female Biscoe                9.6            2.4               19.0       1250.0
          Male   Biscoe               15.2            3.2               23.0       1550.0
'''

print(df.groupby(['species', 'sex', 'island']).agg(['max', 'min']))
print(df.groupby(['species', 'sex', 'island']).agg({'bill_length_mm' : ['max', 'min']}))
'''
                           bill_length_mm      
                                      max   min
species   sex    island                        
Adelie    Female Biscoe              40.5  34.5
                 Dream               42.2  32.1
                 Torgersen           41.1  33.5
          Male   Biscoe              45.6  37.6
                 Dream               44.1  36.3
                 Torgersen           46.0  34.6
Chinstrap Female Dream               58.0  40.9
          Male   Dream               55.8  48.5
Gentoo    Female Biscoe              50.5  40.9
          Male   Biscoe              59.6  44.4
'''

def z_score(x):
    z = (x - x.mean()) / x.std()
    return(z) 

print(df.groupby(['species'])['bill_length_mm'].transform(z_score))
print(df.groupby(['species'])['bill_length_mm'].apply(z_score))
'''
0      0.115870
1      0.266054
2      0.566421
3           NaN
         ...   
343    0.777168
Name: bill_length_mm, Length: 344, dtype: float64
'''

print(df.groupby(['species']).filter(lambda x: x['bill_length_mm'].mean() >= 40))
'''
       species  island  bill_length_mm  bill_depth_mm  flipper_length_mm  body_mass_g     sex
152  Chinstrap   Dream            46.5           17.9              192.0       3500.0  Female
..         ...     ...             ...            ...                ...          ...     ...
339     Gentoo  Biscoe             NaN            NaN                NaN          NaN     NaN

[192 rows x 7 columns]
'''

# 시계열 데이터 다루기

df = sns.load_dataset('taxis')
print(df)
'''
                  pickup             dropoff  passengers  distance  fare   tip  tolls  total   color      payment            pickup_zone                      dropoff_zone pickup_borough dropoff_borough
0    2019-03-23 20:21:09 2019-03-23 20:27:24           1      1.60   7.0  2.15    0.0  12.95  yellow  credit card        Lenox Hill West               UN/Turtle Bay South      Manhattan       Manhattan
1    2019-03-04 16:11:55 2019-03-04 16:19:00           1      0.79   5.0  0.00    0.0   9.30  yellow         cash  Upper West Side South             Upper West Side South      Manhattan       Manhattan
2    2019-03-27 17:53:01 2019-03-27 18:00:25           1      1.37   7.5  2.36    0.0  14.16  yellow  credit card          Alphabet City                      West Village      Manhattan       Manhattan
3    2019-03-10 01:23:59 2019-03-10 01:49:51           1      7.70  27.0  6.15    0.0  36.95  yellow  credit card              Hudson Sq                    Yorkville West      Manhattan       Manhattan
4    2019-03-30 13:27:42 2019-03-30 13:37:14           3      2.16   9.0  1.10    0.0  13.40  yellow  credit card           Midtown East                    Yorkville West      Manhattan       Manhattan
...                  ...                 ...         ...       ...   ...   ...    ...    ...     ...          ...                    ...                               ...            ...             ...
6428 2019-03-31 09:51:53 2019-03-31 09:55:27           1      0.75   4.5  1.06    0.0   6.36   green  credit card      East Harlem North              Central Harlem North      Manhattan       Manhattan
6429 2019-03-31 17:38:00 2019-03-31 18:34:23           1     18.74  58.0  0.00    0.0  58.80   green  credit card                Jamaica  East Concourse/Concourse Village         Queens           Bronx
6430 2019-03-23 22:55:18 2019-03-23 23:14:25           1      4.14  16.0  0.00    0.0  17.30   green         cash    Crown Heights North                    Bushwick North       Brooklyn        Brooklyn
6431 2019-03-04 10:09:25 2019-03-04 10:14:29           1      1.12   6.0  0.00    0.0   6.80   green  credit card          East New York      East Flatbush/Remsen Village       Brooklyn        Brooklyn
6432 2019-03-13 19:31:22 2019-03-13 19:48:02           1      3.85  15.0  3.36    0.0  20.16   green  credit card            Boerum Hill                   Windsor Terrace       Brooklyn        Brooklyn

[6433 rows x 14 columns]
'''

print(df.info())
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 6433 entries, 0 to 6432
Data columns (total 14 columns):
 #   Column           Non-Null Count  Dtype         
---  ------           --------------  -----         
 0   pickup           6433 non-null   datetime64[ns]
 1   dropoff          6433 non-null   datetime64[ns]
 2   passengers       6433 non-null   int64         
 3   distance         6433 non-null   float64       
 4   fare             6433 non-null   float64       
 5   tip              6433 non-null   float64       
 6   tolls            6433 non-null   float64       
 7   total            6433 non-null   float64       
 8   color            6433 non-null   object        
 9   payment          6389 non-null   object        
 10  pickup_zone      6407 non-null   object        
 11  dropoff_zone     6388 non-null   object        
 12  pickup_borough   6407 non-null   object        
 13  dropoff_borough  6388 non-null   object        
dtypes: datetime64[ns](2), float64(5), int64(1), object(6)
memory usage: 703.
'''

print(df['pickup'].dt.year) # dt 접근자를 이용하면 datetime 타입의 열에 한 번에 접근할 수 있다.
print(df['pickup'].dt.month)
print(df['pickup'].dt.day)

# 시간 순으로 데이터 정리
df.sort_values('pickup', inplace=True)
df.reset_index(drop=True, inplace=True)
print(df)
'''
                  pickup             dropoff  passengers  distance  fare   tip  tolls  total   color      payment                    pickup_zone                   dropoff_zone pickup_borough dropoff_borough
0    2019-02-28 23:29:03 2019-02-28 23:32:35           1      0.90   5.0  0.00    0.0   6.30   green         cash                    Old Astoria  Long Island City/Queens Plaza         Queens          Queens
...                  ...                 ...         ...       ...   ...   ...    ...    ...     ...          ...                            ...                            ...            ...             ...
6428 2019-03-31 22:13:37 2019-03-31 22:22:50           1      1.00   7.5  0.70    0.0  12.00  yellow  credit card                           SoHo                Lower East Side      Manhattan       Manhattan
'''

print(df['dropoff'] - df['pickup'])
'''
0      0 days 00:03:32
             ...      
6428   0 days 00:09:13
Length: 6433, dtype: timedelta64[ns]
'''

df.set_index('pickup', inplace=True)
print(df.loc['2019-02']) # 2019년 2월에 해당하는 정보만 추출
'''
                                dropoff  passengers  distance  fare  tip  tolls  total  color payment  pickup_zone                   dropoff_zone pickup_borough dropoff_borough
pickup                                                                                                                                                                          
2019-02-28 23:29:03 2019-02-28 23:32:35           1       0.9   5.0  0.0    0.0    6.3  green    cash  Old Astoria  Long Island City/Queens Plaza         Queens          Queens
'''

# Title: 시계열 데이터 만들기

df = pd.date_range(start='2021-01-01',
              end='2021-12-31',
              freq='M') # 간격 D 일, 3D 3일, W 주, H 시간, T 분, S 초, M 월말, MS 월초 등

print(df)
'''
DatetimeIndex(['2021-01-31', '2021-02-28', '2021-03-31', '2021-04-30',
               '2021-05-31', '2021-06-30', '2021-07-31', '2021-08-31',
               '2021-09-30', '2021-10-31', '2021-11-30', '2021-12-31'],
              dtype='datetime64[ns]', freq='M')
'''