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

# 데이터 요약 정보 및 통곗값 살펴보기

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

# 걸측치 처리하기
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

# 인덱스 다루기

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

# 필터링
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
