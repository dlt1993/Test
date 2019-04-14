# encoding = utf-8
import pandas as pd 
import os

os.getcwd()
os.chdir('E:\\数据分析师的修养\\实战项目\\数据分析入门\\seaborn-data-master')
df = pd.read_csv('tips.csv')

#打印前五行
df.head()

#打印后五行
df.tail()

#打印列名
df.columns

#打印行名
df.index

#打印10~20行前三列数据
df.ix[10:20,0:3]

#提取不连续的行和列的数据
df.iloc[[1,3,5],[2,4]]

#专门提取某一个数据
df.iat[3,2]

#舍弃数据前两列
print df.drop(df.columns[1, 2], axis = 1) #舍弃数据前两列
print df.drop(df.columns[[1, 2]], axis = 0) #舍弃数据前两行

#打印维度
df.shape

#选取第3行
df.iloc[3]

#选取第2到第3行
df.iloc[2:4]

#选取第0行1列的元素
df.iloc[0,1]


#筛选出小费大于$8的数据
df[df.tip > 8]

#筛选同样可以用“或”和“且”作为筛选条件
df[(df.tip > 7)|(df.total_bill > 50)]    #筛选出小费大于$7或总账单大于$50的数据
df[(df.tip > 7)&(df.total_bill > 50)]    #筛选出小费大于$7且总账单大于$50的数据 

#加上筛选条件后，附带day和time
df[['day','time']][(df.tip > 7)|(df.total_bill > 50)]


#描述统计
df.describe()

#数据转置
df.T

#数据排列
df.sort_values(by = 'tip')  #按照tip列升序排列


#缺失值处理
import json     #python有许多内置或第三方模块可以将JSON字符串转换成python字典对象
import pandas as pd
import numpy as np 
from pandas import DataFrame

path = 'E:\\数据分析师的修养\\实战项目\\数据分析入门\\usagov_bitly_data2012-03-16-1331923249.txt'

records = [json.loads(line) for line in open(path)]
frame = DataFrame(records)
frame['tz']       #从以上输出值可以看出数据存在未知或缺失值，接着咱们来处理缺失值。


frame['tz'].fillna(1111111111111)          #以数字代替缺失值
frame['tz'].fillna('Yujie2333333333333')   #用字符串代替缺失值
frame['tz'].fillna(method = 'pad')         #用前一个数据代替缺失值 
frame['tz'].fillna(method = 'bfill')       #用后一个数据代替缺失值

frame['tz'].dropna(axis = 0)               #删除缺失行
frame['tz'].dropna(axis = 1)               #删除缺失列

import pandas as pd 
import numpy as np
#创建一个6*4的数据框，randn函数用于创建随机数
caf_data = pd.DataFrame(np.random.randn(6,4),columns = list('ABCD'))
caf_data

#把第二列数据设置为缺失值
caf_data.ix[2,:] = np.nan
caf_data

#利用插值法填补空缺值
caf_data.interpolate()


#数据分组
group = df.groupby('day')       #按day这一列进行分组
group.first()                   #打印每一组的第一行数据
group.last()                    #打印每一组的最后一行数据

#替换值
Series = pd.Series([0,1,2,3,4,5])

#数值替换，例如0换成10000000000000
Series.replace(0,10000000000000)

#列和列的替换同理
Series.replace([0,1,2,3,4,5],[11111,222222,3333333,44444,55555,666666])



#统计分析
#T检验
#独立样本T检验
import pandas as pd 
from scipy.stats import ttest_ind
os.chdir('E:\\数据分析师的修养\\实战项目\\数据分析入门')
t_test = pd.read_excel('t_test.xlsx')
Group1 = t_test[t_test['group'] == 1]['data']
Group2 = t_test[t_test['group'] == 2]['data']
ttest_ind(Group1,Group2)

# ttest_ind默认两组数据方差齐性的，如果想要设置默认方差不齐，可以设置equal_var=False
ttest_ind(Group1,Group2,equal_var = True)
ttest_ind(Group1,Group2,equal_var = False)

#配对样本T检验
import pandas as pd
from scipy.stats import ttest_rel
t_test = pd.read_excel('t_test.xlsx')
Group1 = t_test[t_test['group'] == 1]['data']
Group2 = t_test[t_test['group'] == 2]['data']
ttest_rel(Group1,Group2)


#单因素方差分析
import pandas as pd
from scipy import stats
t_test = pd.read_excel('t_test.xlsx')
Group1 = t_test[t_test['group'] == 1]['data']
Group2 = t_test[t_test['group'] == 2]['data']
w,p = stats.levene(*args)
#levene方差齐性检验。levene(*args, **kwds)  Perform Levene test for equal variances.如果p<0.05，则方差不齐

#进行方差分析
f,p = stats.f_oneway(*args)


#多因素方差分析
import pandas as pd 
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
MANOVA = pd.read_table('MANOVA.txt')



formula = 'weight ~ C(id) + C(nutrient) + C(id) : C(nutrient)'
anova_results = anova_lm(ols(formula,MANOVA).fit())




