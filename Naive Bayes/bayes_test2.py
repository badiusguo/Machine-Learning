
import numpy as np
import random
import re



def textParse(bigString):                                                   #将字符串转换为字符列表
    # * 会匹配0个或多个规则，split会将字符串分割成单个字符【python3.5+】; 这里使用\W 或者\W+ 都可以将字符数字串分割开，产生的空字符将会在后面的列表推导式中过滤掉
    listOfTokens = re.split(r'\W+', bigString)                              #将特殊符号作为切分标志进行字符串切分，即非字母、非数字
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


if __name__ == '__main__':
    x=textParse(open('email/ham/6.txt').read())
    x