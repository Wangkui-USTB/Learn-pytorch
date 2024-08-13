# Argparse模块常见用法
# 		https://blog.csdn.net/qq_45957458/article/details/123795540
# argparse模块是Python中用来读取命令行参数的模块，程序定义它需要的参数，
# 然后argparse会自动从sys.argv中解析出这些参数。

import  argparse
# 创建一个解释器
parser=argparse.ArgumentParser(description="Test")

# 为解释器添加参数
parser.add_argument('-n', '--name', default="Torture")
parser.add_argument('-u', '--university', default="UCAS", help="发生什么事了")
print(parser.parse_args())

# 获取参数
# 解析参数
args = parser.parse_args()
print("data :",args.name)

#    运行：python .\21_argparse.py --name=wk
# 结果：
# Namespace(name='wk', university='UCAS')
# data : wk