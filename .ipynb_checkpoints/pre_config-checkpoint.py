# encoding: utf-8
# dialect+driver://username:password@host:port/database
from urllib import parse
DIALECT = 'mysql'  # 要用是什么数据库，我使用的是 mysql
DRIVER = 'pymysql'  # 连接数据库驱动，pymysql 是 mysql 的驱动
USERNAME = 'pdl'  # 用户名 ，你的数据库用户名
PASSWORD = 'Suxiehe123!@#'  # 密码 ，你的数据库密码
PASSWORD = parse.quote_plus(PASSWORD)

HOST = '118.195.234.43'  # 服务器 ，数据库所在服务器的ip，本地即 127.0.0.1
PORT = '3306'  # 端口 ，数据库的默认端口 3306
DATABASE = 'shijianqiang'  # 数据库名 ，你需要链接的具体数据库的名字 ，这里是报修数据库的名字

SQLALCHEMY_DATABASE_URI = "{}+{}://{}:{}@{}:{}/{}?charset=utf8".format(DIALECT, DRIVER, USERNAME, PASSWORD, HOST, PORT,
                                                                       DATABASE)  # 拼接成数据库的 URI ，一般不需要修改
SQLALCHEMY_TRACK_MODIFICATIONS = False  # 用于追踪数据库修改 ， 默认为True ，设置为 True 会增加内存消耗