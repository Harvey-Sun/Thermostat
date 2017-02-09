# -*- coding:utf-8 -*-

from CloudQuant import MiniSimulator
import numpy as np
import pandas as pd

username = 'Harvey_Sun'
password = 'P948894dgmcsy'
Strategy_Name = 'Thermostat_buy_and_hold'

INIT_CAP = 100000000
START_DATE = '20130101'
END_DATE = '20161231'
Fee_Rate = 0.001
program_path = 'C:/cStrategy/'

window_cmi = 30
window_atr = 10
window_ = 3


def initial(sdk):
    # 准备数据
    sdk.prepareData(['LZ_GPA_INDEX_CSI500MEMBER', 'LZ_GPA_SLCIND_STOP_FLAG'])
    base_log = pd.read_csv('buy_and_hold.csv', index_col=0)
    sdk.setGlobal('base_log', base_log)


def init_per_day(sdk):
    # 获取当天中证500成分股
    in_zz500 = pd.Series(sdk.getFieldData('LZ_GPA_INDEX_CSI500MEMBER')[-1]) == 1
    stock_list = sdk.getStockList()
    zz500 = list(pd.Series(stock_list)[in_zz500])
    sdk.setGlobal('zz500', zz500)
    # 获取仓位信息
    positions = sdk.getPositions()
    sdk.sdklog(len(positions), '持有股票数量')
    stock_with_position = [i.code for i in positions]
    # 找到中证500外的有仓位的股票
    out_zz500_stock = list(set(stock_with_position) - set(zz500))
    # 以下代码获取当天未停牌股票，即可交易股票
    not_stop = pd.isnull(sdk.getFieldData('LZ_GPA_SLCIND_STOP_FLAG')[-(window_cmi + 1):]).all(
        axis=0)  # 当日和前window1日均没有停牌的股票
    zz500_available = list(pd.Series(stock_list)[np.logical_and(in_zz500, not_stop)])
    sdk.setGlobal('zz500_available', zz500_available)
    # 以下代码获取当天被移出中证500的有仓位的股票中可交易的股票
    out_zz500_available = list(set(pd.Series(stock_list)[not_stop]).intersection(set(out_zz500_stock)))
    sdk.setGlobal('out_zz500_available', out_zz500_available)
    # 订阅所有可交易的股票
    stock_available = list(set(zz500_available + out_zz500_available))
    sdk.sdklog(len(stock_available), '订阅股票数量')
    sdk.subscribeQuote(stock_available)

    base_log = sdk.getGlobal('base_log')
    base_log_today = base_log[base_log['date'] == int(sdk.getNowDate())]
    sdk.setGlobal('base_log_today', base_log_today)


def strategy(sdk):
    base_log_today = sdk.getGlobal('base_log_today')

    minute = sdk.getNowTime()
    if int(minute) in list(base_log_today['time']):
        orders = list(base_log_today[base_log_today['time'] == int(minute)]['order'])
        clean_orders = []
        for i in orders:
            order = i[1:-1].replace('\'', '').replace(' ', '').split(',')
            ORDER= [order[0], float(order[1]), float(order[2]) , int(order[3])]
            clean_orders.append(ORDER)
        sdk.makeOrders(clean_orders)
    else:
        pass


config = {
    'username': username,
    'password': password,
    'initCapital': INIT_CAP,
    'startDate': START_DATE,
    'endDate': END_DATE,
    'strategy': strategy,
    'initial': initial,
    'preparePerDay': init_per_day,
    'feeRate': Fee_Rate,
    'strategyName': Strategy_Name,
    'logfile': '%s.log' % Strategy_Name,
    'rootpath': program_path,
    'executeMode': 'M',
    'feeLimit': 5,
    'cycle': 1,
    'dealByVolume': True,
    'allowForTodayFactors': ['LZ_GPA_INDEX_CSI500MEMBER', 'LZ_GPA_SLCIND_STOP_FLAG']
}

if __name__ == "__main__":
    # 在线运行所需代码
    import os
    config['strategyID'] = os.path.splitext(os.path.split(__file__)[1])[0]
    MiniSimulator(**config).run()


