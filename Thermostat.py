# -*- coding:utf-8 -*-

from CloudQuant import MiniSimulator
import numpy as np
import pandas as pd

username = 'Harvey_Sun'
password = 'P948894dgmcsy'
Strategy_Name = 'Thermostat'

INIT_CAP = 100000000
START_DATE = '20130101'
END_DATE = '20161231'
window_cmi = 30
window_atr = 10
window_ = 3
k1 = 0.75
k2 = 0.5
Fee_Rate = 0.001
program_path = 'C:/cStrategy/'


def initial(sdk):
    # 准备数据
    sdk.prepareData(['LZ_GPA_QUOTE_THIGH', 'LZ_GPA_QUOTE_TLOW', 'LZ_GPA_QUOTE_TCLOSE',
                     'LZ_GPA_INDEX_CSI500MEMBER', 'LZ_GPA_SLCIND_STOP_FLAG'])
    stock_position = dict()
    sdk.setGlobal('stock_position', stock_position)
    buy_and_hold = []
    buy_and_hold_time = []
    sdk.setGlobal('buy_and_hold', buy_and_hold)
    sdk.setGlobal('buy_and_hold_time', buy_and_hold_time)


def init_per_day(sdk):
    stock_position = sdk.getGlobal('stock_position')
    buy_and_hold = sdk.getGlobal('buy_and_hold')
    buy_and_hold_time = sdk.getGlobal('buy_and_hold_time')
    sdk.clearGlobal()
    sdk.setGlobal('stock_position', stock_position)
    sdk.setGlobal('buy_and_hold', buy_and_hold)
    sdk.setGlobal('buy_and_hold_time', buy_and_hold_time)

    today = sdk.getNowDate()
    sdk.sdklog(today, '========================================日期')
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
    not_stop = pd.isnull(sdk.getFieldData('LZ_GPA_SLCIND_STOP_FLAG')[-(window_cmi + 1):]).all(axis=0)  # 当日和前window1日均没有停牌的股票
    sdk.sdklog(pd.Series(sdk.getFieldData('LZ_GPA_SLCIND_STOP_FLAG')[-1], index=stock_list)[['000510', '002129']])
    zz500_available = list(pd.Series(stock_list)[np.logical_and(in_zz500, not_stop)])
    sdk.setGlobal('zz500_available', zz500_available)
    # 以下代码获取当天被移出中证500的有仓位的股票中可交易的股票
    out_zz500_available = list(set(pd.Series(stock_list)[not_stop]).intersection(set(out_zz500_stock)))
    sdk.setGlobal('out_zz500_available', out_zz500_available)
    # 订阅所有可交易的股票
    stock_available = list(set(zz500_available + out_zz500_available))
    sdk.sdklog(len(stock_available), '订阅股票数量')
    sdk.subscribeQuote(stock_available)

    # 计算CMI
    close = pd.DataFrame(sdk.getFieldData('LZ_GPA_QUOTE_TCLOSE')[-window_cmi:], columns=stock_list)[stock_available]
    high = pd.DataFrame(sdk.getFieldData('LZ_GPA_QUOTE_THIGH')[-window_cmi:], columns=stock_list)[stock_available]
    low = pd.DataFrame(sdk.getFieldData('LZ_GPA_QUOTE_TLOW')[-window_cmi:], columns=stock_list)[stock_available]
    cmi = 100 * (np.abs(close.iloc[-1] - close.iloc[0]) / (high.max() - low.min()))
    trend_stocks = cmi[cmi > 20].index
    vibrate_stocks = cmi[cmi <= 20].index
    sdk.setGlobal('trend_stocks', trend_stocks)
    sdk.setGlobal('vibrate_stocks', vibrate_stocks)
    
    # 震荡市指标计算
    # 计算价格中枢
    pivot = (high.iloc[-1] + low.iloc[-1] + close.iloc[-1]) / 3
    # 计算ATR
    temp1 = high - low
    temp2 = np.abs(high - close.shift(-1))
    temp3 = np.abs(low - close.shift(-1))
    max23 = np.where(temp2 > temp3, temp2, temp3)
    tr = np.where(temp1 > max23, temp1, max23)
    atr = pd.Series(tr[-window_atr:].mean(), index=stock_available)
    sdk.setGlobal('atr', atr)
    # 计算其他指标
    mh = high[-window_:].mean()
    ml = low[-window_:].mean()
    sdk.setGlobal('mh', mh)
    sdk.setGlobal('ml', ml)
    up_range = pd.Series(np.where(close.iloc[-1] > pivot, 0.5 * atr, 0.75 * atr), index=stock_available)
    dn_range = pd.Series(np.where(close.iloc[-1] > pivot, 0.75 * atr, 0.5 * atr), index=stock_available)
    sdk.setGlobal('up_range', up_range)
    sdk.setGlobal('dn_range', dn_range)

    # 趋势市指标计算
    ma = close.mean()
    val = close.std()
    up_line = ma + 2 * val
    dn_line = ma - 2 * val
    sdk.setGlobal('ma', ma)
    sdk.setGlobal('up_line', up_line)
    sdk.setGlobal('dn_line', dn_line)

    traded_stock = []
    sdk.setGlobal('traded_stock', traded_stock)


def strategy(sdk):
    if (sdk.getNowTime() >= '093000') & (sdk.getNowTime() < '150000'):
        today = sdk.getNowDate()
        # 获取仓位信息及有仓位的股票
        positions = sdk.getPositions()
        position_dict = dict([[i.code, i.optPosition] for i in positions])
        stock_with_position = [i.code for i in positions]
        # 找到中证500外的有仓位的股票
        zz500 = sdk.getGlobal('zz500')
        out_zz500_stock = list(set(stock_with_position) - set(zz500))
        # 找到目前有仓位且可交易的中证500外的股票
        out_zz500_available = sdk.getGlobal('out_zz500_available')
        out_zz500_tradable = list(set(out_zz500_stock).intersection(set(out_zz500_available)))
        # 获得中证500当日可交易的股票
        zz500_available = sdk.getGlobal('zz500_available')
        # 加载已交易股票
        traded_stock = sdk.getGlobal('traded_stock')
        # 有底仓的股票
        stock_position = sdk.getGlobal('stock_position')
        number = sum([stock_position[stock]['position'] for stock in stock_position.keys()]) / 2.0  # 计算有多少个全仓股
        # 无仓位股票可用资金
        available_cash = sdk.getAccountInfo().availableCash / (500 - number) if number < 500 else 0
        # 底仓开平记录
        buy_and_hold = sdk.getGlobal('buy_and_hold')
        buy_and_hold_time = sdk.getGlobal('buy_and_hold_time')
        # 获取震荡市和趋势市的股票
        trend_stocks = sdk.getGlobal('trend_stocks')
        vibrate_stocks = sdk.getGlobal('vibrate_stocks')

        if sdk.getNowTime() == '093000':
            stock_available = list(set(zz500_available + out_zz500_available))
            # 获取开盘价
            quotes = sdk.getQuotes(stock_available)
            open_prices = []
            for stock in stock_available:
                open_prices.append(quotes[stock].open)
            Open = pd.Series(open_prices, index=stock_available)
            # 计算震荡市下的信号指标
            mh = sdk.getGlobal('mh')
            ml = sdk.getGlobal('ml')
            up_range = sdk.getGlobal('up_range')
            dn_range = sdk.getGlobal('dn_range')
            buy_line = pd.Series(np.where(Open + up_range > ml, Open + up_range, ml), index=stock_available)
            sell_line = pd.Series(np.where(Open - dn_range < mh, Open - dn_range, mh), index=stock_available)
            sdk.setGlobal('buy_line', buy_line)
            sdk.setGlobal('sell_line', sell_line)
            # 建立底仓
            stock_to_build_base = list(set(zz500_available) - set(stock_position.keys()))
            base_hold = []
            date_and_time = []
            for stock in stock_to_build_base:
                price = quotes[stock].current
                volume = 100 * np.floor(available_cash * 0.5 / (100 * price))
                if volume > 0:
                    order = [stock, price, volume, 1]
                    base_hold.append(order)
                    date_and_time.append([today, '093000'])
                    stock_position[stock] = {'position': 1}
                    traded_stock.append(stock)
            sdk.makeOrders(base_hold)
            sdk.sdklog(len(traded_stock), '=======建立底仓股票数量')
            buy_and_hold += base_hold
            buy_and_hold_time += date_and_time
		
        # 去除今天已经有交易的股票，获得当下还可交易的股票
        zz500_tradable = list(set(zz500_available) - set(traded_stock))
        # 取得盘口数据
        quotes = sdk.getQuotes(zz500_tradable + out_zz500_tradable)
        # 其他数据
        ma = sdk.getGlobal('ma')
        up_line = sdk.getGlobal('up_line')
        dn_line = sdk.getGlobal('dn_line')
        atr = sdk.getGlobal('atr')
        sell_line = sdk.getGlobal('sell_line')
        buy_line = sdk.getGlobal('buy_line')

        # 考虑被移出中证500的那些股票
        sell_orders_out500 = []
        date_and_time = []
        if out_zz500_tradable:
            for stock in out_zz500_tradable:
                position = position_dict[stock]
                current_price = quotes[stock].current
                mid = ma[stock]
                if stock_position[stock]['position'] == 1:
                    order = [stock, current_price, position, -1]
                    sell_orders_out500.append(order)
                    date_and_time.append([today, sdk.getNowTime()])
                    del stock_position[stock]
                else:  # 仓位为做多
                    sig1 = (stock_position[stock]['open_mkt'] == 'T') & (current_price < mid)
                    sig2 = ((stock_position[stock]['open_mkt'] == 'V') &
                          (stock in trend_stocks) &
                          (abs(current_price - stock_position[stock]['open_price']) > 3 * atr[stock]))
                    sig3 = ((stock_position[stock]['open_mkt'] == 'V') &
                            (stock in vibrate_stocks) &
                            (current_price < sell_line[stock]))
                    if sig1 or sig2 or sig3:
                        order = [stock, current_price, position, -1]
                        sell_orders_out500.append(order)
                        date_and_time.append([today, sdk.getNowTime()])
                        del stock_position[stock]
        sdk.makeOrders(sell_orders_out500)
        buy_and_hold += sell_orders_out500
        buy_and_hold_time += date_and_time

        # 考虑当日中证500可交易的股票
        buy_orders = []
        sell_orders = []
        for stock in zz500_tradable:
            current_price = quotes[stock].current
            if stock in vibrate_stocks:  # 震荡市
                sell_point = sell_line[stock]
                buy_point = buy_line[stock]
                close_point = ma[stock]
                if stock_position[stock]['position'] == 1:  # 只有底仓
                    if current_price > buy_point:  # 做多
                        volume = 100 * np.floor(available_cash * 0.5 / (100 * current_price))
                        if volume > 0:
                            order = [stock, current_price, volume, 1]
                            buy_orders.append(order)
                            traded_stock.append(stock)
                            stock_position[stock]['position'] = 2
                            stock_position[stock]['open_price'] = current_price
                            stock_position[stock]['open_mkt'] = 'V'
                            stock_position[stock]['open_vol'] = volume
                    elif current_price < sell_point:  # 做空
                        volume = position_dict[stock]
                        order = [stock, current_price, volume, -1]
                        sell_orders.append(order)
                        traded_stock.append(stock)
                        stock_position[stock]['position'] = 0
                        stock_position[stock]['open_price'] = current_price
                        stock_position[stock]['open_mkt'] = 'V'
                        stock_position[stock]['open_vol'] = volume
                elif stock_position[stock]['position'] == 0:  # 当前空头
                    if (stock_position[stock]['open_mkt'] == 'V') & (current_price > buy_point):  # 震荡市下开的仓，平空做多
                        close_volume = stock_position[stock]['open_vol']
                        open_volume = 100 * np.floor(available_cash * 0.5 / (100 * current_price))
                        volume = close_volume + open_volume
                        order = [stock, current_price, volume, 1]
                        buy_orders.append(order)
                        traded_stock.append(stock)
                        stock_position[stock]['position'] = 2
                        stock_position[stock]['open_price'] = current_price
                        stock_position[stock]['open_mkt'] = 'V'
                        stock_position[stock]['open_vol'] = open_volume
                    elif (stock_position[stock]['open_mkt'] == 'T') & (current_price > close_point):  # 趋势市下开的仓，平空
                        volume = stock_position[stock]['open_vol']
                        order = [stock, current_price, volume, 1]
                        buy_orders.append(order)
                        traded_stock.append(stock)
                        stock_position[stock]['position'] = 1
                        stock_position[stock]['open_price'] = 0
                        stock_position[stock]['open_mkt'] = 'No'
                        stock_position[stock]['open_vol'] = 0
                elif stock_position[stock]['position'] == 2:  # 当前多头
                    if (stock_position[stock]['open_mkt'] == 'V') & (current_price < sell_point):  # 震荡市下开的仓，平多做空
                        volume = position_dict[stock]
                        order = [stock, current_price, volume, -1]
                        sell_orders.append(order)
                        traded_stock.append(stock)
                        stock_position[stock]['position'] = 0
                        stock_position[stock]['open_price'] = current_price
                        stock_position[stock]['open_mkt'] = 'V'
                        stock_position[stock]['open_vol'] = volume - stock_position[stock]['open_vol']
                    elif (stock_position[stock]['open_mkt'] == 'T') & (current_price < close_point): # 趋势市下开的仓，平多
                        volume = stock_position[stock]['open_vol']
                        order = [stock, current_price, volume, -1]
                        sell_orders.append(order)
                        traded_stock.append(stock)
                        stock_position[stock]['position'] = 1
                        stock_position[stock]['open_price'] = 0
                        stock_position[stock]['open_mkt'] = 'No'
                        stock_position[stock]['open_vol'] = 0
            elif stock in trend_stocks:  # 趋势市
                buy_point = up_line[stock]
                sell_point = dn_line[stock]
                close_point = ma[stock]
                if stock_position[stock]['position'] == 1:  # 只有底仓
                    if current_price > buy_point:  # 做多
                        volume = 100 * np.floor(available_cash * 0.5 / (100 * current_price))
                        if volume > 0:
                            order = [stock, current_price, volume, 1]
                            buy_orders.append(order)
                            traded_stock.append(stock)
                            stock_position[stock]['position'] = 2
                            stock_position[stock]['open_price'] = current_price
                            stock_position[stock]['open_mkt'] = 'T'
                            stock_position[stock]['open_vol'] = volume
                    elif current_price < sell_point:  # 做空
                        volume = position_dict[stock]
                        order = [stock, current_price, volume, -1]
                        sell_orders.append(order)
                        traded_stock.append(stock)
                        stock_position[stock]['position'] = 0
                        stock_position[stock]['open_price'] = current_price
                        stock_position[stock]['open_mkt'] = 'T'
                        stock_position[stock]['open_vol'] = volume
                elif stock_position[stock]['position'] == 0:  # 目前空头
                    if (stock_position[stock]['open_mkt'] == 'V') & (abs(current_price - stock_position[stock]['open_price']) > 3 * atr[stock]):  # 震荡市开的仓，平空
                        volume = stock_position[stock]['oen_vol']
                        order = [stock, current_price, volume, 1]
                        buy_orders.append(order)
                        traded_stock.append(stock)
                        stock_position[stock]['position'] = 1
                        stock_position[stock]['open_price'] = 0
                        stock_position[stock]['open_mkt'] = 'No'
                        stock_position[stock]['open_vol'] = 0
                    elif (stock_position[stock]['open_mkt'] == 'T') & (current_price > close_point):  # 趋势市开的仓，平空做多
                        close_volume = stock_position[stock]['open_vol']
                        open_volume = 100 * np.floor(available_cash * 0.5 / (100 * current_price))
                        volume = close_volume + open_volume
                        order = [stock, current_price, volume, 1]
                        buy_orders.append(order)
                        traded_stock.append(stock)
                        stock_position[stock]['position'] = 2
                        stock_position[stock]['open_price'] = current_price
                        stock_position[stock]['open_mkt'] = 'T'
                        stock_position[stock]['open_vol'] = open_volume
                elif stock_position[stock]['position'] == 2:  # 目前多头
                    if (stock_position[stock]['open_mkt'] == 'V') & (abs(current_price - stock_position[stock]['open_price']) > 3 * atr[stock]):  # 震荡市开的仓，平多
                        volume = stock_position[stock]['oen_vol']
                        order = [stock, current_price, volume, -1]
                        sell_orders.append(order)
                        traded_stock.append(stock)
                        stock_position[stock]['position'] = 1
                        stock_position[stock]['open_price'] = 0
                        stock_position[stock]['open_mkt'] = 'No'
                        stock_position[stock]['open_vol'] = 0
                    elif (stock_position[stock]['open_mkt'] == 'T') & (current_price < close_point):  # 趋势市开的仓，平多做空
                        volume = position_dict[stock]
                        order = [stock, current_price, volume, -1]
                        sell_orders.append(order)
                        traded_stock.append(stock)
                        stock_position[stock]['position'] = 0
                        stock_position[stock]['open_price'] = current_price
                        stock_position[stock]['open_mkt'] = 'T'
                        stock_position[stock]['open_vol'] = volume - stock_position[stock]['open_vol']
        sdk.makeOrders(sell_orders)
        sdk.makeOrders(buy_orders)
        # 记录下单数据
        if buy_orders or sell_orders or sell_orders_out500:
            sdk.sdklog(sdk.getNowTime(), '=================时间')
            if buy_orders:
                sdk.sdklog('Buy orders')
                sdk.sdklog(np.array(buy_orders))
            if sell_orders:
                sdk.sdklog('Sell orders')
                sdk.sdklog(np.array(sell_orders))
            if sell_orders_out500:
                sdk.sdklog('Sell removed stocks')
                sdk.sdklog(np.array(sell_orders_out500))
        sdk.setGlobal('traded_stock', traded_stock)
        sdk.setGlobal('stock_position', stock_position)
        sdk.setGlobal('buy_and_hold', buy_and_hold)
        sdk.setGlobal('buy_and_hold_time', buy_and_hold_time)

    if (sdk.getNowDate() == '20161230') & (sdk.getNowTime() == '150000'):
        buy_and_hold = sdk.getGlobal('buy_and_hold')
        buy_and_hold_time = sdk.getGlobal('buy_and_hold_time')
        temp = pd.DataFrame(buy_and_hold_time)
        temp = pd.concat([temp, pd.Series(buy_and_hold)], axis=1)
        pd.DataFrame(temp).to_csv('buy_and_hold.csv')



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