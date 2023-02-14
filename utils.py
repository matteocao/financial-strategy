# type hinting
from numbers import Number
from typing import Tuple, Callable, List

# data
from get_all_tickers import get_tickers as gt
import numpy as np
import yfinance as yf
import pandas as pd
from tqdm import tqdm
import os.path
import random

# linear regression
from sklearn.linear_model import LinearRegression

# plot
import plotly.express as px
import plotly.figure_factory as ff

# optimisation
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint


def show_dividends_online(stock_name: str = "MSFT", period: str = 'max') -> None:
    stock_history = yf.Ticker(stock_name).history(period=period)
    fig = px.scatter(stock_history,x=stock_history.index, y="Dividends")
    fig.show()

    
def get_dividends_online(stock_name: str = "MSFT", period: str = 'max') -> np.ndarray:
    stock_history = yf.Ticker(stock_name).history(period=period)
    return stock_history["Dividends"].values


def show_prices_online(stock_name: str = "MSFT", period: str = 'max') -> None:
    stock_history = yf.Ticker(stock_name).history(period=period)
    fig = px.scatter(stock_history,x=stock_history.index, y="Close")
    fig.show()

    
def get_prices_online(stock_name: str = "MSFT", period: str = 'max') -> np.ndarray:
    stock_history = yf.Ticker(stock_name).history(period=period)
    return stock_history["Close"].values


def get_tickers(file_name: str) -> str:
    """get ticker data from file"""
    df = pd.read_csv(file_name)
    names = df["Symbol"].values[:400].tolist() 
    try:
        names.remove('GOOGL')
    except ValueError:
        pass
    return names


def get_macro_data(names: str) -> pd.core.frame.DataFrame:
    """get more detailed data"""
    if os.path.isfile("df_macro.csv"):
        data_df = pd.read_csv("df_macro.csv")
    else:
        data_for_selection = []
        for name in tqdm(names):
            try:
                stock = yf.Ticker(name)
                data_for_selection.append([name, 
                                           stock.info["returnOnEquity"], 
                                           stock.info["sector"], 
                                           stock.info["ebitda"], 
                                           stock.info["enterpriseValue"]])
            except KeyError:
                pass
        data_df = pd.DataFrame(data_for_selection, columns=["Name", "ROE","Sector", "EBITDA","Value"])
        data_df.to_csv("df_macro.csv")
    return data_df


def get_macro_etfs(names: str) -> pd.core.frame.DataFrame:
    """get more detailed data"""
    if os.path.isfile("etfs_macro.csv"):
        data_df = pd.read_csv("etfs_macro.csv")
    else:
        data_for_selection = []
        for name in tqdm(names):
            try:
                stock = yf.Ticker(name)
                data_for_selection.append([name, 
                                           stock.info["sectorWeightings"],
                                           stock.info["totalAssets"]])
            except KeyError:
                pass
        data_df = pd.DataFrame(data_for_selection, columns=["Name", "Sectors","Assets"])
        data_df.to_csv("etfs_macro.csv")
    return data_df


def add_sector_to_etfs(df_etfs):
    """select the most relevant sector and add a column
    to the dataframe"""
    max_sectors = []
    for item2 in df_etfs["Sectors"]:
        item = eval(item2)
        index = np.argmax([v[list(v.keys())[0]] for v in item])
        max_sectors.append(list(item[index].keys())[0])
        
    sectors = [list(k.keys())[0] for k in item]
    df_new = pd.DataFrame()
    df_new['Name'] = df_etfs["Name"].copy()
    df_new['Sectors'] = df_etfs["Sectors"].copy()
    df_new['Assets'] = df_etfs["Assets"].copy()
    df_new["Sector"] = max_sectors
    df_new = df_new.copy()
    return df_new, sectors


def select_data_per_sector(data_df: pd.core.frame.DataFrame, num_per_sector: int = 9) -> Tuple[pd.core.frame.DataFrame, pd.core.frame.DataFrame]:
    sectors = data_df["Sector"].unique()
    selected_data = []
    for sector in sectors:
        try:
            selected_data += data_df[data_df["Sector"] == sector].sort_values('EBITDA',ascending = False).head(num_per_sector)[["Name", "ROE","Sector", "EBITDA","Value"]].values.tolist()
            selected_data_df = pd.DataFrame(selected_data, columns=["Name", "ROE", "Sector", "EBITDA", "Value"])
        except KeyError:
            selected_data += data_df[data_df["Sector"] == sector].sort_values('Assets',ascending = False).head(num_per_sector)[["Name", "Sector", "Assets"]].values.tolist()
            selected_data_df = pd.DataFrame(selected_data, columns=["Name", "Sector", "Value"])
    
    
    return selected_data_df, sectors


def load_or_store_data(names: List[str]) -> Tuple[pd.core.frame.DataFrame,pd.core.frame.DataFrame]:
    # dataframes
    df_p = pd.DataFrame()
    df_d = pd.DataFrame()

    if os.path.isfile("df_p.csv") and os.path.isfile("df_d.csv"):
        df_p = pd.read_csv("df_p.csv")
        df_d = pd.read_csv("df_d.csv")
    else:
        for name in ["MSFT"]:
            p_output = get_prices_online(name, 'max')
            d_output = get_dividends_online(name, 'max')
        # reference length
        p_len = len(p_output)
        d_len = len(d_output)
        assert p_len == d_len

        df_p = pd.DataFrame(p_output.tolist(), columns=["MSFT"])
        df_d = pd.DataFrame(d_output.tolist(), columns=["MSFT"])

        for name in tqdm(names):
            try:
                p_output = get_prices_online(name, 'max')
                d_output = get_dividends_online(name, 'max')
                price_list_padded = p_output.tolist() if len(p_output) == p_len else (p_len - len(p_output))*[0.0001] + p_output.tolist()[-p_len:]
                div_list_padded = d_output.tolist() if len(d_output) == d_len else (d_len - len(d_output))*[0.0] + d_output.tolist()[-d_len:]
                df_p[name] = price_list_padded
                df_d[name] = div_list_padded
            except KeyError:
                pass
        df_p = df_p.copy()
        df_d = df_d.copy()
        df_p.to_csv("df_p.csv")
        df_d.to_csv("df_d.csv")

    return df_p, df_d


def pick_a_split(df_p: pd.core.frame.DataFrame) -> Tuple[Tuple[int,int], Tuple[int,int]]:
    """returns the train and validation tuples"""
    maximum = len(df_p)
    kfold = 6
    unit = maximum//kfold
    train_unit = random.randint(1,kfold-1)
    print("The chosen splitting for training: ", train_unit, "/", kfold)
    return ((train_unit-1)*unit, train_unit*unit-100), ((train_unit)*unit, (train_unit + 1)*unit-1)


# new functions
def get_prices_train(data, stock_name: str = "MSFT", split: Tuple[int,int] = (0,1500)) -> np.ndarray:
    return data[stock_name].values[split[0]:split[1]]


def get_dividends_train(data, stock_name: str = "MSFT", split: Tuple[int,int] = (0,1500)) -> np.ndarray:
    return data[stock_name].values[split[0]:split[1]]


def get_prices_val(data, stock_name: str = "MSFT", split: Tuple[int,int] = (1800,3300)) -> np.ndarray:
    return data[stock_name].values[split[0]:split[1]]


def get_dividends_val(data, stock_name: str = "MSFT", split: Tuple[int,int] = (1800,3300)) -> np.ndarray:
    return data[stock_name].values[split[0]:split[1]]


def portfolio_gain_risk_no_constr(w: np.ndarray, cov_mat: np.ndarray, gain: np.ndarray, gain_div: np.ndarray) -> float:
    pow2 = lambda x : pow(x,2) if isinstance(x,Number) else sum([pow(xx,2) for xx in x])
    # minimise risks (making sure, for example, taht you take the less risky one among correlated entities
    return w.dot(cov_mat.dot(w))/np.linalg.norm(cov_mat) - w.dot(gain/np.linalg.norm(gain))


# optimisation problem (with addition of dividends)
def portfolio_gain_risk_no_constr_with_div(w: np.ndarray, cov_mat: np.ndarray, gain: np.ndarray, gain_div: np.ndarray) -> float:
    pow2 = lambda x : pow(x,2) if isinstance(x,Number) else sum([pow(xx,2) for xx in x])
    # minimise risks (making sure, for example, taht you take the less risky one among correlated entities
    return w.dot(cov_mat.dot(w))/np.linalg.norm(cov_mat) - 0.5*(w.dot(gain/np.linalg.norm(gain))) - 0.5*(w.dot(gain_div/np.linalg.norm(gain_div))) 


# optimisation problem (with addition of dividends)
def portfolio_gain_risk_no_constr_only_div(w: np.ndarray, cov_mat: np.ndarray, gain: np.ndarray, gain_div: np.ndarray) -> float:
    pow2 = lambda x : pow(x,2) if isinstance(x,Number) else sum([pow(xx,2) for xx in x])
    # minimise risks (making sure, for example, taht you take the less risky one among correlated entities
    return w.dot(cov_mat.dot(w))/np.linalg.norm(cov_mat) - (w.dot(gain_div/np.linalg.norm(gain_div))) 


def optimise(portfolio_fnc_with_par: Callable[[Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]], np.ndarray], 
             cov_mat: np.ndarray, gain: np.ndarray, gain_div: np.ndarray) -> np.ndarray:
    dim: int = len(gain)
    linear_constraint_1 = LinearConstraint(np.ones((dim,)), [1], [1])  # sum of weight = 1
    linear_constraint_2 = LinearConstraint(np.eye(dim), np.zeros(dim), np.ones(dim))  # positive weights
    w0 = np.ones((dim,))/np.linalg.norm(np.ones((dim,)))
    portfolio_fnc = lambda x : portfolio_fnc_with_par(x, cov_mat, gain, gain_div)
    res = minimize(portfolio_fnc, w0, method='trust-constr',
                   constraints=[linear_constraint_1, linear_constraint_2],
                   options={'verbose': 1})
    optimum_w = res.x
    return optimum_w


def compute_cov_mat(data, names: Tuple[str], splits) -> np.ndarray:
    """data of the prices"""
    try:
        ref_length = len(get_prices_train(data, "ENI.MI", splits[0]))
    except KeyError:
        ref_length = len(get_prices_train(data, "MSFT", splits[0]))
    all_prices = []
    for stock in names:
        try:
            ggg = get_prices_train(data, stock, splits[0])
            check = len(ggg)
            if check == ref_length:
                all_prices.append(ggg)
            else:
                all_prices.append(np.zeros((ref_length-len(ggg),)).tolist() + ggg.tolist())
        except KeyError:
            pass
        
    return np.cov(all_prices)


def compute_gains(data_p, data_d, names: Tuple[str], splits) -> np.ndarray:
    all_dividends = []
    all_prices = []
    for stock in names:
        try:
            y = get_prices_train(data_p, stock, splits[0])
            X = np.array(range(len(y))).reshape(-1,1)
            reg = LinearRegression().fit(X, y)
            all_dividends.append(sum(get_dividends_train(data_d, stock, splits[0])/y))
            all_prices.append(max(0.0001,reg.coef_[0]))
        except KeyError:
            pass
    return np.array(all_dividends), np.array(all_prices)


def optimise_per_sector(func, cov_mat_all, gain_all, gain_div_all, sectors, selected_data_df, names):
    best_picks = []
    for sector in sectors:
        indices = list(selected_data_df[selected_data_df["Sector"] == sector].index)
        gain = [gain_all[j] for j in indices]
        gain_div = [gain_div_all[j] for j in indices]
        cov_mat = np.array([np.array([cov_mat_all[j] for j in indices]).T[jj] for jj in indices])
        # print(sector, gain, gain_div, cov_mat)
        optimum_w = optimise(func, cov_mat, gain, gain_div)
        names_sector = [names[jjj] for jjj in indices]
        list_for_df = zip(names_sector, optimum_w)
        optimum_w_df = pd.DataFrame(list_for_df, columns=["name", "weight"])
        # print(optimum_w_df[optimum_w_df["weight"] == optimum_w_df["weight"].max()]["name"].values)
        best_picks.append(optimum_w_df[optimum_w_df["weight"] == optimum_w_df["weight"].max()]["name"].values[0])
        fig = px.bar(optimum_w_df, x="name", y="weight", title=sector)
        fig.show()
    return best_picks


def accumulate(data, n_months: int = 36, p_month: float = 300, stock: str = "RACE.MI", splits = ((0,1),(1500,2500))) -> Tuple[List[float],List[float],float] :
    """this is a function"""
    y = get_prices_val(data, stock, splits[1])
    buy_prices = []
    numbers = []
    for i in range(n_months):
        buy_prices.append(y[i*22])
        numbers.append(p_month/buy_prices[-1] if buy_prices[-1]>0.0001 else 0)
    number = sum(numbers)
    final_gain = number * y[-1]
    roi = final_gain/(n_months*p_month)
    return buy_prices, numbers, roi


def dividend_reinvest(data_p, data_d, m0: float = 100, stock: str = "RACE.MI", splits = ((0,1),(1500,2500))) -> float:
    all_dividends = get_dividends_val(data_d, stock, splits[1])
    all_prices = get_prices_val(data_p, stock, splits[1])
    current_number = m0/all_prices[0] if all_prices[0] > 0.0001 else 0
    for j, divi in enumerate(all_dividends):
        if divi > 0.0001:
            divi_gain = divi * current_number
            current_number += divi_gain/all_prices[j]
            # print(divi, current_number)
    roi = current_number * all_prices[-1]/m0
    return roi