import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

import plotly.express as px
from dateutil.relativedelta import relativedelta

import math
import yfinance as yf

class px_plot(object):
  def __init__(self, stock, figsize=(20,8)):
    self.stock=stock
    self.figsize=figsize
  
  def px_fig_update(self, fig, x_title, y_title, with_zero_line=False):

    width, height = self.figsize
    width = width * 100
    height = height * 100

    fig.update_layout(
      font={"size": 12},
      title_font={"size": 20},
      title_x=0.5,
      legend_title_font={"size": 16},
      width=width,
      height=height,
      xaxis_title= x_title,
      yaxis_title= y_title,
      plot_bgcolor="white",
      legend_title_text="",
      legend=dict(
        traceorder="normal",
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
      )
    )

    fig.update_xaxes(
      title_font={"size": 16},
      showline=True,
      linewidth=2,
      linecolor="black",
      mirror=True,
      gridcolor=(None),
    )

    fig.update_yaxes(
      title_font={"size": 16},
      showline=True,
      linewidth=2,
      linecolor="black",
      mirror=True,
      gridcolor=(None),
    )

    if with_zero_line:
      fig.add_hline(y=0, line_dash="dash", row=1, col=1, line_color="#000000", line_width=2)

def consecutive_analysis(pct_change, sd_th, n_d, n_y=5, trade_d=250, return_output=False):

  lo = lambda s, l: f"Last occured on {str(l[s].index[-1].date())}" if len(l[s]) > 0 else ""

  l = pct_change[-n_y*trade_d:]
  s_con_above = (l >= sd_th).rolling(n_d).sum() == n_d
  s_con_below = (l <= -sd_th).rolling(n_d).sum() == n_d

  s_cum_above = l.rolling(n_d).sum() >= sd_th
  s_cum_below = l.rolling(n_d).sum() <= -sd_th

  con_above = s_con_above.sum()/len(l)
  con_below = s_con_below.sum()/len(l)

  cum_above = len(l[s_cum_above])/len(l)
  cum_below = len(l[s_cum_below])/len(l)

  # con_above = ((l >= sd_th).rolling(n_d).sum() == n_d).sum()/len(l)
  # con_below = ((l <= -sd_th).rolling(n_d).sum() == n_d).sum()/len(l)

  # cum_above = len(l[l.rolling(n_d).sum() >= sd_th])/len(l)
  # cum_below = len(l[l.rolling(n_d).sum() <= -sd_th])/len(l)

  # n Consecutive days above
  print(f"Consecutive change of above {sd_th:.2%} for {n_d} days in the last {n_y} years:\nProbability {con_above:.1%} {lo(s_con_above, l)}")
  # n Consecutive days below
  print(f"Consecutive change of below {sd_th:.2%} for {n_d} days in the last {n_y} years:\nProbability {con_below:.1%} {lo(s_con_below, l)}")

  # n Cumulative days above
  print(f"Cumulative change of above {sd_th:.2%} in {n_d} days in the last {n_y} years:\nProbability {cum_above:.1%} {lo(s_cum_above, l)}")
  # n Cumulative days below
  print(f"Cumulative change of below {sd_th:.2%} in {n_d} days in the last {n_y} years:\nProbability {cum_below:.1%} {lo(s_cum_below, l)}")

  if return_output:
    df = pd.DataFrame(
        {"change_type":["consecutive", "consecutive", "cumulative", "cumulative"],
         "change":["above", "below", "above", "below"],
         "up_dn":[f"{sd_th:.2%}", f"{sd_th:.2%}", f"{sd_th:.2%}", f"{sd_th:.2%}"],
         "n_days":[n_d, n_d, n_d, n_d],
         "n_years":[n_y, n_y, n_y, n_y],
         "prob":[f"{con_above:.1%}", f"{con_below:.1%}", f"{cum_above:.1%}", f"{cum_below:.1%}"],
         "last_occured":[f"{lo(s_con_above, l)}", f"{lo(s_con_below, l)}", f"{lo(s_cum_above, l)}", f"{lo(s_cum_below, l)}"]}
    )

    return df
  
 # Volatility and condition

def sd_and_cond(data, sd_th, sd_fix):

  sd_dict = {
      "th":{1:dict(), 5:dict(), 10:dict()},
      "act":{1:dict(), 5:dict(), 10:dict()},
      "fix":{1:dict(), 5:dict(), 10:dict()},
      }

  cond = {
      "th":{1:dict(), 5:dict(), 10:dict()},
      "act":{1:dict(), 5:dict(), 10:dict()},
      "fix":{1:dict(), 5:dict(), 10:dict()},
      }

  for n in [1, 5, 10]:

    for met in ["th", "act", "fix"]:

      if met == "th":
        sd_dict[met][n] = sd_th*(n**0.5)
      elif met == "act":
        sd_dict[met][n] = data[f"PCT Change {n} STD"].iloc[-1]
      else:
        sd_dict[met][n] = sd_fix

      cond[met][n]["above"] = data[f"PCT Change {n}"] > sd_dict[met][n]
      cond[met][n]["below"] = data[f"PCT Change {n}"] < -sd_dict[met][n]
      cond[met][n]["exceeding"] = abs(data[f"PCT Change {n}"]) > sd_dict[met][n]

  return sd_dict, cond

def profit_estimate(sd_dict, cond, p_and_l, cont_size, trade_d=250):

  f = lambda prob, pl: (1 - prob) * pl[0] + prob * pl[1]
  prob_loss = {
    "th":{1:dict(), 5:dict(), 10:dict()},
    "act":{1:dict(), 5:dict(), 10:dict()},
    "fix":{1:dict(), 5:dict(), 10:dict()},
    }
  profit_expect = {
    "th":{1:dict(), 5:dict(), 10:dict()},
    "act":{1:dict(), 5:dict(), 10:dict()},
    "fix":{1:dict(), 5:dict(), 10:dict()},
    }

  for n in [1, 5, 10]:

    for met in ["th", "act", "fix"]:

      if met == "th":
        print(f'Default volitality threshold for {n} days: {sd_dict[met][n]:.3f}')

      elif met == "act":
        print(f'1-Year volitality for {n} days: {sd_dict[met][n]:.3f}')

      else:
        print(f'Fixed volitality for {n} days: {sd_dict[met][n]:.3f}')

      for k in ["above", "below", "exceeding"]:
        p_loss = cond[met][n][k][-trade_d:].sum()/trade_d
        print(f'Probability of {k} the limit: {p_loss:.3f}')

        profit = f(p_loss, p_and_l[n][k]) * cont_size
        print(f'Profit per contract: {profit:.2f}')

        profit_expect[met][n][k] = profit
        prob_loss[met][n][k] = p_loss

      print()

  return prob_loss, profit_expect
  
def accepted_min_max(p, s, r):
  # Assume 6.5 hours of trading
  t_h = 6.5
  assert r <= t_h
  print(f"Daily volatility: {s * 100}%")
  print(f"Price: {p}")
  print(f"Remaining hours: {r}")

  # Volatility for the remaining hours
  s_r = s * ((r/t_h) ** 0.5)

  print(f"Volatility accepted: {round(s_r * 100, 2)}%")
  max_p = math.ceil(p * math.exp(s_r) * 10)/10
  min_p = math.floor(p * math.exp(-s_r) * 10)/10
  print(f"Max accepted: {max_p}")
  print(f"Min accepted: {min_p}")

def projected_min_max(p, s, d):

  print(f"Daily volatility: {s * 100}%")
  print(f"Price: {p}")
  print(f"Projected days: {d}")

  # Projected volatility
  s_d = s * (d ** 0.5)

  print(f"Volatility accepted: {round(s_d * 100, 2)}%")
  max_p = math.ceil(p * math.exp(s_d) * 10)/10
  min_p = math.floor(p * math.exp(-s_d) * 10)/10
  print(f"Max accepted: {max_p}")
  print(f"Min accepted: {min_p}")
  
  
  
 