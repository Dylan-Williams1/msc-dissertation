# Specifications for the OHLCV-implementable subset of Kakushadze (2016).
# Each entry: kakushadze number, exposed parameters (published defaults),
# a step decomposition used as the Section 2 reference (against which the
# Section 5 code is audited for Code-Documentation Fidelity), and the body.

SPECS = [

dict(n=1, params=[("d_std", 20), ("d_arg", 5)],
     inputs=["close", "returns"],
     steps=[
        "base_t = stddev(returns, d_std) where returns < 0, else close.",
        "Apply a sign-preserving square to base_t.",
        "Take the within-window position of the maximum over the trailing d_arg days.",
        "Cross-sectional rank, then subtract 0.5 to centre the signal.",
     ],
     body="""    base = where(lt(returns, 0.0), stddev(returns, d_std), close)
    alpha = rank(ts_argmax(signedpower(base, 2.0), d_arg)) - 0.5"""),

dict(n=2, params=[("d_delta", 2), ("d_corr", 6)],
     inputs=["open", "close", "volume"],
     steps=[
        "d_delta-day change in log volume; cross-sectional rank.",
        "Intraday return (close - open) / open; cross-sectional rank.",
        "d_corr-day rolling time-series correlation of the two rank series.",
        "Negate.",
     ],
     body="""    alpha = -1 * correlation(rank(delta(log(volume), d_delta)),
                             rank((close - open_) / open_), d_corr)"""),

dict(n=3, params=[("d_corr", 10)],
     inputs=["open", "volume"],
     steps=[
        "Cross-sectional ranks of open and of volume.",
        "d_corr-day rolling correlation of the two rank series; negate.",
     ],
     body="""    alpha = -1 * correlation(rank(open_), rank(volume), d_corr)"""),

dict(n=4, params=[("d_rank", 9)],
     inputs=["low"],
     steps=[
        "Cross-sectional rank of the low.",
        "Time-series rank of that quantity over d_rank days; negate.",
     ],
     body="""    alpha = -1 * ts_rank(rank(low), d_rank)"""),

dict(n=6, params=[("d_corr", 10)],
     inputs=["open", "volume"],
     steps=["d_corr-day rolling correlation of raw open and raw volume; negate."],
     body="""    alpha = -1 * correlation(open_, volume, d_corr)"""),

dict(n=7, params=[("d_adv", 20), ("d_delta", 7), ("d_rank", 60)],
     inputs=["close", "volume"],
     steps=[
        "adv20 = d_adv-day mean dollar volume.",
        "If today's volume exceeds adv20: negative time-series rank of |delta(close, d_delta)| "
        "over d_rank days, multiplied by the sign of that same change.",
        "Otherwise the signal is the constant -1.",
     ],
     body="""    adv20 = adv(d_adv)
    dc = delta(close, d_delta)
    alpha = where(lt(adv20, volume),
                  (-1 * ts_rank(abs(dc), d_rank)) * sign(dc),
                  -1.0)"""),

dict(n=8, params=[("d_sum", 5), ("d_delay", 10)],
     inputs=["open", "returns"],
     steps=[
        "term = (d_sum-day sum of open) * (d_sum-day sum of returns).",
        "Change in term over d_delay days; cross-sectional rank; negate.",
     ],
     body="""    term = ts_sum(open_, d_sum) * ts_sum(returns, d_sum)
    alpha = -1 * rank(term - delay(term, d_delay))"""),

dict(n=9, params=[("d_window", 5)],
     inputs=["close"],
     steps=[
        "dc = one-day close change.",
        "If the d_window-day minimum of dc is positive (uninterrupted up-run), pass dc through.",
        "Else if the d_window-day maximum of dc is negative (uninterrupted down-run), pass dc through.",
        "Otherwise negate dc.",
     ],
     body="""    dc = delta(close, 1)
    alpha = where(gt(ts_min(dc, d_window), 0.0), dc,
                  where(lt(ts_max(dc, d_window), 0.0), dc, -1 * dc))"""),

dict(n=10, params=[("d_window", 4)],
     inputs=["close"],
     steps=[
        "Identical construction to Alpha#9 with a d_window-day run test.",
        "Cross-sectional rank of the result.",
     ],
     body="""    dc = delta(close, 1)
    inner = where(gt(ts_min(dc, d_window), 0.0), dc,
                  where(lt(ts_max(dc, d_window), 0.0), dc, -1 * dc))
    alpha = rank(inner)"""),

dict(n=12, params=[],
     inputs=["close", "volume"],
     steps=[
        "Sign of the one-day volume change.",
        "Multiplied by the negated one-day close change.",
     ],
     body="""    alpha = sign(delta(volume, 1)) * (-1 * delta(close, 1))"""),

dict(n=13, params=[("d_cov", 5)],
     inputs=["close", "volume"],
     steps=[
        "d_cov-day rolling covariance of the cross-sectional ranks of close and volume.",
        "Cross-sectional rank of that covariance; negate.",
     ],
     body="""    alpha = -1 * rank(covariance(rank(close), rank(volume), d_cov))"""),

dict(n=14, params=[("d_delta", 3), ("d_corr", 10)],
     inputs=["open", "volume", "returns"],
     steps=[
        "Negated cross-sectional rank of the d_delta-day change in returns.",
        "Multiplied by the d_corr-day rolling correlation of open and volume.",
     ],
     body="""    alpha = (-1 * rank(delta(returns, d_delta))) * correlation(open_, volume, d_corr)"""),

dict(n=15, params=[("d_corr", 3), ("d_sum", 3)],
     inputs=["high", "volume"],
     steps=[
        "d_corr-day rolling correlation of the ranks of high and volume.",
        "Cross-sectional rank, summed over d_sum days; negate.",
     ],
     body="""    alpha = -1 * ts_sum(rank(correlation(rank(high), rank(volume), d_corr)), d_sum)"""),

dict(n=16, params=[("d_cov", 5)],
     inputs=["high", "volume"],
     steps=[
        "d_cov-day rolling covariance of the ranks of high and volume.",
        "Cross-sectional rank; negate.",
     ],
     body="""    alpha = -1 * rank(covariance(rank(high), rank(volume), d_cov))"""),

dict(n=17, params=[("d_tsr_close", 10), ("d_adv", 20), ("d_tsr_vol", 5)],
     inputs=["close", "volume"],
     steps=[
        "Negated rank of the d_tsr_close-day time-series rank of close.",
        "Times the rank of the second difference of close.",
        "Times the rank of the d_tsr_vol-day time-series rank of volume / adv20.",
     ],
     body="""    adv20 = adv(d_adv)
    alpha = ((-1 * rank(ts_rank(close, d_tsr_close)))
             * rank(delta(delta(close, 1), 1))
             * rank(ts_rank(volume / adv20, d_tsr_vol)))"""),

dict(n=18, params=[("d_std", 5), ("d_corr", 10)],
     inputs=["open", "close"],
     steps=[
        "d_std-day standard deviation of |close - open|, plus (close - open).",
        "Plus the d_corr-day rolling correlation of close and open.",
        "Cross-sectional rank; negate.",
     ],
     body="""    alpha = -1 * rank((stddev(abs(close - open_), d_std) + (close - open_))
                      + correlation(close, open_, d_corr))"""),

dict(n=19, params=[("d_delay", 7), ("d_sum", 250)],
     inputs=["close", "returns"],
     steps=[
        "Negated sign of (close - close lagged d_delay) + (d_delay-day close change).",
        "Times 1 + rank(1 + d_sum-day sum of returns).",
     ],
     body="""    alpha = ((-1 * sign((close - delay(close, d_delay)) + delta(close, d_delay)))
             * (1 + rank(1 + ts_sum(returns, d_sum))))"""),

dict(n=20, params=[("d_delay", 1)],
     inputs=["open", "high", "low", "close"],
     steps=[
        "Negated rank of (open - prior high), times rank of (open - prior close), "
        "times rank of (open - prior low).",
     ],
     body="""    alpha = ((-1 * rank(open_ - delay(high, d_delay)))
             * rank(open_ - delay(close, d_delay))
             * rank(open_ - delay(low, d_delay)))"""),

dict(n=21, params=[("d_long", 8), ("d_short", 2), ("d_adv", 20)],
     inputs=["close", "volume"],
     steps=[
        "ma_long and sd_long over d_long days; ma_short over d_short days.",
        "If ma_long + sd_long < ma_short: -1 (short the overextended name).",
        "Else if ma_short < ma_long - sd_long: +1.",
        "Else +1 when volume / adv20 >= 1, otherwise -1.",
        "NOTE: the published condition is (1 < v/adv) || (v/adv == 1); implemented as >=, "
        "which is algebraically identical.",
     ],
     body="""    ma_l = ts_sum(close, d_long) / d_long
    sd_l = stddev(close, d_long)
    ma_s = ts_sum(close, d_short) / d_short
    vr = volume / adv(d_adv)
    alpha = where(lt(ma_l + sd_l, ma_s), -1.0,
                  where(lt(ma_s, ma_l - sd_l), 1.0,
                        where(ge(vr, 1.0), 1.0, -1.0)))"""),

dict(n=22, params=[("d_corr", 5), ("d_delta", 5), ("d_std", 20)],
     inputs=["high", "close", "volume"],
     steps=[
        "d_delta-day change in the d_corr-day correlation of high and volume.",
        "Times the rank of the d_std-day standard deviation of close; negate.",
     ],
     body="""    alpha = -1 * (delta(correlation(high, volume, d_corr), d_delta)
                  * rank(stddev(close, d_std)))"""),

dict(n=23, params=[("d_mean", 20), ("d_delta", 2)],
     inputs=["high"],
     steps=[
        "If today's high exceeds its d_mean-day average: negated d_delta-day change in high.",
        "Otherwise zero.",
     ],
     body="""    alpha = where(lt(ts_sum(high, d_mean) / d_mean, high),
                  -1 * delta(high, d_delta), 0.0)"""),

dict(n=24, params=[("d_long", 100), ("d_delta", 3), ("k", 0.05)],
     inputs=["close"],
     steps=[
        "g = d_long-day change in the d_long-day mean close, divided by close lagged d_long.",
        "If g <= k: negated distance of close above its d_long-day minimum.",
        "Otherwise the negated d_delta-day close change.",
        "NOTE: published condition (g < k) || (g == k); implemented as <=.",
     ],
     body="""    g = delta(ts_sum(close, d_long) / d_long, d_long) / delay(close, d_long)
    alpha = where(le(g, k),
                  -1 * (close - ts_min(close, d_long)),
                  -1 * delta(close, d_delta))"""),

dict(n=26, params=[("d_tsr", 5), ("d_corr", 5), ("d_max", 3)],
     inputs=["high", "volume"],
     steps=[
        "d_tsr-day time-series ranks of volume and high.",
        "d_corr-day rolling correlation of the two.",
        "Rolling maximum over d_max days; negate.",
     ],
     body="""    alpha = -1 * ts_max(correlation(ts_rank(volume, d_tsr),
                                    ts_rank(high, d_tsr), d_corr), d_max)"""),

dict(n=28, params=[("d_adv", 20), ("d_corr", 5)],
     inputs=["high", "low", "close", "volume"],
     steps=[
        "d_corr-day correlation of adv20 with the low.",
        "Plus the midpoint (high + low) / 2, minus close.",
        "Rescaled cross-sectionally to unit gross exposure.",
     ],
     body="""    alpha = scale((correlation(adv(d_adv), low, d_corr) + ((high + low) / 2)) - close)"""),

dict(n=29, params=[("d_delta", 5), ("d_min", 2), ("d_sum", 1),
                   ("d_prod", 1), ("d_tsmin", 5), ("d_delay", 6), ("d_tsr", 5)],
     inputs=["close", "returns"],
     steps=[
        "Nested rank chain on the negated rank of the d_delta-day change in (close - 1).",
        "Rolling minimum over d_min days, summed over d_sum days, logged, scaled, re-ranked.",
        "Rolling product over d_prod days, then rolling minimum over d_tsmin days.",
        "Plus the d_tsr-day time-series rank of negated returns lagged d_delay days.",
     ],
     body="""    inner = ts_min(rank(rank(-1 * rank(delta(close - 1, d_delta)))), d_min)
    left = ts_min(product(rank(rank(scale(log(ts_sum(inner, d_sum))))), d_prod), d_tsmin)
    alpha = left + ts_rank(delay(-1 * returns, d_delay), d_tsr)"""),

dict(n=30, params=[("d_short", 5), ("d_long", 20)],
     inputs=["close", "volume"],
     steps=[
        "Sum of the signs of the last three daily close changes.",
        "1 minus its cross-sectional rank, times the d_short-day volume sum.",
        "Divided by the d_long-day volume sum.",
     ],
     body="""    sgn = (sign(close - delay(close, 1))
           + sign(delay(close, 1) - delay(close, 2))
           + sign(delay(close, 2) - delay(close, 3)))
    alpha = ((1.0 - rank(sgn)) * ts_sum(volume, d_short)) / ts_sum(volume, d_long)"""),

dict(n=31, params=[("d_delta_long", 10), ("d_decay", 10), ("d_delta_short", 3),
                   ("d_adv", 20), ("d_corr", 12)],
     inputs=["low", "close", "volume"],
     steps=[
        "Linear-decay average of the doubly-ranked, negated d_delta_long-day close change; triple rank.",
        "Plus the rank of the negated d_delta_short-day close change.",
        "Plus the sign of the scaled d_corr-day correlation of adv20 and low.",
     ],
     body="""    alpha = (rank(rank(rank(decay_linear(-1 * rank(rank(delta(close, d_delta_long))),
                                        d_decay))))
             + rank(-1 * delta(close, d_delta_short))
             + sign(scale(correlation(adv(d_adv), low, d_corr))))"""),

dict(n=33, params=[],
     inputs=["open", "close"],
     steps=["Cross-sectional rank of -(1 - open/close), i.e. the intraday open-to-close drift."],
     body="""    alpha = rank(-1 * ((1 - (open_ / close)) ** 1))"""),

dict(n=34, params=[("d_short", 2), ("d_long", 5)],
     inputs=["close", "returns"],
     steps=[
        "Ratio of short- to long-window return volatility; 1 minus its rank.",
        "Plus 1 minus the rank of the one-day close change; cross-sectional rank.",
     ],
     body="""    alpha = rank((1 - rank(stddev(returns, d_short) / stddev(returns, d_long)))
                 + (1 - rank(delta(close, 1))))"""),

dict(n=35, params=[("d_vol", 32), ("d_price", 16), ("d_ret", 32)],
     inputs=["high", "low", "close", "volume", "returns"],
     steps=[
        "d_vol-day time-series rank of volume.",
        "Times 1 minus the d_price-day time-series rank of (close + high - low).",
        "Times 1 minus the d_ret-day time-series rank of returns.",
     ],
     body="""    alpha = (ts_rank(volume, d_vol)
             * (1 - ts_rank((close + high) - low, d_price))
             * (1 - ts_rank(returns, d_ret)))"""),

dict(n=37, params=[("d_corr", 200), ("d_delay", 1)],
     inputs=["open", "close"],
     steps=[
        "d_corr-day correlation of the lagged open-close gap with close; rank.",
        "Plus the rank of the current open-close gap.",
     ],
     body="""    alpha = (rank(correlation(delay(open_ - close, d_delay), close, d_corr))
             + rank(open_ - close))"""),

dict(n=38, params=[("d_tsr", 10)],
     inputs=["open", "close"],
     steps=[
        "Negated rank of the d_tsr-day time-series rank of close.",
        "Times the rank of close / open.",
     ],
     body="""    alpha = (-1 * rank(ts_rank(close, d_tsr))) * rank(close / open_)"""),

dict(n=39, params=[("d_delta", 7), ("d_decay", 9), ("d_adv", 20), ("d_sum", 250)],
     inputs=["close", "volume", "returns"],
     steps=[
        "d_delta-day close change times 1 minus the rank of the decayed volume/adv20 ratio.",
        "Negated cross-sectional rank of that product.",
        "Times 1 plus the rank of the d_sum-day return sum.",
     ],
     body="""    alpha = ((-1 * rank(delta(close, d_delta)
                        * (1 - rank(decay_linear(volume / adv(d_adv), d_decay)))))
             * (1 + rank(ts_sum(returns, d_sum))))"""),

dict(n=40, params=[("d_std", 10), ("d_corr", 10)],
     inputs=["high", "volume"],
     steps=[
        "Negated rank of the d_std-day standard deviation of high.",
        "Times the d_corr-day correlation of high and volume.",
     ],
     body="""    alpha = (-1 * rank(stddev(high, d_std))) * correlation(high, volume, d_corr)"""),

dict(n=43, params=[("d_adv", 20), ("d_tsr_vol", 20), ("d_delta", 7), ("d_tsr_px", 8)],
     inputs=["close", "volume"],
     steps=[
        "d_tsr_vol-day time-series rank of volume / adv20.",
        "Times the d_tsr_px-day time-series rank of the negated d_delta-day close change.",
     ],
     body="""    alpha = (ts_rank(volume / adv(d_adv), d_tsr_vol)
             * ts_rank(-1 * delta(close, d_delta), d_tsr_px))"""),

dict(n=44, params=[("d_corr", 5)],
     inputs=["high", "volume"],
     steps=["d_corr-day correlation of raw high with the rank of volume; negate."],
     body="""    alpha = -1 * correlation(high, rank(volume), d_corr)"""),

dict(n=45, params=[("d_delay", 5), ("d_mean", 20), ("d_corr", 2),
                   ("d_short", 5), ("d_long", 20)],
     inputs=["close", "volume"],
     steps=[
        "Rank of the d_mean-day average of close lagged d_delay days.",
        "Times the d_corr-day correlation of close and volume.",
        "Times the rank of the d_corr-day correlation of the short and long close sums; negate.",
     ],
     body="""    alpha = -1 * ((rank(ts_sum(delay(close, d_delay), d_mean) / d_mean)
                   * correlation(close, volume, d_corr))
                  * rank(correlation(ts_sum(close, d_short),
                                     ts_sum(close, d_long), d_corr)))"""),

dict(n=46, params=[("d_far", 20), ("d_near", 10), ("k", 0.25)],
     inputs=["close"],
     steps=[
        "curv = ((close[-d_far] - close[-d_near]) / d_near) - ((close[-d_near] - close) / d_near).",
        "If curv > k: -1. Else if curv < 0: +1. Otherwise the negated one-day close change.",
     ],
     body="""    curv = (((delay(close, d_far) - delay(close, d_near)) / d_near)
            - ((delay(close, d_near) - close) / d_near))
    alpha = where(gt(curv, k), -1.0,
                  where(lt(curv, 0.0), 1.0, -1 * (close - delay(close, 1))))"""),

dict(n=49, params=[("d_far", 20), ("d_near", 10), ("k", -0.1)],
     inputs=["close"],
     steps=[
        "Same curvature term as Alpha#46.",
        "If curv < k: +1. Otherwise the negated one-day close change.",
     ],
     body="""    curv = (((delay(close, d_far) - delay(close, d_near)) / d_near)
            - ((delay(close, d_near) - close) / d_near))
    alpha = where(lt(curv, k), 1.0, -1 * (close - delay(close, 1)))"""),

dict(n=51, params=[("d_far", 20), ("d_near", 10), ("k", -0.05)],
     inputs=["close"],
     steps=["Identical to Alpha#49 with a shallower curvature threshold k."],
     body="""    curv = (((delay(close, d_far) - delay(close, d_near)) / d_near)
            - ((delay(close, d_near) - close) / d_near))
    alpha = where(lt(curv, k), 1.0, -1 * (close - delay(close, 1)))"""),

dict(n=52, params=[("d_min", 5), ("d_delay", 5), ("d_long", 240),
                   ("d_short", 20), ("d_tsr", 5)],
     inputs=["low", "volume", "returns"],
     steps=[
        "Change in the d_min-day rolling low between now and d_delay days ago.",
        "Times the rank of the long-minus-short horizon return spread, scaled by (d_long - d_short).",
        "Times the d_tsr-day time-series rank of volume.",
     ],
     body="""    tsl = ts_min(low, d_min)
    alpha = (((-1 * tsl) + delay(tsl, d_delay))
             * rank((ts_sum(returns, d_long) - ts_sum(returns, d_short))
                    / (d_long - d_short))
             * ts_rank(volume, d_tsr))"""),

dict(n=55, params=[("d_window", 12), ("d_corr", 6)],
     inputs=["high", "low", "close", "volume"],
     steps=[
        "Stochastic position of close within its d_window-day high-low range.",
        "d_corr-day correlation of its rank with the rank of volume; negate.",
     ],
     body="""    sto = ((close - ts_min(low, d_window))
           / (ts_max(high, d_window) - ts_min(low, d_window)))
    alpha = -1 * correlation(rank(sto), rank(volume), d_corr)"""),

dict(n=60, params=[("d_arg", 10)],
     inputs=["high", "low", "close", "volume"],
     steps=[
        "Close location value within the daily range, multiplied by volume; ranked and scaled, doubled.",
        "Minus the scaled rank of the d_arg-day argmax of close; negate the difference.",
     ],
     body="""    clv = ((close - low) - (high - close)) / (high - low)
    alpha = 0 - (1 * ((2 * scale(rank(clv * volume)))
                      - scale(rank(ts_argmax(close, d_arg)))))"""),

dict(n=68, params=[("d_adv", 15), ("d_corr", 8.91644), ("d_tsr", 13.9333),
                   ("w", 0.518371), ("d_delta", 1.06157)],
     inputs=["high", "low", "close", "volume"],
     steps=[
        "Time-series rank of the correlation between ranked high and ranked adv15.",
        "Compared against the rank of the change in a close/low blend.",
        "Boolean indicator, negated.",
     ],
     body="""    left = ts_rank(correlation(rank(high), rank(adv(d_adv)), d_corr), d_tsr)
    right = rank(delta((close * w) + (low * (1 - w)), d_delta))
    alpha = lt(left, right) * -1"""),

dict(n=85, params=[("w", 0.876703), ("d_adv", 30), ("d_corr1", 9.61331),
                   ("d_tsr_px", 3.70596), ("d_tsr_vol", 10.1595), ("d_corr2", 7.11408)],
     inputs=["high", "low", "close", "volume"],
     steps=[
        "Rank of the correlation between a high/close blend and adv30.",
        "Raised to the power of the rank of the correlation between the ranked midpoint and ranked volume.",
     ],
     body="""    a1 = rank(correlation((high * w) + (close * (1 - w)), adv(d_adv), d_corr1))
    a2 = rank(correlation(ts_rank((high + low) / 2, d_tsr_px),
                          ts_rank(volume, d_tsr_vol), d_corr2))
    alpha = power(a1, a2)"""),

dict(n=88, params=[("d_decay1", 8.06882), ("d_tsr_close", 8.44728), ("d_adv", 60),
                   ("d_tsr_adv", 20.6966), ("d_corr", 8.01266),
                   ("d_decay2", 6.65053), ("d_tsr_out", 2.61957)],
     inputs=["open", "high", "low", "close", "volume"],
     steps=[
        "Decayed rank spread (open + low) - (high + close); cross-sectional rank.",
        "Decayed correlation of time-series ranked close and adv60; time-series ranked.",
        "Elementwise minimum of the two legs.",
     ],
     body="""    p1 = rank(decay_linear((rank(open_) + rank(low)) - (rank(high) + rank(close)),
                           d_decay1))
    p2 = ts_rank(decay_linear(correlation(ts_rank(close, d_tsr_close),
                                          ts_rank(adv(d_adv), d_tsr_adv),
                                          d_corr), d_decay2), d_tsr_out)
    alpha = emin(p1, p2)"""),

dict(n=92, params=[("d_decay1", 14.7221), ("d_tsr1", 18.8683), ("d_adv", 30),
                   ("d_corr", 7.58555), ("d_decay2", 6.94024), ("d_tsr2", 6.80584)],
     inputs=["open", "high", "low", "close", "volume"],
     steps=[
        "Indicator that midpoint + close falls below low + open; decayed and time-series ranked.",
        "Decayed correlation of ranked low and ranked adv30; time-series ranked.",
        "Elementwise minimum of the two legs.",
     ],
     body="""    ind = lt(((high + low) / 2) + close, low + open_)
    p1 = ts_rank(decay_linear(ind, d_decay1), d_tsr1)
    p2 = ts_rank(decay_linear(correlation(rank(low), rank(adv(d_adv)), d_corr),
                              d_decay2), d_tsr2)
    alpha = emin(p1, p2)"""),

dict(n=95, params=[("d_min", 12.4105), ("d_sum", 19.1351), ("d_adv", 40),
                   ("d_corr", 12.8742), ("d_tsr", 11.7584), ("p", 5)],
     inputs=["open", "high", "low", "close", "volume"],
     steps=[
        "Rank of the open's distance above its rolling minimum.",
        "Time-series rank of the fifth power of the ranked midpoint-vs-adv40 correlation.",
        "Boolean indicator that the first is below the second.",
     ],
     body="""    left = rank(open_ - ts_min(open_, d_min))
    right = ts_rank(power(rank(correlation(ts_sum((high + low) / 2, d_sum),
                                           ts_sum(adv(d_adv), d_sum), d_corr)), p),
                    d_tsr)
    alpha = lt(left, right)"""),

dict(n=99, params=[("d_sum", 19.8975), ("d_adv", 60), ("d_corr1", 8.8136),
                   ("d_corr2", 6.28259)],
     inputs=["high", "low", "volume"],
     steps=[
        "Rank of the correlation between the summed midpoint and summed adv60.",
        "Compared against the rank of the low-volume correlation; indicator negated.",
     ],
     body="""    left = rank(correlation(ts_sum((high + low) / 2, d_sum),
                            ts_sum(adv(d_adv), d_sum), d_corr1))
    right = rank(correlation(low, volume, d_corr2))
    alpha = lt(left, right) * -1"""),

dict(n=101, params=[("eps", 0.001)],
     inputs=["open", "high", "low", "close"],
     steps=["Intraday body (close - open) divided by the daily range plus eps."],
     body="""    alpha = (close - open_) / ((high - low) + eps)"""),
]


# ---------------------------------------------------------------------------
# Exclusions. Selection is on DATA AVAILABILITY and EXECUTION CONVENTION only.
# No alpha is excluded, or retained, on the basis of performance.
# ---------------------------------------------------------------------------
EXCLUSIONS = {}
for _n in [5, 11, 25, 27, 32, 36, 41, 47, 50, 57, 61, 62, 64, 65, 66, 71, 72,
           73, 74, 75, 77, 78, 81, 83, 84, 86, 94, 96, 98]:
    EXCLUSIONS[_n] = ("data_availability",
                      "requires vwap; not present in the OHLCV panel (Section 6)")
EXCLUSIONS[42] = ("data_availability+execution",
                  "requires vwap; also a delay-0 alpha (Kakushadze fn.11)")
EXCLUSIONS[56] = ("data_availability", "requires market cap; outside the OHLCV tier")
for _n in [58, 59, 63, 67, 69, 70, 76, 79, 80, 82, 87, 89, 90, 91, 93, 97, 100]:
    EXCLUSIONS[_n] = ("data_availability",
                      "requires IndClass industry classification for indneutralize()")
EXCLUSIONS[48] = ("data_availability+execution",
                  "requires IndClass; also a delay-0 alpha (Kakushadze fn.11)")
for _n in [53, 54]:
    EXCLUSIONS[_n] = ("execution_convention",
                      "delay-0 alpha (Kakushadze fn.11): traded at the close of the "
                      "computation day, incompatible with the locked signal-at-close-t / "
                      "execute-at-t+1-open convention")
