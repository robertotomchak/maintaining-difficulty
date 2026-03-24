from random import uniform as random
from math import e, log

M = 1.0

def margin(dp, dn):
    return dp - dn + M

def ratio(dp, dn):
    exp_dp = e ** (-dp)
    exp_dn = e ** (-dn)
    exp_sum = exp_dp + exp_dn
    term = exp_dp / exp_sum
    return 2 * term * term

N = 1_000_000


margin_bigger = 0
breaks_margin = 0
for i in range(N):
    dp, dn = random(0, 2), random(0, 2)
    if dp - dn + M > 0:
        breaks_margin += 1
        if margin(dp, dn) > ratio(dp, dn):
            margin_bigger += 1


print(f"Breaks margin: {round(100*breaks_margin/N, 2)} ({breaks_margin} / {N})")
print(f"Margin > Ratio: {round(100*margin_bigger/breaks_margin, 2)} ({margin_bigger} / {breaks_margin})")