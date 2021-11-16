import math
from TinyStatistician import TinyStatistician

tstat = TinyStatistician()
a = [1, 42, 300, 10, 59]
print("Values: ", a)
print()

print("Mean: ")
print(tstat.mean(a))
print(82.4)
print()

print("median: ")
print(tstat.median(a))
print(42.0)
print()

# print("quartile(a, 25): ")
# print(tstat.quartile(a, 25))
# print(10.0)
# print()

# print("quartile(a, 75): ")
# print(tstat.quartile(a, 75))
# print(59.0)
# print()

print("var: ")
print(tstat.var(a))
print(12279.439999999999)
print()

print("std: ")
print(tstat.std(a))
print(110.81263465868862)
print()

data = [42, 7, 69, 18, 352, 3, 650, 754, 438, 2659]
epsilon = 1e-5
err = "Error, grade 0 :("
tstat = TinyStatistician()
assert abs(tstat.mean(data) - 499.2) < epsilon, err
assert abs(tstat.median(data) - 210.5) < epsilon, err

quartile = tstat.quartile(data)
assert abs(quartile[0] - 18) < epsilon, err
assert abs(quartile[1] - 650) < epsilon, err
assert abs(tstat.percentile(data, 10) - 3) < epsilon, err
assert abs(tstat.percentile(data, 28) - 18) < epsilon, err
assert abs(tstat.percentile(data, 83) - 754) < epsilon, err
assert abs(tstat.var(data) - 589194.56) < epsilon, err
assert abs(tstat.std(data) - 767.5900989460456) < epsilon, err
