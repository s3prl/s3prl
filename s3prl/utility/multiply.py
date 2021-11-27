import sys

if len(sys.argv) < 2:
    print("Usage: python3 multiply.py number1 number2 ...")

result = 1
for number in sys.argv[1:]:
    tgt = float(number)
    result *= tgt
print(result)