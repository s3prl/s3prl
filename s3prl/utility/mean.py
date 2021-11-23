import sys

numbers = [float(item) for item in sys.argv[1:]]
print(sum(numbers) / len(numbers))
