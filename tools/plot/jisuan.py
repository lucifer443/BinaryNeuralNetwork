import numpy

with open('tools/plot/cos.txt', 'r') as f:
    mydata = f.readlines()
for line in mydata:
    line_data = line.split(',')
    breakpoint()       
    numbers_float = map(float, line_data)
a = 1