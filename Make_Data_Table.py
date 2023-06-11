import csv
import re

"""code that will take the three numbers in each row (two in the #, # pair, and one to the right of the # symbol, 
next to the quantity name) of the file and output a .csv with proper formatting. Each row should have the name of the 
quantity as well"""

import csv
import re

# Read input file
with open('input.txt', 'r') as file:
    lines = file.readlines()

# Extract data
data = []
for line in lines:
    line_data = line.strip().split('#')
    bounds = [float(value) for value in line_data[0].split(',')]
    name_and_value = line_data[1].strip()

    # Extract name and optional true value using regular expressions
    match = re.match(r'(-?\d+(\.\d+)?\s*,?\s+)?([a-zA-Z_][\w-]*)', name_and_value)
    if match:
        true_value, name = match.group(1), match.group(3)

        if true_value:
            true_value = float(true_value.strip().replace(',', ''))
        else:
            true_value = None

        data.append([name] + bounds + [true_value])
    else:
        print(f"Error: Could not parse line: {line}")

# Write to CSV file
with open('output.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Name', 'Lower Bound', 'Upper Bound', 'True Value'])

    for row in data:
        csvwriter.writerow(row)

import csv

# Read the existing CSV file
with open('output.csv', 'r', newline='') as csvfile:
    csvreader = csv.reader(csvfile)
    header = next(csvreader)
    data = [row for row in csvreader]

# Add the new column with values from the list
new_values = [ 1.74629836e+01,  2.19935295e+00,  3.85127752e+01,  1.95114180e+01,
  7.10192771e+01,  5.04453947e+01, -9.11302113e+01, -7.43599277e+01,
  8.24046063e+01, -3.59998169e+01, -3.59924416e+01, -4.02860804e+01,
 -3.05000043e+00, -5.99911928e+00,  4.68343094e+00, -2.32789705e+01,
 -1.02493682e+01, -5.98481069e+01, -7.20144121e+00,  5.63526650e-01,
 -1.03549368e-02, -6.00645034e+01,  1.98220184e+00,  3.03000001e+00,
  1.43803655e+00, -1.06890808e+02,  6.99999990e+00, -9.58811286e+01,
  2.99330399e+01,  6.40000000e-01,  6.61458393e-01,  1.00011364e+02,
  8.21925055e-02,  2.11899648e-01,  2.45238005e-01,  1.52408435e+03,
  3.01365070e+01,  4.47039989e+00,  4.83576881e+00,  1.01927395e+01,
  4.82020541e+00]


for i, row in enumerate(data):
    row.append(new_values[i])

# Write the updated data back to the CSV file
with open('output.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(header + ['New Column'])
    csvwriter.writerows(data)
