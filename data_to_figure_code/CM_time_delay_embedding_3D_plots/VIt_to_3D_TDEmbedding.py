import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load your input data (replace 'your_data.txt' with your actual data file)
# data = np.loadtxt('phasic/2014_09_10_0013_VIt.txt')[::10,1]
# data = np.loadtxt('tonic/2014_12_11_0017_VIt.txt')[::10,1]
# quantity = "I" # current
# units = "pA"
Voltage = np.loadtxt('tonic/2014_12_11_0017_VIt.txt')[:,0]
Current = np.loadtxt('tonic/2014_12_11_0017_VIt.txt')[:,1]
data = Voltage
quantity = "V" # current
units = "mV"

skip_window = 1

plot_original = True
if plot_original:
    plt.figure()
    plt.plot(data[::skip_window])
    plt.show()

# Create the time-delay embedding
def time_delay_embedding(data, delay=1):
    embedded_data = np.zeros((data.shape[0] - delay * 2, 3))
    for i in range(embedded_data.shape[0]):
        embedded_data[i] = [data[i], data[i + delay], data[i + 2 * delay]]
    return embedded_data

# Set the time delay (you can adjust this value)
delay = 5

# Generate the embedded data
embedded_data = time_delay_embedding(data, delay)[::skip_window]

# Plot the 3D scatterplot using matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(embedded_data[:, 0], embedded_data[:, 1], embedded_data[:, 2], s=0.1)

ax.set_xlabel(f'{quantity}(t) ({units})')
ax.set_ylabel(f'{quantity}(t + {delay}) ({units})')
ax.set_zlabel(f'{quantity}(t + {2 * delay}) ({units})')
plt.title(f"Voltage with  tau={delay} timesteps")
plt.show()


# Plot the 3D scatterplot of {V(t), V(t-12tau), I(t)} using matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(embedded_data[:, 0], embedded_data[:, 1], Current[0: - delay * 2][::skip_window], s=0.1)

ax.set_xlabel(f'V(t) ({units})')
ax.set_ylabel(f'V(t + {delay}) ({units})')
ax.set_zlabel(f'I(t)')
plt.title(f"3D Down-projection with  tau={delay} timesteps")
plt.show()


#================== Test Rossler System =================================
from scipy.integrate import solve_ivp

def rossler_system(t, state, a=0.2, b=0.2, c=5.7):
    x, y, z = state
    dxdt = -y - z
    dydt = x + a * y
    dzdt = b + z * (x - c)
    return [dxdt, dydt, dzdt]
# Set initial conditions and parameters
initial_state = [0.1, 0.1, 0.1]
t_span = (0, 1000)
t_eval = np.linspace(*t_span, 10000)
# Solve the Rössler system
solution = solve_ivp(rossler_system, t_span, initial_state, t_eval=t_eval)
# Extract the x variable time series
Rossler_data = solution.y[0]

# Set the time delay (you can adjust this value)
delay = 3

# Generate the embedded data
embedded_data = time_delay_embedding(Rossler_data, delay)

# Plot the 3D scatterplot using matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(embedded_data[:, 0], embedded_data[:, 1], embedded_data[:, 2], s=0.1)

ax.set_xlabel(f'{quantity}(t) ({units})')
ax.set_ylabel(f'{quantity}(t + {delay}) ({units})')
ax.set_zlabel(f'{quantity}(t + {2 * delay}) ({units})')
plt.title(f"Rossler with tau={delay} timesteps")

plt.show()
