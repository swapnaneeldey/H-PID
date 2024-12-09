import numpy as np
import matplotlib.pyplot as plt

"""
# Given datasets
num_steps_051 = [-4.2009, -4.1859, -4.0902, -4.0782, -4.0540]
num_steps_101 = [-3.4600, -3.4803, -3.4005, -3.4144, -3.4978]
num_steps_201 = [-3.3300, -3.3691, -3.3437, -3.3209, -3.2957]

# True value for logZ
true_value = -3.3290

# Prepare data for plotting
data = [num_steps_051, num_steps_101, num_steps_201]

# Plot
plt.figure(figsize=(8, 6))
boxprops = dict(linewidth=2)  # Set thickness for the box edges
whiskerprops = dict(linewidth=2)  # Set thickness for whiskers
capprops = dict(linewidth=2)  # Set thickness for caps
medianprops = dict(linewidth=2)  # Set thickness for the median line

plt.boxplot(data, 
            labels=['50', '100', '200'],
            boxprops=boxprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            medianprops=medianprops)

plt.ylabel(r'$\log Z$', fontsize=18)
plt.xlabel('Number of Discretization Steps', fontsize=18)
#plt.grid(True)
plt.axhline(y=true_value, color='r', linestyle='--', label=f'True value: {true_value}')

# Adjust tick parameters
plt.tick_params(axis='x', labelsize=16)  # Set x-axis tick label size
plt.tick_params(axis='y', labelsize=16)  # Set y-axis tick label size

# Add legend
plt.legend(fontsize=16)

# Save the plot
plt.savefig('boxplot.png')

# Show the plot
plt.show()
"""

# Given datasets
num_samples_1000  = [-4.1945, -4.4736, -4.4348, -4.2835, -4.1161]
num_samples_5000  = [-3.5245, -3.4593, -3.5425, -3.5036, -3.4823]
num_samples_10000 = [-3.3390, -3.3200, -3.3300, -3.3230, -3.3296]

# True value for logZ
true_value = -3.3290

# Prepare data for plotting
data = [num_samples_1000, num_samples_5000, num_samples_10000]

# Plot
plt.figure(figsize=(8, 6))
boxprops = dict(linewidth=2)  # Set thickness for the box edges
whiskerprops = dict(linewidth=2)  # Set thickness for whiskers
capprops = dict(linewidth=2)  # Set thickness for caps
medianprops = dict(linewidth=2)  # Set thickness for the median line

plt.boxplot(data, 
            labels=['1000', '5000', '10000'],
            boxprops=boxprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            medianprops=medianprops)

plt.ylabel(r'$\log Z$', fontsize=18)
plt.xlabel('Number of Samples', fontsize=18)
#plt.grid(True)
plt.axhline(y=true_value, color='r', linestyle='--', label=f'True value: {true_value}')

# Adjust tick parameters
plt.tick_params(axis='x', labelsize=16)  # Set x-axis tick label size
plt.tick_params(axis='y', labelsize=16)  # Set y-axis tick label size

# Add legend
plt.legend(fontsize=16)

# Save the plot
plt.savefig('boxplot.png')



