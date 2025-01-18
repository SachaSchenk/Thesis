import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.sparse import diags
from scipy.optimize import minimize
from scipy.special import legendre

# Define constants
C_T = 5e8  # J/m^2/K
Q0 = 341.3  # W/m^2, solar radiation constant
alpha1 = 0.70  # Albedo of ice - wrong in your paper
alpha2 = 0.289   # Albedo of water
T1 = 260        # K, temperature threshold
T2 = 290        # K, temperature threshold
M = 0.1         # K^-1
epsilon = 0.61
sigma_0 = 5.67e-8  # W/m^2/K^4, Stefan-Boltzmann constant
D = 0.30  # Diffusivity
num_of_leg_pol = 8  # number of legendre polynomials


# Human intervention parameters
# Define Legendre polynomial expansion for Q_intervention
def Q_intervention(x, b):
    # Compute Q_intervention as a truncated series of Legendre polynomials up to m=5
    Q_int = sum(b[m] * legendre(m)(x) for m in range(num_of_leg_pol))
    return Q_int

# Distribution of solar radiation over latitudes
def Q_distribution(x):
    return Q0 * (1 - 0.241 * (3 * x ** 2 - 1))     # Solar radiation distribution of earth with no intervention

# Albedo function based on temperature
def albedo(T):
    return alpha1 + (alpha2 - alpha1) * (1 + np.tanh(M * (T - (T1 + T2) / 2))) / 2

# Discretize x (latitude) and t (time)
N = 1000  # Number of spatial points
x = np.linspace(-1, 1, N)  # x ranges from -1 (South Pole) to 1 (North Pole)
h = x[1] - x[0]  # Spatial step size
t = np.linspace(0, 100 * 365 * 24 * 3600, 1000)  # 100 years in seconds, 1000 time points
# t = np.linspace(-100 * 365 * 24 * 3600, 0, 1000)  # -100 to 0

# Convert x to theta in degrees
theta_degree = np.arcsin(x) * (180 / np.pi)  # Convert from radians to degrees
# Convert time to years for plotting later
time_years = t / (365 * 24 * 3600)

# Calculate the first derivative ∂T/∂x with boundary conditions
def first_derivative_matrix(N, x):
    main_diag = np.zeros(N)
    off_diag1 = -1/2 * np.ones(N-1)
    off_diag2 = 1/2 * np.ones(N - 1)
    off_diag1[-1] = 0  # South Pole boundary condition
    off_diag2[0] = 0  # North Pole boundary condition
    diffusion_op = diags([off_diag1, main_diag, off_diag2], [-1, 0, 1], shape=(N, N))
    return diffusion_op / h

# Define the second derivative ∂²T/∂x² with boundary conditions
def second_derivative_matrix(N, x):
    main_diag = -2 * np.ones(N)
    off_diag1 = np.ones(N-1)
    off_diag2 = np.ones(N - 1)
    off_diag1[-1] = 2  # South pole boundary condition
    off_diag2[0] = 2  # North pole boundary condition
    diffusion_op = diags([off_diag1, main_diag, off_diag2], [-1, 0, 1], shape=(N, N))
    return diffusion_op / h**2

# Define the system of ODEs
def energy_balance(T, t, intervention=False, Q_int=0, mu_increasing=False):
    # Solar radiation adjusted by human intervention if specified
    if intervention:
        Q = Q_distribution(x) - Q_int
    else:
        Q = Q_distribution(x)
    solar_in = Q * (1 - albedo(T))

    # Outgoing longwave radiation
    longwave_out = epsilon * sigma_0 * T**4

    # Compute time-dependent mu values for each time step
    if mu_increasing:
        mu = mu_initial + t / (10 * 365 * 24 * 3600) * np.ones(N)
        # mu increases 1 units every 10 years at a constant rate - from 29.3 to 39.3 in 100 years
    else:
        mu = mu_initial

    # Diffusion terms
    d2T_dx2 = second_derivative_matrix(N, x).dot(T)             # ∂²T/∂x² term
    dT_dx = first_derivative_matrix(N, x).dot(T)                        # ∂T/∂x term
    diffusion_term = D * ((1 - x**2) * d2T_dx2 - 2 * x * dT_dx)  # Complete diffusion term

    # Energy balance equation
    dT_dt = (solar_in - longwave_out + mu + diffusion_term) / C_T
    return dT_dt

# Define the average temperature function
def compute_global_average_temperature(T_profile):
    """
    Compute the global average temperature at each time step.
    T_profile: 2D array (time steps x latitude bands)
    Returns:
    global_avg_temp: 1D array (global average temperature for each time step)
    """
    return np.mean(T_profile, axis=1)  # Average over latitude bands (x)

# Initial values for stabilizing simulation
T_initial = 288 * np.ones(N)    # Initial temperature profile (constant 288 K)
mu_initial = 29.3 * np.ones(N)    # this value is made up

# We first use mu_initial and T_initial to solve the ODEs to let the temperature distribution stabilize
T_stabilizing = odeint(energy_balance, T_initial, t, args=(False, 0, False))
# Compute global average temperature over time for stabilization
global_avg_temp_stabilizing = compute_global_average_temperature(T_stabilizing)

T_initial_new = T_stabilizing[-1]   # Use the stabilized temperature distribution as new initial temperature distribution





# Optimization: function to minimize: global average temperature at final time
def objective_function(b, use_T1=False, use_T2=False):
    """
    Objective function to minimize.

    b: Coefficients for Q_intervention
    use_T1: Boolean to toggle inclusion of T1 in the objective function
    use_T2: Boolean to toggle inclusion of T2 in the objective function

    Returns:
    Total error combining T0, T1 (if use_T1=True), and T2 (if use_T2=True)
    """
    # Compute Q_intervention with given coefficients b
    Q_int = Q_intervention(x, b)

    # Solve the energy balance equation with intervention
    T_with_Q_int = odeint(energy_balance, T_initial, t, args=(True, Q_int, True))  # Use mu_increasing and intervention

    # Compute the global average temperature at the final time (T0)
    T0_final = compute_global_average_temperature(T_with_Q_int)[-1]

    # Initialize total error with T0
    total_error = T0_final

    # Add T1 to the error if toggled on
    if use_T1:
        T1_final = np.mean(T_with_Q_int[-1] * x)
        total_error += T1_final

    # Add T2 to the error if toggled on
    if use_T2:
        T2_final = np.mean(T_with_Q_int[-1] * (0.5 * (3 * x ** 2 - 1)))
        total_error += T2_final

    return total_error


# Constraint 1: Norm of Q_intervention should not exceed δ
def constraint1(b):
    Q_int = Q_intervention(x, b)
    return delta - np.sum(Q_int, 0)  # Scalar constraint (inequality)

# Constraint 2: Q_intervention cannot exceed Q_distribution at any latitude
def constraint2(b):
    excess = Q_distribution(x) - Q_intervention(x, b)
    return excess  # Array constraint (inequality)

# Constraint 3: Q_intervention must be non-negative
def constraint3(b):
    return Q_intervention(x, b)  # Array constraint (inequality)

# Combine constraints with optional δ constraint toggle
def joined_constraints(b):
    #(constraint1, constraint2, constraint3 (inequalities))
    constraints = [
        {'type': 'ineq', 'fun': constraint1},  # Inequality constraint
        {'type': 'ineq', 'fun': constraint2},  # Inequality constraint
        {'type': 'ineq', 'fun': constraint3}   # Inequality constraint
    ]
    return constraints

# Optimization parameters
initial_guess = 10 * np.random.rand(num_of_leg_pol)  # Initial guess for Legendre coefficients
print("Initial guess Legendre Coefficients: ", initial_guess)
delta = 2500  # Constraint norm for Q_intervention W/m2/s blocked at any given time


# cons = {'type': 'ineq', 'fun': joined_constraints}
cons = joined_constraints(initial_guess)  # Get the right set of constraints



# Perform optimization
result = minimize(objective_function, initial_guess, method='COBYLA', constraints=cons, args=(False, False))  # type: ignore
# Set use_T1 and use_T2 as needed
# Optimal coefficients
b_optimal = result.x

#b_optimal = [12.77337895, -20.06163715, 18.92733972, -11.04697092, 21.36755224, -33.40656263, 35.32836835, -23.88146856] # delta = 12849.34
#Optimal Legendre Coefficients with delta = 12849.34 : [12.77337895, -20.06163715, 18.92733972, -11.04697092, 21.36755224, -33.40656263, 35.32836835, -23.88146856] # delta = 12849.34


print("Optimal Legendre Coefficients with delta =", delta, ":", b_optimal)
# print(constraint1(b_optimal))
# print(constraint2(b_optimal))
# print(constraint3(b_optimal))
# To check that the constraints are met (every value must be non-negative)



# Optimal Q_intervention given b_optimal
Q_int_optimal = Q_intervention(x, b_optimal)
# To use reverse intervention, wrap it inside list(reversed()) which should give approx same final temp due to symmetry
# Q_int_init_guess = Q_intervention(x, initial_guess)


# Solve the ODEs with and without intervention
T_no_intervention = odeint(energy_balance, T_initial_new, t, args=(False, 0, True))
T_with_intervention = odeint(energy_balance, T_initial_new, t, args=(True, Q_int_optimal, True))
# T_with_intervention_init = odeint(energy_balance, T_initial_new, t, args=(True, Q_int_init_guess, True))
# This line was to check if the optimization process found a better solution than the random initial Legendre values

# Compute global average temperature over time
global_avg_temp_no_intervention = compute_global_average_temperature(T_no_intervention)
global_avg_temp_with_intervention = compute_global_average_temperature(T_with_intervention)
# global_avg_temp_with_intervention_init = compute_global_average_temperature(T_with_intervention_init)

print("Starting Average Global Temperature:", global_avg_temp_no_intervention[0])
print("Final Average Global Temperature without Intervention: ", global_avg_temp_no_intervention[-1])
print("Final Average Global Temperature with optimal Intervention: ", global_avg_temp_with_intervention[-1])


# Fig 1a: Plotting temperature distribution over time stabilizing and without human intervention
fig1a, axs = plt.subplots(2, 1, figsize=(10, 18))  # 2 rows, 1 column

# Times to plot
times_to_plot = [0, int(len(t) / 20), int(len(t) / 2), -1]  # Plot at initial, 5, 50, and 100 years
times_to_plot2 = [0, int(len(t) / 2), -1]  # Plot at initial, 50, and 100 years
times_to_plot3 = [0, int(len(t) / 20), int(len(t) / 4), -1]  # Plot at initial, 5, 25, and 100 years

for i, time_index in enumerate(times_to_plot3):
    axs[0].plot(theta_degree, T_stabilizing[time_index], label=f'Time = {time_years[time_index]:.1f} years')

for i, time_index in enumerate(times_to_plot2):
    axs[1].plot(theta_degree, T_no_intervention[time_index], label=f'Time = {time_years[time_index]:.1f} years')

# Compute global y-axis limits across all datasets
min_temp_all = min(np.min(T_stabilizing), np.min(T_no_intervention), np.min(T_with_intervention))
max_temp_all = max(np.max(T_stabilizing), np.max(T_no_intervention), np.max(T_with_intervention))

# Add a buffer for visibility (±5 units/degrees)
y_min = min_temp_all - 5
y_max = max_temp_all + 5

# Titles for subplots
titles = [
    "Stabilizing Temperature Profile Over Time - T initial constant",
    "Temperature Profile Over Time (No Human Intervention)"
]

# Set titles, labels, and limits for plots
for i, ax in enumerate(axs):
#    ax.set_title(titles[i], fontsize=14)
    ax.set_xlabel("Latitude θ", fontsize=14)
    ax.set_ylabel("Temperature (K)", fontsize=14)
    ax.set_xlim(min(theta_degree), max(theta_degree))  # Latitude range
    ax.set_ylim(y_min, y_max)  # Use global y-axis range
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Set legend inside the plot, at the bottom center
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=1, fontsize=12, frameon=True)

# Adjust layout
plt.tight_layout(pad=4.0, h_pad=5.0)



# Figure 1b (Temperature Profile Over Time with Human Intervention)
fig1b, ax2 = plt.subplots(figsize=(10, 6))  # Create a new figure

# Plot (with intervention)
for i, time_index in enumerate(times_to_plot):
    ax2.plot(theta_degree, T_with_intervention[time_index], label=f'Time = {time_years[time_index]:.1f} years')

# Set title, labels, and limits
#ax2.set_title(r'Temperature Profile over Time (With Human Intervention) - $\delta = ' + f'{delta:.0f}' + r' \, \mathrm{W/m^2/s}$', fontsize=14)
ax2.set_xlabel("Latitude θ", fontsize=14)
ax2.set_ylabel("Temperature (K)", fontsize=14)
ax2.set_xlim(min(theta_degree), max(theta_degree))  # Latitude range
ax2.set_ylim(y_min, y_max)  # Use the same global y-axis range
ax2.tick_params(axis='both', which='major', labelsize=12)

# Set legend inside the plot, at the bottom center
ax2.legend(loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=1, fontsize=12, frameon=True)

# Adjust layout
plt.tight_layout()



# Figure 1c (Temperature Profile at 100y with and without Human Intervention)
fig1c = plt.figure(figsize=(10, 6))  # Create a new figure

# Plot (100y with and without intervention)
plt.plot(theta_degree, T_with_intervention[-1], label="Optimal Intervention at 100 years", color="green")
plt.plot(theta_degree, T_no_intervention[-1], label="No Intervention at 100 years", color="red")
plt.plot(theta_degree, T_no_intervention[0], label="Temperature Distribution at 0 years", color="blue", linestyle='--')

# Set title, labels, and limits
#plt.title(r'Temperature Profile with and without Human Intervention at 100 years - $\delta = ' + f'{delta:.0f}' + r' \, \mathrm{W/m^2/s}$', fontsize=14)
plt.xlabel("Latitude θ", fontsize=14)
plt.ylabel("Temperature (K)", fontsize=14)
plt.xlim(min(theta_degree), max(theta_degree))  # Latitude range
plt.ylim(y_min, y_max)  # Use the same global y-axis range
plt.tick_params(axis='both', which='major', labelsize=12)

# Set legend inside the plot, at the bottom center
plt.legend(loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=1, fontsize=12, frameon=True)

# Adjust layout
plt.tight_layout()



# Fig 2: Global Average Temperatures over time

# --- Compute global y-axis limits (same as before) ---
min_temp_all = min(np.min(global_avg_temp_stabilizing),
                   np.min(global_avg_temp_no_intervention),
                   np.min(global_avg_temp_with_intervention))

max_temp_all = max(np.max(global_avg_temp_stabilizing),
                   np.max(global_avg_temp_no_intervention),
                   np.max(global_avg_temp_with_intervention))

# Add buffer (±1 unit) for better visualization
y_min = min_temp_all - 5
y_max = max_temp_all + 5

# Create figure for the stabilizing plot
fig2a = plt.figure(figsize=(10, 6))
ax1 = fig2a.add_subplot(111)
ax1.plot(time_years, global_avg_temp_stabilizing, label="Stabilizing Average Temperature", color="blue")
#ax1.set_title("Stabilizing Global Average Temperature Over Time - T initial constant", fontsize=14)
ax1.set_xlabel("Time (years)", fontsize=14)
ax1.set_ylabel("Global Average Temperature (K)", fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=12)
#ax1.legend(loc="upper left", ncol=1, fontsize=12, frameon=True)

# Set the same y-axis range
ax1.set_ylim(y_min, y_max)
# Restrict x-axis to 0 to 100 years
ax1.set_xlim(0, 100)

# Create figure for the comparison plot (no and with human intervention)
fig2b = plt.figure(figsize=(10, 6))
ax2 = fig2b.add_subplot(111)
ax2.plot(time_years, global_avg_temp_no_intervention, label="No Human Intervention", color="red")
# ax2.plot(time_years, global_avg_temp_with_intervention, label="With Human Intervention", color="green")
#ax2.set_title(r'Global Average Temperature over Time with and without Intervention - $\delta = ' + f'{delta:.0f}' + r' \, \mathrm{W/m^2/s}$', fontsize=14)
ax2.set_xlabel("Time (years)", fontsize=14)
ax2.set_ylabel("Global Average Temperature (K)", fontsize=14)
ax2.tick_params(axis='both', which='major', labelsize=12)
# ax2.legend(loc="upper left", ncol=1, fontsize=12, frameon=True)

# Set the same y-axis range
ax2.set_ylim(y_min, y_max)
# Restrict x-axis to 0 to 100 years
ax2.set_xlim(0, 100)


# Adjust layout
plt.tight_layout()




# Fig 3: Plot optimal Q_intervention and Q_distribution
fig3, axs = plt.subplots(2, 1, figsize=(10, 12))
axs[1].plot(theta_degree, Q_distribution(x) - Q_int_optimal, label=r'$\delta = 12849 \, \mathrm{W/m^2/s}$', color="black")
axs[1].plot(theta_degree, Q_distribution(x), label=r'$\delta = 0 \, \mathrm{W/m^2/s}$', color="orange")
#axs[1].set_title('Distribution of Intervention over Latitudes\n'+ 'compared to incoming Solar Radiation - '+ r'$\delta = ' + f'{delta:.0f}' + r' \, \mathrm{W/m^2/s}$', fontsize=14)
axs[1].set_xlabel("Latitude θ", fontsize=14)
axs[1].set_ylabel("Surface sunlight (W/m2/s)", fontsize=14)
axs[1].legend(fontsize=12)
axs[1].tick_params(axis='both', which='major', labelsize=12)
# Restrict x-axis to -90 to 90 degrees
axs[1].set_xlim(-90, 90)

axs[0].plot(theta_degree, Q_int_optimal, label="Optimal Q_intervention", color="black")
#axs[0].set_title(r'Distribution of Intervention over Latitudes - $\delta = ' + f'{delta:.0f}' + r' \, \mathrm{W/m^2/s}$', fontsize=14)
axs[0].set_xlabel("Latitude θ", fontsize=14)
axs[0].set_ylabel("blocked sunlight (W/m2/s)", fontsize=14)
# axs[0].legend(fontsize=12)
# Restrict x-axis to -90 to 90 degrees
axs[0].set_xlim(-90, 90)
axs[0].tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout(pad=3.0, h_pad=3.0)


# Fig 4: Create animations Temperature Distributions

# Create the first figure for the stabilizing profile plot
fig1, ax1 = plt.subplots(figsize=(10, 6))

# Initialize empty line for stabilization plot
line1, = ax1.plot([], [], label="Stabilizing", color="blue")

# Set axis labels, title, and grid for the first figure
ax1.set_title("Stabilizing Temperature Profile Over Time - T initial constant", fontsize=14)
ax1.set_xlabel("Latitude θ", fontsize=12)
ax1.set_ylabel("Temperature (K)", fontsize=12)
ax1.set_xlim(min(theta_degree), max(theta_degree))  # Latitude range

# Create the second figure for the "No vs. With Human Intervention" plot
fig2, ax2 = plt.subplots(figsize=(10, 6))

# Initialize empty lines for no intervention and with intervention
line2, = ax2.plot([], [], label="No Intervention", color="red")
line3, = ax2.plot([], [], label="With Intervention", color="green")

# Add a static line for the stabilized final distribution in the second subplot
final_stabilized_line = ax2.plot(theta_degree, T_stabilizing[-1], label="Starting Distribution", color="blue", linestyle='--')[0]

# Set axis labels, title, and grid for the second figure
ax2.set_title("Temperature Profile Over Time (No vs. With Human Intervention)", fontsize=14)
ax2.set_xlabel("Latitude θ", fontsize=12)
ax2.set_ylabel("Temperature (K)", fontsize=12)
ax2.set_xlim(min(theta_degree), max(theta_degree))  # Latitude range

# Compute global y-axis limits across all datasets
min_temp_all = min(np.min(T_stabilizing), np.min(T_no_intervention), np.min(T_with_intervention))
max_temp_all = max(np.max(T_stabilizing), np.max(T_no_intervention), np.max(T_with_intervention))

# Add a buffer for safety (±5 units)
y_min = min_temp_all - 5
y_max = max_temp_all + 5

# Apply the same y-axis range to both figures
ax1.set_ylim(y_min, y_max)
ax2.set_ylim(y_min, y_max)

# Adjusting legends inside subplots at the bottom-center
ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=1, fontsize=12, frameon=True)
ax2.legend(loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=1, fontsize=12, frameon=True)

# Set tight layout for both figures
fig1.tight_layout()
fig2.tight_layout()

# Initialize the plot (called once)
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    return line1, line2, line3

# Update function for each frame
def update(frame):
    # Update stabilization line in the first figure
    line1.set_data(theta_degree, T_stabilizing[frame])

    # Update no intervention and with intervention lines in the second figure
    line2.set_data(theta_degree, T_no_intervention[frame])
    line3.set_data(theta_degree, T_with_intervention[frame])

    # Update titles with current year in both figures
    ax1.set_title(f"Stabilizing Temperature Profile - Year: {time_years[frame]:.1f}", fontsize=14)
    ax2.set_title(f"Temperature Distribution With vs Without Intervention - $\\delta = {delta} \\, \\mathrm{{W/m^2/s}}$ - Year: {time_years[frame]:.1f}", fontsize=14)


    return line1, line2, line3

# Create the animation
frames = range(0, len(time_years), 20)
ani1 = FuncAnimation(fig1, update, frames=frames, interval=100)  # type: ignore
ani2 = FuncAnimation(fig2, update, frames=frames, interval=100)  # type: ignore

# Save the animation to a file
# For mp4
# ani1.save(r'C:\Users\Sacha\Downloads\animations-scriptie\fig4a-ani-stabilizing.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
# For GIF:
ani1.save(r'C:\Users\Sacha\Downloads\animations-scriptie\fig4a-ani-stabilizing.gif', fps=10, writer='pillow')
ani2.save(rf'C:\Users\Sacha\Downloads\animations-scriptie\fig4b-ani-delta{delta}.gif', fps=10, writer='pillow')
fig1a.savefig(r'C:\Users\Sacha\Downloads\animations-scriptie\fig1a-stabilizing-temperature-distribution.png')
fig1b.savefig(rf'C:\Users\Sacha\Downloads\animations-scriptie\fig1b-temp-profile-delta{delta}.png')
fig1c.savefig(rf'C:\Users\Sacha\Downloads\animations-scriptie\fig1c-temp-profile-compare-delta{delta}.png')
fig2a.savefig(r'C:\Users\Sacha\Downloads\animations-scriptie\fig2a-stabilizing-average-temperature.png')
fig2b.savefig(rf'C:\Users\Sacha\Downloads\animations-scriptie\fig2b-compare-average-temp-delta{delta}.png')
fig3.savefig(rf'C:\Users\Sacha\Downloads\animations-scriptie\fig3-intervention-distribution-delta{delta}.png')

# Change the start of all the file paths if you want to save the animations
# Otherwise you can just remove this



# optimal Legendre values per delta scenario
b_optimal25 = [2.47953630e+00, -1.46069856e-02, 9.07119728e+00, -1.79603939e-02, 8.14240402e+00, -2.44784840e-03, 3.19148479e+00, 1.62311402e-03] # delta=2500
b_optimal5 = [4.95797872,  9.23128638, 16.17205797, 14.70196581, 16.24743947, 15.28163493, 9.46505683, 7.62752356] # delta=5000
b_optimal75 = [7.43544914, 14.95411568, 22.64989934, 22.07767407, 24.60158216, 23.11820085, 17.07497206, 11.61191212] # delta=7500
b_optimal10 = [9.92122649, 18.06331955, 24.79447182, 22.33601898, 28.02918462, 30.31599273, 25.65176886, 17.68132053] # delta=10.000
b_optimal125 = [12.42369288, 19.75639176, 19.47646499, 12.24644959, 22.13127931, 33.1407567, 34.36523969, 23.25307882] # delta=12.500
b_optimal15 = [14.92617042, 22.55563928, 15.54756242, 4.08602631, 16.99523532, 35.36445628, 40.92773183, 26.39057813] # delta=15000

# Intervention distribution per delta scenario
Q_int_optimal25 = Q_intervention(x, b_optimal25)
Q_int_optimal5 = Q_intervention(x, b_optimal5)
Q_int_optimal75 = Q_intervention(x, b_optimal75)
Q_int_optimal10 = Q_intervention(x, b_optimal10)
Q_int_optimal125 = Q_intervention(x, b_optimal125)
Q_int_optimal15 = Q_intervention(x, b_optimal15)

# Temperature profile per delta scenario
T_with_intervention25 = odeint(energy_balance, T_initial_new, t, args=(True, Q_int_optimal25, True))
T_with_intervention5 = odeint(energy_balance, T_initial_new, t, args=(True, Q_int_optimal5, True))
T_with_intervention75 = odeint(energy_balance, T_initial_new, t, args=(True, Q_int_optimal75, True))
T_with_intervention10 = odeint(energy_balance, T_initial_new, t, args=(True, Q_int_optimal10, True))
T_with_intervention125 = odeint(energy_balance, T_initial_new, t, args=(True, Q_int_optimal125, True))
T_with_intervention15 = odeint(energy_balance, T_initial_new, t, args=(True, Q_int_optimal15, True))

# Global average temperature per delta scenario
global_avg_temp_with_intervention25 = compute_global_average_temperature(T_with_intervention25)
global_avg_temp_with_intervention5 = compute_global_average_temperature(T_with_intervention5)
global_avg_temp_with_intervention75 = compute_global_average_temperature(T_with_intervention75)
global_avg_temp_with_intervention10 = compute_global_average_temperature(T_with_intervention10)
global_avg_temp_with_intervention125 = compute_global_average_temperature(T_with_intervention125)
global_avg_temp_with_intervention15 = compute_global_average_temperature(T_with_intervention15)


# Create figure 5 for the comparison plot (no intervention and every delta)
fig5 = plt.figure(figsize=(10, 6))
plt.plot(time_years, global_avg_temp_no_intervention, label="No Human Intervention", color="red")
plt.plot(time_years, global_avg_temp_with_intervention25, label=r'$\delta = 2500 \, \mathrm{W/m^2/s}$', color="blue")
plt.plot(time_years, global_avg_temp_with_intervention5, label=r'$\delta = 5000 \, \mathrm{W/m^2/s}$', color="orange")
plt.plot(time_years, global_avg_temp_with_intervention75, label=r'$\delta = 7500 \, \mathrm{W/m^2/s}$', color="green")
plt.plot(time_years, global_avg_temp_with_intervention10, label=r'$\delta = 10000 \, \mathrm{W/m^2/s}$', color="brown")
plt.plot(time_years, global_avg_temp_with_intervention125, label=r'$\delta = 12500 \, \mathrm{W/m^2/s}$', color="deeppink")
plt.plot(time_years, global_avg_temp_with_intervention15, label=r'$\delta = 15000 \, \mathrm{W/m^2/s}$',color="aqua")
#plt.title(r'Global Average Temperature over Time for delta 2500, 5000, 7500, 10000, 12500 and 15000, fontsize=14)
plt.xlabel("Time (years)", fontsize=14)
plt.ylabel("Global Average Temperature (K)", fontsize=14)
axs[1].tick_params(axis='both', which='major', labelsize=12)
plt.legend(loc="upper left", ncol=1, fontsize=12, frameon=True)

# Restrict x-axis to 0 to 100 years
plt.xlim(0, 100)

# Adjust layout
plt.tight_layout()



# Data
delta_values = np.array([0, 2500, 5000, 7500, 10000, 12500, 15000])  # Delta values
global_avg_temps = np.array([
    global_avg_temp_no_intervention[-1],
    global_avg_temp_with_intervention25[-1],
    global_avg_temp_with_intervention5[-1],
    global_avg_temp_with_intervention75[-1],
    global_avg_temp_with_intervention10[-1],
    global_avg_temp_with_intervention125[-1],
    global_avg_temp_with_intervention15[-1],
])  # Corresponding final temperatures

# Fit a linear approximation
coefficients = np.polyfit(delta_values, global_avg_temps, 1)  # Linear fit (degree=1)
linear_fit = np.poly1d(coefficients)  # Create linear fit function

# Generate values for the linear fit line
delta_fit = np.linspace(delta_values.min(), delta_values.max(), 100)  # More points for a smooth line
temps_fit = linear_fit(delta_fit)

# Plot the data points and the linear fit
plt.figure(figsize=(10, 6))
plt.plot(delta_fit, temps_fit, label="Linear Approximation", color="black", linestyle="--")  # Linear fit line
plt.scatter(delta_values, global_avg_temps, label="Data Points", c=['red', 'blue', 'orange', 'green', 'brown', 'deeppink', 'aqua'], s=80, zorder=5)  # Data points

# Add labels, title, and legend
plt.xlabel(r"Allowed Intervention Quantity $\delta$ (W/m$^2$)", fontsize=14)
plt.ylabel("Final Global Average Temperature (K)", fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=12)
#plt.title("Final Global Average Temperature vs. $\delta$", fontsize=16)
plt.legend(fontsize=12)
plt.xlim(0, 15000)

plt.tight_layout()


# Fig 7: Plot optimal Q_intervention for every delta
fig7, axs = plt.subplots(2, 1, figsize=(10, 12))
axs[1].plot(theta_degree, Q_distribution(x), label=r'$\delta = 0 \, \mathrm{W/m^2/s}$', color="black")
axs[1].plot(theta_degree, Q_distribution(x) - Q_int_optimal25, label=r'$\delta = 2500 \, \mathrm{W/m^2/s}$')
axs[1].plot(theta_degree, Q_distribution(x) - Q_int_optimal5, label=r'$\delta = 5000 \, \mathrm{W/m^2/s}$')
axs[1].plot(theta_degree, Q_distribution(x) - Q_int_optimal75, label=r'$\delta = 7500 \, \mathrm{W/m^2/s}$')
axs[1].plot(theta_degree, Q_distribution(x) - Q_int_optimal10, label=r'$\delta = 10000 \, \mathrm{W/m^2/s}$')
axs[1].plot(theta_degree, Q_distribution(x) - Q_int_optimal125, label=r'$\delta = 12500 \, \mathrm{W/m^2/s}$')
axs[1].plot(theta_degree, Q_distribution(x) - Q_int_optimal15, label=r'$\delta = 15000 \, \mathrm{W/m^2/s}$')
#axs[1].set_title('Distribution of Intervention over Latitudes for every delta compared to incoming Solar Radiation', fontsize=14)
axs[1].set_xlabel("Latitude θ", fontsize=14)
axs[1].set_ylabel("Surface sunlight (W/m2/s)", fontsize=14)
axs[1].legend(fontsize=12)
axs[1].tick_params(axis='both', which='major', labelsize=12)
# Restrict x-axis to -90 to 90 degrees
axs[1].set_xlim(-90, 90)

axs[0].plot(theta_degree, Q_int_optimal25, label=r'$\delta = 2500 \, \mathrm{W/m^2/s}$')
axs[0].plot(theta_degree, Q_int_optimal5, label=r'$\delta = 5000 \, \mathrm{W/m^2/s}$')
axs[0].plot(theta_degree, Q_int_optimal75, label=r'$\delta = 7500 \, \mathrm{W/m^2/s}$')
axs[0].plot(theta_degree, Q_int_optimal10, label=r'$\delta = 10000 \, \mathrm{W/m^2/s}$')
axs[0].plot(theta_degree, Q_int_optimal125, label=r'$\delta = 12500 \, \mathrm{W/m^2/s}$')
axs[0].plot(theta_degree, Q_int_optimal15, label=r'$\delta = 15000 \, \mathrm{W/m^2/s}$')
#axs[0].set_title(r'Distribution of Intervention over Latitudes for every delta', fontsize=14)
axs[0].set_xlabel("Latitude θ", fontsize=14)
axs[0].set_ylabel("Blocked sunlight (W/m2/s)", fontsize=14)
axs[0].tick_params(axis='both', which='major', labelsize=12)
axs[0].legend(fontsize=12)
# Restrict x-axis to -90 to 90 degrees
axs[0].set_xlim(-90, 90)
plt.tight_layout(pad=3.0, h_pad=3.0)


# Target temperatures
target_temp_1 = global_avg_temp_no_intervention[0]
target_temp_2 = global_avg_temp_no_intervention[0] + 1.5

# Linear fit coefficients
m, c = coefficients

# Solve for delta values
delta_target_1 = (target_temp_1 - c) / m
delta_target_2 = (target_temp_2 - c) / m

# Print results
print(f"Delta where the linear fit temperature equals starting temperature: {delta_target_1:.2f}")
print(f"Delta where the linear fit temperature equals starting temperature + 1.5: {delta_target_2:.2f}")

"""



# Show animation and plots
plt.show()