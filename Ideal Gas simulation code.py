import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Defining the Particle class and functions
class Particle: #defining the physics of elastic collisions
    def __init__(self, mass, radius, position, velocity):
        self.mass = mass #mass of the particle
        self.radius = radius #radius of the partivle
        self.position = np.array(position) #position of the particle
        self.velocity = np.array(velocity) #velocity of the particle
        self.solpos = [np.copy(self.position)] #positions recorded during the simulation
        self.solvel = [np.copy(self.velocity)] #velocicities recorded during the simulation
        self.solvel_mag = [np.linalg.norm(np.copy(self.velocity))] #magnitude recorded during the simulation
        
    def compute_step(self, step):  #computing position of the particle in the next step
        self.position += step * self.velocity  #updating the position based on the current velocity and the time step
        self.solpos.append(np.copy(self.position))  #recording updated position
        self.solvel.append(np.copy(self.velocity))  #recording current velocity
        self.solvel_mag.append(np.linalg.norm(np.copy(self.velocity)))  #recording magnitude of the current velocity

    def check_coll(self, particle):  #check for collision with another particle
        r1, r2 = self.radius, particle.radius  #get radii of the two particles
        x1, x2 = self.position, particle.position  #get positions of the two particles
        di = x2 - x1  #calculating vector between the two particles
        norm = np.linalg.norm(di)  #calculating distance between the two particles
        return norm - (r1 + r2) * 1.1 < 0  #return True if the distance < sum of radii multiplied by a safety factor
        
    def compute_coll(self, particle, step):  #computing velocity after collision with another particle
        m1, m2 = self.mass, particle.mass  #get masses of the two particles
        v1, v2 = self.velocity, particle.velocity  #get velocities of the two particles
        x1, x2 = self.position, particle.position  #get positions of the two particles
        di = x2 - x1  #calculating vector between the two particles
        self.velocity = v1 - 2. * m2 / (m1 + m2) * np.dot(v1 - v2, di) / (np.linalg.norm(di)**2.) * di  #calculating new velocity of the current particle
        particle.velocity = v2 - 2. * m1 / (m2 + m1) * np.dot(v2 - v1, (-di)) / (np.linalg.norm(di)**2.) * (-di)  #calculating new velocity of the colliding particle
            
    def compute_refl(self, step, size):  #computing velocity after hitting an edge
        r, v, x = self.radius, self.velocity, self.position  #get radius, velocity, and position of particle
        projx = step * abs(np.dot(v, np.array([1.,0.])))  #calculating projection of velocity on x-axis
        projy = step * abs(np.dot(v, np.array([0.,1.])))  #calculating projection of velocity on y-axis
        if abs(x[0]) - r < projx or abs(size - x[0]) - r < projx:  #check if particle hits the left or right edge
            self.velocity[0] *= -1  #reflecting the x-component of velocity
        if abs(x[1]) - r < projy or abs(size - x[1]) - r < projy:  #check if the particle hits the top or bottom edge
            self.velocity[1] *= -1  #reflecting the y-component of velocity

def solve_step(particle_list, step, size):  #solving a step for every particle
    #detect an edge-hitting and collisions of every particle
    for i in range(len(particle_list)):  #iterating over all particles
        particle_list[i].compute_refl(step, size)  #computing reflection if the particle hits an edge
        for j in range(i + 1, len(particle_list)):  #iterating over remaining particles
            if particle_list[i].check_coll(particle_list[j]):  #check for collision with another particle
                particle_list[i].compute_coll(particle_list[j], step)  #computing velocity after collision
    #computing position of every particle
    for particle in particle_list:  #iterating over all particles
        particle.compute_step(step)  #computing position of the particle after the step

def init_list_random(N, radius, mass, boxsize):  #generating N particles randomly
    particle_list = []  #initializing an empty list to store particles
    for i in range(N):  #looping over the number of particles
        v_mag = np.random.rand(1) * 6  #generating random magnitude for velocity
        v_ang = np.random.rand(1) * 2 * np.pi  #generating random angle for velocity
        v = np.append(v_mag * np.cos(v_ang), v_mag * np.sin(v_ang))  #calculating the velocity vector
        collision = True  #setting collision flag to True
        while collision:  #repeat until no collision occurs
            collision = False  #assuming no collision initially
            pos = radius + np.random.rand(2) * (boxsize - 2 * radius)  #generating random position within the box
            newparticle = Particle(mass, radius, pos, v)  #creating new particle with the generated properties
            for j in range(len(particle_list)):  #looping over existing particles
                collision = newparticle.check_coll(particle_list[j])  #checking for collision with each existing particle
                if collision:  #if collision occurs
                    break  #exit loop
        particle_list.append(newparticle)  #adding new particle to the list
    return particle_list  #returning the list of particles

def total_Energy(particle_list, index):  #computing total energy of the system
    return sum([particle_list[i].mass / 2. * particle_list[i].solvel_mag[index]**2 for i in range(len(particle_list))])
    #summing KE of all particles in the system at a specific index (frame)
    #sum is computed using a list comprehension iterating over all particles in the list

# Simulation parameters
particle_number = 200 #number of particles
boxsize = 200 #size of the box
tfin = 50 # total time of simulation
stepnumber = 200 #final number of frames
timestep = tfin / stepnumber #timestep for each frame

# Initializing particles and plot
particle_list = init_list_random(particle_number, radius=2, mass=1, boxsize=boxsize) #setting the Temperature!!!!
fig, (ax, hist) = plt.subplots(1, 2, figsize=(12, 6)) 
ax.axis('equal')
ax.axis([-1, 30, -1, 30]) #setting the limits for x and y axes
ax.xaxis.set_visible(False) #hiding the x-axis labels
ax.yaxis.set_visible(False) #hiding the y-axis labels
ax.set_xlim([0, boxsize]) #setting the x-axis limits
ax.set_ylim([0, boxsize]) #setting the y-axis limits
ax.set_title('Ideal Gas Particles (T=60K, P=1atm)')  # Title for the left subplot

# Drawing the ideal gas particles as circles
circles = [plt.Circle((particle_list[i].solpos[0][0], particle_list[i].solpos[0][1]), particle_list[i].radius, ec="black", fc="deepskyblue", lw=1.5, zorder=20) for i in range(particle_number)]
for circle in circles:
    ax.add_patch(circle)

def update(frame):  #animation function
    solve_step(particle_list, timestep, boxsize)  #solving a step for every particle
    for i, particle in enumerate(particle_list):
        circles[i].center = particle.solpos[frame][0], particle.solpos[frame][1]  #updating position of each particle in the plot

    hist.clear()  # Clear the histogram plot
    vel_mod = [particle_list[j].solvel_mag[frame] for j in range(len(particle_list))]  #calculating speed mag. for each particle at the current frame
    hist.hist(vel_mod, bins=30, density=True, color='silver', label="Simulation Data")  #plotting histogram of speed magnitudes
    hist.set_xlabel("Speed (m/s)")  #setting x-axis label
    hist.set_ylabel("Frequency Density")  #setting y-axis label
    
    #Computing the 2D Boltzmann distribution function
    E = total_Energy(particle_list, frame)  #calculating total energy of the system at the current frame
    Average_E = E / len(particle_list)  #calculating average energy per particle
    k = 1.38064852e-23  # Boltzmann constant
    T = 2 * Average_E / (3 * k)  #calculating temperature of the system
    m = particle_list[0].mass  #mass of particles
    v = np.linspace(0, 12, 120)  #creating an array of speeds (x-axis limit basically)
    fv =  4 * np.pi * v**2 * ((m) / (2 * np.pi * k * T))**(3/2) * np.exp((-m * v**2) / (2 * T * k))  #calculating the Maxwell-Boltzmann distribution
    hist.plot(v, fv, color='blue', linewidth=2, label="Maxwell–Boltzmann distribution")  
    hist.legend(loc="upper right")  
    hist.set_title(f"Frame: {frame+1}/{stepnumber}") 
    hist.set_ylim([0, 0.5])  #locking the right plot at this limit for visibility
    hist.grid(True)  #adding grid to the right subplot
    
#Creating and saving the animation
ani = FuncAnimation(fig, update, frames=stepnumber, repeat=False)
ani.save('particle_simulation.gif', writer='pillow', fps=30)
plt.show()
