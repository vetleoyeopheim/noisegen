import numpy as np
import noise
import math
from scipy.ndimage import sobel
from scipy.ndimage import gaussian_filter as gaussf
from numba import njit

class NoiseMap:
    """
    Master class for noise maps
    Used for generating different types of coherent and random gradient noise
    """
    def __init__(self, x_range, y_range):
        self.x_range = x_range
        self.y_range = y_range
        

    def gen_simplex_map(self,scale,octa, pers, lac, seed):
        """
        Generates a map with Simplex noise.
        Uses the following module: https://github.com/caseman/noise

        """
        h_map = np.zeros((self.height, self.length))
        
        for i in range(self.height):
            for j in range(self.length):
                coords = (i,j)
                z = noise.snoise2(coords[0]/scale, coords[1]/scale, octaves=octa, \
                persistence = pers, lacunarity=lac, repeatx = self.height, repeaty = self.length, base= seed)
                h_map[i][j] = z
        
        h_map = self.normalize_array(h_map)
        
        return h_map


 
    def gen_smoothnoise_map(self, sigma):
        """
        Generate a noisemap from random noise that is then smoothed with gaussian blur
        """

        #Generate a noisemap with random noise ranging from 0 to 1
        n_map = self.gen_white_noise()
        #Filter with gaussian blur
        n_map = gaussf(n_map, sigma = sigma)
        return n_map

    def gen_twostate_noisemap(self, sigma, prob):
        """
        Generate a noise map based on two states, ocean(0) and land(1)
        The algorithm starts with a 2d array of 0
        Then the state of each element is determined by the state of previous elements in the x and y direction, and given probabilities
        Finally the map is put through a gaussian blur filter with a given sigma
        prob is a list of 4 probabilities for the state being 0 for 4 different cases of neighbouring elements : [prob_x0_y0, prob_x1_y0, prob_x0_y1, prob_x1_y1]
        
        """
        n_map = np.zeros((self.height, self.length))
        
        #Call a function using numba to cut down on computation time
        n_map = twostate_loop(n_map, self.height, self.length, prob)

        n_map = gaussf(n_map, sigma = sigma)
        return n_map



    
    def gen_meshgrid(self):
        i = np.arange(0,self.height)
        j = np.arange(0,self.length)
        grid = np.meshgrid(j,i)
        return grid

    def sobel_filter(self, nmap):
        """
        Calculate a sobel filter for the noise map and add it back to the noise map
        """
        sob_filter = sobel(nmap)
        sob_filter = self.normalize_array(sob_filter)
        return(nmap + 0.3 * sob_filter)

    #Exponential transform for noise function
    def exp_transform(self,nmap):
        nmap = np.exp(self.exp_factor * nmap)
        return nmap
    
    def sin_transform(self, nmap):
        nmap = np.sin(nmap/math.pi)
        return nmap

    def arctan_transform(self, nmap, factor = 15):
        nmap = np.arctan(factor * nmap/math.pi)
        return nmap
    
    def normalize_array(self, array):
        array_norm = (array - array.min())/(array.max() - array.min())
        return array_norm


class VoronoiNoise(NoiseMap):

    def __init__(self, x_range, y_range, n_points):
        """
        n_points is the number of randomly distributed measuring points in the Voronoi noise map
        """
        super().__init__(x_range, y_range)
        self.n_points = n_points

    def gen_voronoi_map(self):
        """
        Generate a Voronoi (Worley) noise map
        n_points is the number of randomly distributed points on the map that the voronoi cells are defined by
        """

        loc_coords = []

        for n in range(self.n_points):
            x_loc = np.random.randint(0,self.height)
            y_loc = np.random.randint(0,self.length)
            pnt = np.array((x_loc,y_loc))
            loc_coords.append(pnt)

        x = self.height
        y = self.length

        voronoi_map = voronoi_distances(loc_coords, self.n_points, x, y)

        #Normalize map
        voronoi_map = self.normalize_array(voronoi_map)

        return voronoi_map

class WhiteNoise(NoiseMap):

    def __init__(self, x_range, y_range):
        super().__init__(x_range, y_range)

    def gen_white_noise(self):
        """
        Generates a map with white noise
        """
        return(np.random.random(size = (self.height, self.length)))

class LatMap():
    """
    Latitude map takes height and length to calculate an array with a distance to the equator
    """
    def __init__(self, height, length):

        self.height = height
        self.length = length

    def gen_lat_map(self, symmetric = True, invert = True):
        """
        Generates a latitude map with values that are smallest towards the pole and smallest close to the equator
        Range of values are 0-1
        Generates a latitude map that is symmetric about the equator by default.
        If not, distance will be relative to the other pole. That is useful for temperature calculations due to inverse season between south and north
        
        """
        i = np.arange(0,self.height)
        j = np.arange(0,self.length)
        lat_map = np.meshgrid(j,i)
        lat_map = lat_map[1]
        if symmetric is True:
            lat_map = abs(lat_map - self.height/2)
        else:
            lat_map = abs(lat_map - self.height)
        
        if invert:
            lat_map = self.invert_map(lat_map)

        lat_map = (lat_map - lat_map.min())/(lat_map.max() - lat_map.min())     #Normalize map to the 0-1 range
        return lat_map

    def invert_map(self, amap):
        amap = (amap - 1.0) * (-1.0)          #This inverts the latitude map so that the equator equals 1 and the poles equals zero
        return amap


@njit
def twostate_loop(init_map, nx, ny, prob):

    n_map = init_map

    for i in range(1,nx,1):
        for j in range(1,ny,1):
            if n_map[i - 1][j] == 0 and n_map[i][j - 1] == 0:
                prob_zero = prob[0]       #prob_zero is the probability that the element will be in state zero
            elif n_map[i - 1][j] == 1 and n_map[i][j - 1] == 0:
                prob_zero = prob[1]
            elif n_map[i - 1][j] == 0 and n_map[i][j - 1] == 1:
                prob_zero = prob[2]
            elif n_map[i - 1][j] == 1 and n_map[i][j - 1] == 1:
                prob_zero = prob[3]

            rand_draw = np.random.random()
            if prob_zero > rand_draw:
                n_map[i][j] = 0
            else:
                n_map[i][j] = 1

    return n_map

@njit
def voronoi_distances(loc_coords, n_points, x, y):

    point_distances = np.zeros((n_points))
    dist_arr = np.zeros((x,y))
    for i in range(len(dist_arr)):
        for j in range(len(dist_arr[0])):
            for k in range(len(loc_coords)):
                point = loc_coords[k]
                dist = np.sqrt((point[0] - i)**2 + (point[1] - j)**2)
                point_distances[k] = dist

            min_dist_ind = np.argmin(point_distances)
            dist_arr[i][j] = point_distances[min_dist_ind]

    return dist_arr
