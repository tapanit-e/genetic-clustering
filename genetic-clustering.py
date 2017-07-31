import random
import math
import heapq
import itertools

class Chromosome:

	def __init__(self, min_val, max_val, dimensions, index, init_centroid = None):
		self.index = index
		if init_centroid is not None:
			self.centroid = init_centroid
		else:
			self.centroid = []
			for i in range(dimensions):
				self.centroid.append(random.uniform(min_val, max_val))
		
class Cluster:
	
	def __init__(self, num_clusters, dataset, population = 30, th = 0.8, iterations = 5000, init_centroid = None):
		self.num_clusters = num_clusters
		self.dataset = dataset
		self.population = population
		self.iterations = iterations
		self.th = th
		self.dimensions = len(dataset[0])
		min_val = dataset[0][0]
		max_val = dataset[0][0]
		for i in range(len(dataset)):
			for j in range(len(dataset[i])):
				if dataset[i][j] < min_val:
					min_val = dataset[i][j]
				elif dataset[i][j] > max_val:
					max_val = dataset[i][j]
		self.min = min_val
		self.max = max_val
		self.chromosomes = []
		for i in range(self.population):
			chromosome_list = []
			for j in range(self.num_clusters):
				chromosome_list.append(Chromosome(self.min, self.max, self.dimensions, j))
			self.chromosomes.append(chromosome_list)
		if init_centroid is not None:
			init_centroid_list = []
			for i in range(len(init_centroid)):
				init_centroid_list.append(Chromosome(self.min, self.max, self.dimensions, i, init_centroid = init_centroid[i]))
			for i in range(self.population):
			self.chromosomes[0] = init_centroid_list
		silhouette = -1
		while silhouette < self.th and self.iterations > 0:
			clusters = self.cluster()
			fittest_list = self.fittest(clusters)
			self.mate(fittest_list)
			self.mutate()
			silhouette = heapq.nlargest(1, fittest_list)[0]
			self.iterations -= 1
			print silhouette
		
	def euclidean(self, point_one, point_two):
		sum = 0
		for i in range(len(point_one)):
			try:
				sum += (point_one[i] - point_two.centroid[i]) ** 2
			except AttributeError:
				sum += (point_one[i] - point_two[i]) ** 2
		return math.sqrt(sum)
		
	def silhouette(self, mat):
		lowest = None
		for item in mat:
			for compare in mat:
				if item != compare and item is not None and compare is not None and len(item) > 1 and len(compare) > 1:
					sil = self.__silhouette(item, compare)
					if lowest is None or sil < lowest:
						lowest = sil
				else:
					sil = -1.0
		return lowest
		
		
	def __silhouette(self, vector_one, vector_two):
		first_total = 0
		div_first = 0
		second_total = 0
		div_second = 0
		for item in vector_one:
			for compare in vector_one:
				if item != compare:
					first_total += self.euclidean(item, compare)
					div_first += 1
			for compare in vector_two:
				second_total += self.euclidean(item, compare)
				div_second += 1
		first_total = first_total / div_first
		second_total = second_total / div_second
		return (second_total - first_total) / max(first_total, second_total)
		
	def cluster(self):
		clusters = []
		for item in self.chromosomes:
			clusters.append([None] * len(item))
			for dataset_item in self.dataset:
				min_dist, winner = None, None
				for chromosome in item:
					dist = self.euclidean(dataset_item, chromosome)
					if min_dist is None or dist < min_dist:
						min_dist = dist
						winner = chromosome
				if clusters[len(clusters) - 1][winner.index] is None:
					clusters[len(clusters) - 1][winner.index] = []
				clusters[len(clusters) - 1][winner.index].append(dataset_item)
		return clusters
			
	def fittest(self, clusters):
		silhouettes = []
		for cluster in clusters:
			silhouette = self.silhouette(cluster)
			silhouettes.append(silhouette)
		self.silhouettes = silhouettes
		return silhouettes
		
	def mate(self, fittest_list):
		for i in range(len(fittest_list) - 1):
			index = heapq.nlargest(i + 1, zip(fittest_list, itertools.count()))[-1][1]
			next_index = heapq.nlargest(i + 2, zip(fittest_list, itertools.count()))[-1][1]
			fittest_chromosome_list = self.chromosomes[index]
			next_fittest_chromosome_list = self.chromosomes[next_index]
			for j in range(len(self.chromosomes[next_index])):
				for k in range(len(self.chromosomes[next_index][j].centroid)):
					self.chromosomes[next_index][j].centroid[k] = next_fittest_chromosome_list[j].centroid[k] + ((next_fittest_chromosome_list[j].centroid[k] - fittest_chromosome_list[j].centroid[k]) * 0.1)
					
	def mutate(self):
		for i in range(1, len(self.silhouettes)):
			index = heapq.nlargest(i + 1, zip(self.silhouettes, itertools.count()))[-1][1]
			for j in range(len(self.chromosomes[index])):
				for k in range(len(self.chromosomes[index][j].centroid)):
					self.chromosomes[index][j].centroid[k] += random.uniform(self.min, self.max) / abs(self.max - self.min) if random.random() < 0.5 else -random.uniform(self.min, self.max) / abs(self.max - self.min)
		
if __name__ == '__main__':
	data = [[5.1,3.5,1.4,0.2],
[4.9,3.0,1.4,0.2],
[4.7,3.2,1.3,0.2],
[4.6,3.1,1.5,0.2],
[5.0,3.6,1.4,0.2],
[5.4,3.9,1.7,0.4],
[4.6,3.4,1.4,0.3],
[5.0,3.4,1.5,0.2],
[4.4,2.9,1.4,0.2],
[4.9,3.1,1.5,0.1],
[5.4,3.7,1.5,0.2],
[4.8,3.4,1.6,0.2],
[4.8,3.0,1.4,0.1],
[4.3,3.0,1.1,0.1],
[5.8,4.0,1.2,0.2],
[5.7,4.4,1.5,0.4],
[5.4,3.9,1.3,0.4],
[5.1,3.5,1.4,0.3],
[5.7,3.8,1.7,0.3],
[5.1,3.8,1.5,0.3],
[5.4,3.4,1.7,0.2],
[5.1,3.7,1.5,0.4],
[4.6,3.6,1.0,0.2],
[5.1,3.3,1.7,0.5],
[4.8,3.4,1.9,0.2],
[5.0,3.0,1.6,0.2],
[5.0,3.4,1.6,0.4],
[5.2,3.5,1.5,0.2],
[5.2,3.4,1.4,0.2],
[4.7,3.2,1.6,0.2],
[4.8,3.1,1.6,0.2],
[5.4,3.4,1.5,0.4],
[5.2,4.1,1.5,0.1],
[5.5,4.2,1.4,0.2],
[4.9,3.1,1.5,0.1],
[5.0,3.2,1.2,0.2],
[5.5,3.5,1.3,0.2],
[4.9,3.1,1.5,0.1],
[4.4,3.0,1.3,0.2],
[5.1,3.4,1.5,0.2],
[5.0,3.5,1.3,0.3],
[4.5,2.3,1.3,0.3],
[4.4,3.2,1.3,0.2],
[5.0,3.5,1.6,0.6],
[5.1,3.8,1.9,0.4],
[4.8,3.0,1.4,0.3],
[5.1,3.8,1.6,0.2],
[4.6,3.2,1.4,0.2],
[5.3,3.7,1.5,0.2],
[5.0,3.3,1.4,0.2],
[7.0,3.2,4.7,1.4],
[6.4,3.2,4.5,1.5],
[6.9,3.1,4.9,1.5],
[5.5,2.3,4.0,1.3],
[6.5,2.8,4.6,1.5],
[5.7,2.8,4.5,1.3],
[6.3,3.3,4.7,1.6],
[4.9,2.4,3.3,1.0],
[6.6,2.9,4.6,1.3],
[5.2,2.7,3.9,1.4],
[5.0,2.0,3.5,1.0],
[5.9,3.0,4.2,1.5],
[6.0,2.2,4.0,1.0],
[6.1,2.9,4.7,1.4],
[5.6,2.9,3.6,1.3],
[6.7,3.1,4.4,1.4],
[5.6,3.0,4.5,1.5],
[5.8,2.7,4.1,1.0],
[6.2,2.2,4.5,1.5],
[5.6,2.5,3.9,1.1],
[5.9,3.2,4.8,1.8],
[6.1,2.8,4.0,1.3],
[6.3,2.5,4.9,1.5],
[6.1,2.8,4.7,1.2],
[6.4,2.9,4.3,1.3],
[6.6,3.0,4.4,1.4],
[6.8,2.8,4.8,1.4],
[6.7,3.0,5.0,1.7],
[6.0,2.9,4.5,1.5],
[5.7,2.6,3.5,1.0],
[5.5,2.4,3.8,1.1],
[5.5,2.4,3.7,1.0],
[5.8,2.7,3.9,1.2],
[6.0,2.7,5.1,1.6],
[5.4,3.0,4.5,1.5],
[6.0,3.4,4.5,1.6],
[6.7,3.1,4.7,1.5],
[6.3,2.3,4.4,1.3],
[5.6,3.0,4.1,1.3],
[5.5,2.5,4.0,1.3],
[5.5,2.6,4.4,1.2],
[6.1,3.0,4.6,1.4],
[5.8,2.6,4.0,1.2],
[5.0,2.3,3.3,1.0],
[5.6,2.7,4.2,1.3],
[5.7,3.0,4.2,1.2],
[5.7,2.9,4.2,1.3],
[6.2,2.9,4.3,1.3],
[5.1,2.5,3.0,1.1],
[5.7,2.8,4.1,1.3],
[6.3,3.3,6.0,2.5],
[5.8,2.7,5.1,1.9],
[7.1,3.0,5.9,2.1],
[6.3,2.9,5.6,1.8],
[6.5,3.0,5.8,2.2],
[7.6,3.0,6.6,2.1],
[4.9,2.5,4.5,1.7],
[7.3,2.9,6.3,1.8],
[6.7,2.5,5.8,1.8],
[7.2,3.6,6.1,2.5],
[6.5,3.2,5.1,2.0],
[6.4,2.7,5.3,1.9],
[6.8,3.0,5.5,2.1],
[5.7,2.5,5.0,2.0],
[5.8,2.8,5.1,2.4],
[6.4,3.2,5.3,2.3],
[6.5,3.0,5.5,1.8],
[7.7,3.8,6.7,2.2],
[7.7,2.6,6.9,2.3],
[6.0,2.2,5.0,1.5],
[6.9,3.2,5.7,2.3],
[5.6,2.8,4.9,2.0],
[7.7,2.8,6.7,2.0],
[6.3,2.7,4.9,1.8],
[6.7,3.3,5.7,2.1],
[7.2,3.2,6.0,1.8],
[6.2,2.8,4.8,1.8],
[6.1,3.0,4.9,1.8],
[6.4,2.8,5.6,2.1],
[7.2,3.0,5.8,1.6],
[7.4,2.8,6.1,1.9],
[7.9,3.8,6.4,2.0],
[6.4,2.8,5.6,2.2],
[6.3,2.8,5.1,1.5],
[6.1,2.6,5.6,1.4],
[7.7,3.0,6.1,2.3],
[6.3,3.4,5.6,2.4],
[6.4,3.1,5.5,1.8],
[6.0,3.0,4.8,1.8],
[6.9,3.1,5.4,2.1],
[6.7,3.1,5.6,2.4],
[6.9,3.1,5.1,2.3],
[5.8,2.7,5.1,1.9],
[6.8,3.2,5.9,2.3],
[6.7,3.3,5.7,2.5],
[6.7,3.0,5.2,2.3],
[6.3,2.5,5.0,1.9],
[6.5,3.0,5.2,2.0],
[6.2,3.4,5.4,2.3],
[5.9,3.0,5.1,1.8]]
	cluster = Cluster(3, data, init_centroid = [[ 6.853846153846153,
       3.0769230769230766,
       5.715384615384615,
       2.053846153846153 ],[ 5.88360655737705,
       2.740983606557377,
       4.388524590163935,
       1.4344262295081966 ],  [ 5.005999999999999,
       3.4180000000000006,
       1.464,
       0.2439999999999999 ]])
		
		