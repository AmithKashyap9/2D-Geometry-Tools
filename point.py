import numpy as np
import matplotlib.pyplot as plt

class HC():
	"""
	Homogeneous coordinates class for representing points and lines.
	"""

	def __init__(self, points, verbose=False):
		"""
		Initializes the HC class.

		Args:
			points (list): List of points in either pixel or homogeneous coordinates.
			verbose (bool, optional): Whether to print verbose output. Defaults to False.
		"""

		self.HC = None
		self.pixel = None
		self.verbose = verbose

		assert all(len(point) == len(points[0]) for point in points), "All points in the input list must have equal length"
		assert len(points[0]) in [2,3], "The points should be of length 2 or 3, if 2 its assumed as pixel representation, else it is treated as HC representation"

		if len(points[0]) == 2:
			points = np.array(points, dtype=float).T.reshape(2, -1)
			self.HC = np.vstack([points, np.ones((1, points.shape[1]))])
			self.pixel = points
		else:
			points = np.array(points, dtype=float).T.reshape(3, -1)
			self.HC = points
			self.pixel = np.divide(points[:2, :], points[2:, :], out=np.ones(points[:2, :].shape) * np.inf, where=points[2:, :] != 0)

		if verbose:
			print("----------------------------------")
			print("Printing Verbose output after initialization. (Set verbose to False to stop printing)")
			print("Input in Pixels: ")
			print(self.pixel)
			print("Input in HC: ")
			print(self.HC)
			print("----------------------------------")

	def normalize(self):
		"""
		Normalizes the homogeneous coordinates.
		"""

		self.HC = np.divide(self.HC, self.HC[2, :], out=np.ones_like(self.HC) * np.inf, where=self.HC[2, :] != 0)
		self.pixel = np.divide(self.HC[:2, :], self.HC[2, :], out=np.ones_like(self.HC[:2, :]) * np.inf, where=self.HC[2, :] != 0)

		if self.verbose:
			print("----------------------------------")
			print("Printing Verbose output after normalization. (Set verbose to False to stop printing)")
			print("Input in Pixels: ")
			print(self.pixel)
			print("Input in HC: ")
			print(self.HC)
			print("----------------------------------")

	def __len__(self):
		"""
		Returns the number of points or lines represented.
		"""
		return self.HC.shape[1]

class Point(HC):
	"""
	Class for representing points.
	"""

	def __init__(self, points, verbose=False):
		"""
		Initializes the Point class.

		Args:
			points (list): List of points in either pixel or homogeneous coordinates.
			verbose (bool, optional): Whether to print verbose output. Defaults to False.
		"""
		super().__init__(points, verbose)

	def line_eqn(self, point_B):
		"""
		Calculates the line equation passing through two points.

		Args:
			point_B (Point): Another point to form the line.

		Returns:
			Line: Line passing through the two points.
		"""
		assert len(point_B) == 1, "len of point should be 1, multi-point function is not yet implemented :/"
		line = np.cross(self.HC.T, point_B.HC.T)
		return Line(line)

	def centroid(self):
		"""
		Calculates and returns the centroid of the points.
		"""
		return np.mean(self.pixel, axis=1)

	def compute_orthogonal_distance(self, lines):
		"""
		Computes the orthogonal distance between points and lines.

		Args:
			lines (Line): Line objects.

		Returns:
			array: Array containing orthogonal distances.
		"""
		num_points = self.HC.shape[1]
		num_lines = lines.HC.shape[1]

		distances = np.zeros((num_points, num_lines))

		for i in range(num_points):
			point = self.HC[:, i].reshape(3, 1)
			for j in range(num_lines):
				line = lines.HC[:, j].reshape(3, 1)
				numerator = np.abs(np.dot(point.T, line))
				denominator = np.sqrt(np.sum(line[:2] ** 2))
				distances[i, j] = numerator / denominator

		return distances

	def distance_to_points(self, other_points):
		"""
		Computes the distance between the current point and a list of other points.

		Args:
			other_points (list): List of Point objects.

		Returns:
			numpy.ndarray: Array containing distances from the current point to each point in other_points.
		"""
		distances = []
		for point in other_points:
			distance = np.linalg.norm(self.pixel - point.pixel)
			distances.append(distance)
		return np.array(distances)

	def plot(self):
		"""
		Plots the points.
		"""
		plt.scatter(self.pixel[0], self.pixel[1], label="Point")
		plt.xlabel('X')
		plt.ylabel('Y')
		plt.title('Plot of Point')
		plt.legend()
		plt.grid(True)
		plt.show()

class Line(HC):
	"""
	Class for representing lines.
	"""

	def __init__(self, points, verbose=False):
		"""
		Initializes the Line class.

		Args:
			points (list): List of points representing lines in either pixel or homogeneous coordinates.
			verbose (bool, optional): Whether to print verbose output. Defaults to False.
		"""
		super().__init__(points, verbose)

	def line_intersection(self, line_B):
		"""
		Computes the intersection point of two lines.

		Args:
			line_B (Line): Another line object.

		Returns:
			Point: Intersection point.
		"""
		assert len(line_B) ==1, "len of line should be 1, multi-line function is not yet implemented :/"
		point = np.cross(self.HC.T, line_B.HC.T)
		return Point(point)

	def compute_orthogonal_distance(self, points):
		"""
		Computes the orthogonal distance between lines and points.

		Args:
			points (Point): Point objects.

		Returns:
			array: Array containing orthogonal distances.
		"""
		num_lines = self.HC.shape[1]
		num_points = points.HC.shape[1]

		distances = np.zeros((num_lines, num_points))

		for i in range(num_lines):
			line = self.HC[:, i].reshape(3, 1)
			for j in range(num_points):
				point = points.HC[:, j].reshape(3, 1)
				numerator = np.abs(np.dot(point.T, line))
				denominator = np.sqrt(np.sum(line[:2] ** 2))
				distances[i, j] = numerator / denominator

		return distances



	def plot(self, x_range=(-10, 10), point_objects=None, point_colors=None):
		"""
		Plots the lines.

		Args:
			x_range (tuple, optional): Range for x values. Defaults to (-10, 10).
			point_objects (list, optional): List of Point objects to plot. Defaults to None.
			point_colors (list, optional): List of colors for each point. Defaults to None.
		"""
		plt.figure(figsize=(8, 6))  # Adjust figure size as needed

		for i in range(self.HC.shape[1]):
			line = self.HC[:, i]
			if line[1] == 0:  # If the denominator is zero (slope is infinity)
				plt.axvline(x=-line[0] / line[2], linestyle='--', label=f'Line {i+1}')  # Plot vertical line
			else:
				x_values = np.linspace(x_range[0], x_range[1], 100)
				y_values = (-line[0] / line[1]) * x_values - (line[2] / line[1])
				plt.plot(x_values, y_values, label=f'Line {i+1}')

		# Plot points if provided
		if point_objects:
			if point_colors:
				assert len(point_objects) == len(point_colors), "Number of point objects and colors must match"
				for i, point_obj in enumerate(point_objects):
					color = point_colors[i]
					plt.scatter(point_obj.pixel[0], point_obj.pixel[1], marker='X', label='Point', color=color)
			else:
				for point_obj in point_objects:
					plt.scatter(point_obj.pixel[0], point_obj.pixel[1], marker='X', label='Point', color='r')

		plt.xlabel('X')
		plt.ylabel('Y')
		plt.title('Plot of Line')
		plt.legend()
		plt.grid(True)
		plt.show()

	def slope(self):
		"""
		Calculates and returns the slope of the lines.
		"""
		slopes = []
		for i in range(self.HC.shape[1]):
			line = self.HC[:, i]
			if line[1] == 0:  # If the denominator is zero (slope is infinity)
				slopes.append(float('inf'))
			else:
				slopes.append(-line[0] / line[1])
		return slopes


class LineSegment(Line):
	"""
	Class for representing line segments.
	"""

	def __init__(self, point_A, point_B, verbose=False):
		"""
		Initializes the LineSegment class.

		Args:
			point_A (Point): Start point of the line segment.
			point_B (Point): End point of the line segment.
			verbose (bool, optional): Whether to print verbose output. Defaults to False.
		"""
		assert len(point_A) == 1 and len(point_B) == 1, "Both points should be single points"
		line = np.cross(point_A.HC.T, point_B.HC.T)
		super().__init__(line, verbose)
		self.point_A = point_A
		self.point_B = point_B

	def sample_points(self, interval, verbose=False):
		"""
		Samples points at equal intervals between point_A and point_B.

		Args:
			interval (float): Interval between sampled points.
			verbose (bool, optional): Whether to print verbose output. Defaults to False.

		Returns:
			list: List of sampled points represented as lists.
		"""
		assert len(self.point_A) == 1 and len(self.point_B) == 1, "Both points should be single points"

		# Calculate the total number of intervals
		num_intervals = int(np.linalg.norm(self.point_B.pixel - self.point_A.pixel) // interval)
		
		# Generate points at equal intervals
		points = []
		for i in range(num_intervals + 1):
			alpha = i / num_intervals
			new_point = self.point_A.pixel + alpha * (self.point_B.pixel - self.point_A.pixel)
			points.append(new_point.reshape(-1).tolist())

		return points

	def centroid(self):
		"""
		Calculates and returns the centroid of the line segment.
		"""
		centroid = (self.point_A.pixel + self.point_B.pixel) / 2
		return centroid

	def line_segment_at_angle(self, angle_degrees, point_A):
		"""
		Returns a new LineSegment that is at a given angle (measured clockwise) from the original line segment,
		pivoted at point_A.

		Args:
			angle_degrees (float): Angle in degrees measured clockwise from the original line segment.
			point_A (Point): The pivot point for the rotation.

		Returns:
			LineSegment: A new LineSegment object representing the line segment at the specified angle
			pivoted at point_A.
		"""
		assert len(point_A) == 1, "point_A should be a single point"
		radians = np.radians(angle_degrees)
		# Calculate the direction vector of the new line segment
		direction_vector = self.point_B.pixel - self.point_A.pixel
		# Calculate the rotated direction vector
		rotated_direction_vector = np.array([
			np.cos(radians) * direction_vector[0] - np.sin(radians) * direction_vector[1],
			np.sin(radians) * direction_vector[0] + np.cos(radians) * direction_vector[1]
		])
		# Calculate the end point of the new line segment
		point_B_rotated = point_A.pixel + rotated_direction_vector
		point_B_rotated = [point_B_rotated.reshape(-1).tolist()]
		return LineSegment(point_A, Point(point_B_rotated))

def plot_points(point_objects):
	"""
	Plots the given points.

	Args:
		point_objects (list): List of Point objects to plot.
	"""
	colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  
	plt.figure(figsize=(8, 6))  

	for i, point_obj in enumerate(point_objects):
		color = colors[i % len(colors)]  # Use modulo to cycle through colors if more points than colors
		plt.scatter(point_obj.pixel[0], point_obj.pixel[1], label=f'Point {i+1}', color=color)

	plt.xlabel('X')
	plt.ylabel('Y')
	plt.title('Plot of Points')
	plt.legend()
	plt.grid(True)
	plt.show()


if __name__ == "__main__":
	pass
