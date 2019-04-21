
class EasySolutions(object):
	"""
	Example
	-------
	easy = EasySolutions()
	"""

	def __init__(self):
		self.methods = [func for func in dir(EasySolutions) if callable(getattr(EasySolutions, func)) and '__' not in func]
		self.methods.remove('run_problem')

	def run_problem(self, problem):
		problem = [func for func in self.methods if int(func.split('_')[1]) == problem][0]
		method = getattr(EasySolutions, problem)
		print(method.__doc__)

	def twoSum_1(self, nums, target):
		"""
		Description
		-----------
		Given an array of integers, return indices of the two numbers such that they add up to a specific target.

		Assumptions
		-----------
		You may assume that each input would have exactly one solution, and you may not use the same element twice.

		Parameters
		----------
		:type nums: 	List[int]
		:type target: 	int
		:rtype: 		List[int]

		Example
		-------
		easy.twoSum_1([2, 7, 11, 15], 9) = [0, 1]

		Results
		-------
		36 ms 	- 66.59%
		12.9 MB - 5.02%
		"""
		
		index = {}
		for i, x in enumerate(nums):
			if target - x in index:
				return [index[target - x], i]
			index[x] = i

	def reverse_7(self, x):
		"""
		Description
		-----------
		Given a 32-bit signed integer, reverse digits of an integer.

		Parameters
		----------
		:type x: 	int
		:rtype: 	int

		Example
		-------
		easy.reverse_7(-123) = -321

		Results
		-------
		24 ms 	- 100.00%
		11.5 MB - 5.56%
		"""
		
		# s = cmp(x, 0)
		# r = int(`s*x`[::-1])

		# return s*r * (r < 2**31)

		return 0

	def numJewelsInStones_771(self, J, S):
		"""
		Description
		-----------
		You're given strings J representing the types of stones that are jewels, and S representing the stones you have.  Each character in S is a type of stone you have.  You want to know how many of the stones you have are also jewels.

		Assumptions
		-----------
		The letters in J are guaranteed distinct, and all characters in J and S are letters. Letters are case sensitive, so "a" is considered a different type of stone from "A".
		S and J will consist of letters and have length at most 50.
		The characters in J are distinct.

		Parameters
		----------
		:type J: 	str
		:type S: 	str
		:rtype: 	int

		Example
		-------
		easy.numJewelsInStones_771("aA", "aAAbbbb") = 3

		Results
		-------
		24 ms 	- 55.19%
		11.9 MB - 5.25%
		"""

		jewels = 0
		for i in J:
		    jewels += sum([1 for j in S if j == i])

		return jewels

class MediumSolutions(object):
	"""
	Example
	-------
	medium = MediumSolutions()
	"""

	def __init__(self):
		self.methods = [func for func in dir(MediumSolutions) if callable(getattr(MediumSolutions, func)) and '__' not in func]
		self.methods.remove('run_problem')

	def run_problem(self, problem):
		problem = [func for func in self.methods if int(func.split('_')[1]) == problem][0]

		method = getattr(MediumSolutions, problem)

		print(method.__doc__)

class HardSolutions(object):
	"""
	Example
	-------
	hard = HardSolutions()
	"""

	def __init__(self):
		self.methods = [func for func in dir(HardSolutions) if callable(getattr(HardSolutions, func)) and '__' not in func]
		self.methods.remove('run_problem')

	def run_problem(self, problem):
		problem = [func for func in self.methods if int(func.split('_')[1]) == problem][0]

		method = getattr(HardSolutions, problem)

		print(method.__doc__)