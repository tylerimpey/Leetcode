
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

    def isPalindrome_9(self, x):
        """
        Description
        -----------
        Determine whether an integer is a palindrome. An integer is a palindrome when it reads the same backward as forward.

        Parameters
        ----------
        :type x: 	int
        :rtype: 	bool

        Example
        -------
        easy.isPalindrome_1(-121) = False

        Results
        -------
        80 ms 	- 97.28%
        11.7 MB - 5.35%
        """
        
        return str(x)[::-1] == str(x)

    def romanToInt_13(self, s):
        """
        Description
        -----------
        Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M. Given a roman numeral, convert it to an integer. Input is guaranteed to be within the range from 1 to 3999.

        Parameters
        ----------
        :type s:    str
        :rtype:     int

        Example
        -------
        easy.romanToInt_13(LVIII) = 58

        Results
        -------
        52 ms 	- 98.37%
        11.7 MB - 5.47%
        """
        
        translate = {'I': 1,
                     'V': 5,
                     'X': 10,
                     'L': 50,
                     'C': 100,
                     'D': 500,
                     'M': 1000}
    
        result   = 0
        previous = 0
        for i in s[::-1]:
            if translate[i] >= previous:
                result += translate[i]
                previous = translate[i]
            else:
                result -= translate[i]
        
        return result

    def longestCommonPrefix_14(self, strs):
        """
        Description
        -----------
        Write a function to find the longest common prefix string amongst an array of strings. If there is no common prefix, return an empty string "".

        Parameters
        ----------
        :type strs:     List[str]
        :rtype:         str

        Example
        -------
        easy.longestCommonPrefix_14(["flower","flow","flight"]) = "fl"

        Results
        -------
        20 ms 	- 98.37%
        12.0 MB - 5.47%
        """
        
        r = [len(set(c)) == 1 for c in zip(*strs)] + [0]
        return strs[0][:r.index(0)] if strs else ''

    def searchInsert_35(self, nums, target):
        """
        Description
        -----------
        Given a sorted array and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.

        Assumptions
        -----------
        You may assume no duplicates in the array.

        Parameters
        ----------
        :type nums:     List[int]
        :type target:   int
        :rtype:         int
        """
        
        for i in range(len(nums)):
            if nums[i] >= target:
                return i
            
        return len(nums)

    def lengthOfLastWord_58(self, s):
        """
        Description
        -----------
        Given a string s consists of upper/lower-case alphabets and empty space characters ' ', return the length of last word in the string. If the last word does not exist, return 0.

        Assumptions
        -----------
        A word is defined as a character sequence consists of non-space characters only.

        Parameters
        ----------
        :type s:    str
        :rtype:     int
        """
        
        s = s.strip()
        s = s.split(' ')
        
        return len(s[-1])

    def plusOne_66(self, digits):
        """
        Description
        -----------
        Given a non-empty array of digits representing a non-negative integer, plus one to the integer.

        Assumptions
        -----------
        The digits are stored such that the most significant digit is at the head of the list, and each element in the array contain a single digit. You may assume the integer does not contain any leading zero, except the number 0 itself.

        Parameters
        ----------
        :type digits: List[int]
        :rtype: List[int]
        """
        
        number = ""
        for digit in digits:
            number += str(digit)
            
        result = [int(digit) for digit in str(int(number) + 1)]
        
        return result

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