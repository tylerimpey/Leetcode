
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

    def isValid_20(self, s):
        """
        Description
        -----------
        Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid. An input string is valid if: Open brackets must be closed by the same type of brackets. Open brackets must be closed in the correct order. Note that an empty string is also considered valid.
        
        Parameters
        ----------
        :type s:    str
        :rtype:     bool
        """
        
        close = {'(': ')',
                 '{': '}',
                 '[': ']'}
        
        opening = ['(', '[', '{']
        
        queue = []
        for i in s:
            if i in opening:
                queue.append(close[i])
            if i not in opening:
                if i not in queue:
                    return False
                if i != queue[-1]:
                    return False
                if i == queue[-1]:
                    queue = queue[:-1]
                    
        if len(queue) == 0:
            return True

    def removeDuplicates_26(self, nums):
        """
        Description
        -----------
        Given a sorted array nums, remove the duplicates in-place such that each element appear only once and return the new length. Do not allocate extra space for another array, you must do this by modifying the input array in-place with O(1) extra memory.
        
        Parameters
        ----------
        :type nums:     List[int]
        :rtype:         int
        """
        
        if len(nums) <=1:
            return len(nums) 
        
        fast, slow = 0,0 
        
        count = 1
        while fast < len(nums):
            if nums[fast] != nums[slow]:
                count += 1 
                slow  += 1
                nums[slow] = nums[fast]
            fast += 1 
            
        return count

    def removeElement_27(self, nums, val):
        """
        Description
        -----------
        Given an array nums and a value val, remove all instances of that value in-place and return the new length. Do not allocate extra space for another array, you must do this by modifying the input array in-place with O(1) extra memory. The order of elements can be changed. It doesn't matter what you leave beyond the new length.
        
        Parameters
        ----------
        :type nums:     List[int]
        :type val:      int
        :rtype:         int
        """
        
        length = len(nums)
        i = 0
        while i < length:
            if nums[i] == val:
                nums[-1], nums[:-1] = nums[i], nums[:i]+nums[i+1:]
                length -= 1
            else:
                i += 1
                
        return i

    def strStr_28(self, haystack, needle):
        """
        Description
        -----------
        Return the index of the first occurrence of needle in haystack, or -1 if needle is not part of haystack.
        
        Assumptions
        -----------
        What should we return when needle is an empty string? This is a great question to ask during an interview. For the purpose of this problem, we will return 0 when needle is an empty string. This is consistent to C's strstr() and Java's indexOf().
        
        Parameters
        ---------
        :type haystack:     str
        :type needle:       str
        :rtype:             int
        """
        
        length = len(needle)
        
        if haystack == needle:
            return 0
        
        for i in range(len(haystack)-length+1):
            if haystack[i:i+length] == needle:
                return i
            
        return -1

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

    def maxSubArray_53(self, nums):
        """
        Description
        -----------
        Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.
        
        Parameters
        ----------
        :type nums:     List[int]
        :rtype:         int
        """
        
        current = 0
        result = nums[0]
        for i in nums:
            current += i
            result  = max(current,result)
            current = max(0,current)
        return result

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
        :type digits:   List[int]
        :rtype:         List[int]
        """
        
        number = ""
        for digit in digits:
            number += str(digit)
            
        result = [int(digit) for digit in str(int(number) + 1)]
        
        return result

    def addBinary_67(self, a, b):
        """
        Description
        -----------
        Given two binary strings, return their sum (also a binary string). The input strings are both non-empty and contains only characters 1 or 0.
        
        Parameters
        ----------
        :type a:    str
        :type b:    str
        :rtype:     str
        """
        
        result = list(str(int(a) + int(b)))[::-1]
        
        to_return = ''
        
        remainder = 0
        for i in range(len(result)):
            temp = int(result[i]) + remainder
            if temp >= 2:
                to_return += str(temp % 2)
                remainder = 1
            else:
                to_return += str(temp)
                remainder = 0
        
        if remainder == 1:
            to_return += '1'
        
        return to_return[::-1]

    def mySqrt_69(self, x):
        """
        Description
        -----------
        Compute and return the square root of x, where x is guaranteed to be a non-negative integer. Since the return type is an integer, the decimal digits are truncated and only the integer part of the result is returned.
        
        Parameters
        ----------
        :type x:    int
        :rtype:     int
        """
        
        if x == 0:
            return 0
        
        if x <= 3:
            return 1
        
        high = x
        mid = x / 2.
        low = 1
        
        while mid**2 < x or mid**2 > x + 0.01:
            if mid**2 > x:
                high = mid
                mid = (low + mid)/2.
            else:
                low = mid
                mid = (high + mid)/2.
                
        return int(mid)

    def climbStairs_70(self, n):
        """
        Description
        -----------
        You are climbing a stair case. It takes n steps to reach to the top. Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top? Note: Given n will be a positive integer.
        
        Parameters
        ----------
        :type n:    int
        :rtype:     int
        """
        
        fib = tmp = 1
        for _ in range(n):
            fib, tmp = tmp, fib + tmp
        
        return fib

    def generate_118(self, numRows):
        """
        Description
        -----------
        Given a non-negative integer numRows, generate the first numRows of Pascal's triangle.
        
        Parameters
        ----------
        :type numRows: int
        :rtype: List[List[int]]
        """
        
        # if numRows == 0:
        #     return []
        # elif numRows == 1:
        #     return [[1]]
        
        # temp = [[1]]
        
        # for _ in range(numRows-1):
        #     row = []
        #     for j in range(len(temp[-1])+1):
        #         if j == 0 or j == len(temp[-1]):
        #             row.append(1)
        #         else:
        #             row.append(temp[-1][j-1] + temp[-1][j])
        #     temp.append(row)
            
        # return temp

        pascal = [[1]*(i+1) for i in range(numRows)]
        for i in range(numRows):
            for j in range(1,i):
                pascal[i][j] = pascal[i-1][j-1] + pascal[i-1][j]
        return pascal

    def getRow_119(self, rowIndex):
        """
        Description
        -----------
        Given a non-negative index k where k ≤ 33, return the kth index row of the Pascal's triangle. Note that the row index starts from 0.
        
        Parameters
        ----------
        :type rowIndex: int
        :rtype:         List[int]
        """
        
        result = [1]
        for i in range(rowIndex):
            result.append(result[i]*(rowIndex-i)/(i+1))
        return result

    def maxProfit_121(self, prices):
        """
        Description
        -----------
        Say you have an array for which the ith element is the price of a given stock on day i. If you were only permitted to complete at most one transaction (i.e., buy one and sell one share of the stock), design an algorithm to find the maximum profit. Note that you cannot sell a stock before you buy one.
        
        Parameters
        ----------
        :type prices:   List[int]
        :rtype:         int
        """
        
        result, buy = 0, float('inf')
        for p in prices:
            result, buy = max(result, p-buy), min(buy, p)
        return result

    def isPalindrome_125(self, s):
        """
        Description
        -----------
        Given a string, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases. Note: For the purpose of this problem, we define empty string as valid palindrome.

        Parameters
        ----------
        :type s: str
        :rtype: bool
        """

        s = [l.lower() for l in s if l.isalnum()]
        return s == s[::-1]

    def singleNumber_136(self, nums):
        """
        Description
        -----------
        Given a non-empty array of integers, every element appears twice except for one. Find that single one. Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory?
        
        Parameters
        ----------
        :type nums:     List[int]
        :rtype:         int
        """
        
        return sum(list(set(nums)))*2 - sum(nums)

    def twoSum_167(self, numbers, target):
        """
        Description
        -----------
        Given an array of integers that is already sorted in ascending order, find two numbers such that they add up to a specific target number. The function twoSum should return indices of the two numbers such that they add up to the target, where index1 must be less than index2. Note: Your returned answers (both index1 and index2) are not zero-based. You may assume that each input would have exactly one solution and you may not use the same element twice.

        Parameters
        ----------
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """
        
        left = 0
        rite = len(numbers) - 1
        while rite > left:
            if numbers[left] + numbers[rite] < target:
                left += 1
            elif numbers[left] + numbers[rite] > target:
                rite -= 1
            elif numbers[left] + numbers[rite] == target:
                return [left + 1, rite + 1]
        else:
            return None

    def majorityElement_169(self, nums):
        """
        Description
        -----------
        Given an array of size n, find the majority element. The majority element is the element that appears more than ⌊ n/2 ⌋ times. You may assume that the array is non-empty and the majority element always exist in the array.
        
        Parameters
        ----------
        :type nums:     List[int]
        :rtype:         int
        """
        
        # unique = list(set(nums))
        # for number in unique:
        #     count = sum([1 if num == number else 0 for num in nums])
        #     if count > int(len(nums)/2):
        #         return number
        
        return sorted(nums)[len(nums)/2]

    def titleToNumber_171(self, s):
        """
        Description
        -----------
        Given a column title as appear in an Excel sheet, return its corresponding column number.
        
        Parameters
        ----------
        :type s:    str
        :rtype:     int
        """
        
        # result = 0
        # for i, j in enumerate(s[::-1]):
        #     result += (26**i)*(ord(j) - ord('A') + 1)
        # return result

        return sum([(26**i)*(ord(j)-ord('A')+1) for i,j in enumerate(s[::-1])])

    def trailingZeroes_172(self, n):
        """
        Description
        -----------
        Given an integer n, return the number of trailing zeroes in n!. Note: Your solution should be in logarithmic time complexity.
        
        Parameters
        ----------
        :type n:    int
        :rtype:     int
        """
        
        if n < 5:
            return 0
        x = 0
        while n != 0:
            n //= 5
            x += n
            
        return x

    def rotate_189(self, nums, k):
        """
        Description
        -----------
        Given an array, rotate the array to the right by k steps, where k is non-negative.
        
        Parameters 
        ----------
        :type nums: List[int]
        :type k: int
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        
        # n = len(nums)
        # if any((n == 1, k == 0, k == n)):
        #     return nums
        
        # r = k % n
        # nums[:r], nums[r:] = nums[-r:], nums[:-r]

        nums[:k], nums[k:] = nums[len(nums)-k:len(nums)], nums[:len(nums)-k]

    def reverseBits_190(self, n):
        """
        Description
        -----------
        Reverse bits of a given 32 bits unsigned integer.
        
        Parameters
        ----------
        :type n:    int
        :rtype:     int
        """
        
        # b_form = '{0:b}'.format(n)
        # b_form = '0'*(32-len(b_form)) + b_form
        
        # return int(b_form[::-1], 2)

        return int('{0:032b}'.format(n)[::-1],2)

    def hammingWeight_191(self, n):
        """
        Description
        -----------
        Write a function that takes an unsigned integer and return the number of '1' bits it has (also known as the Hamming weight).
        
        Parameters
        ----------
        :type n:    int
        :rtype:     int
        """

        # count = 0 
        # while n > 0:
        #     if n % 2:
        #         count += 1
        #     n >>= 1
        # return count
        
        return sum([int(digit) for digit in "{0:b}".format(n)])

    def isHappy_202(self, n):
        """
        Description
        -----------
        Write an algorithm to determine if a number is "happy". A happy number is a number defined by the following process: Starting with any positive integer, replace the number by the sum of the squares of its digits, and repeat the process until the number equals 1 (where it will stay), or it loops endlessly in a cycle which does not include 1. Those numbers for which this process ends in 1 are happy numbers.
        
        Parameters
        ----------
        :type n:    int
        :rtype:     bool
        """
        
        n = str(n)
        
        unique = [n]
        
        while n != '1':
            val = sum([int(l)**2 for l in n])
            n = str(val)
            if n in unique:
                return False
            unique.append(n)
        
        return True

    def containsDuplicate_217(self, nums):
        """
        Description
        -----------
        Given an array of integers, find if the array contains any duplicates. Your function should return true if any value appears at least twice in the array, and it should return false if every element is distinct.
        
        Parameters
        ----------
        :type nums:     List[int]
        :rtype:         bool
        """
        
        unique = list(set(nums))

        return len(unique) != len(nums)

    def isAnagram_242(self, s, t):
        """
        Description
        -----------
        Given two strings s and t , write a function to determine if t is an anagram of s. What if the inputs contain unicode characters? How would you adapt your solution to such case?
        
        Assumptions
        -----------
        You may assume the string contains only lowercase alphabets.
        
        Parameters
        ----------
        :type s:    str
        :type t:    str
        :rtype:     bool
        """
        
        return sorted([ord(l) for l in s]) == sorted([ord(l) for l in t])

    def canAttendMeetings_252(self, intervals):
        """
        Description
        -----------
        Given an array of meeting time intervals consisting of start and end times [[s1,e1],[s2,e2],...] (si < ei), determine if a person could attend all meetings.
        
        Parameters
        ----------
        :type intervals: List[List[int]]
        :rtype: bool
        """
        
        intervals.sort(key=lambda x: x[0])
        for i in range(1,len(intervals)):
            if intervals[i-1][1] > intervals[i][0]:
                return False
        return True

    def addDigits_258(self, num):
        """
        Description
        -----------
        Given a non-negative integer num, repeatedly add all its digits until the result has only one digit. Could you do it without any loop/recursion in O(1) runtime?
        
        Parameters
        ----------
        :type num:  int
        :rtype:     int
        """
        
        # num = str(num)
        # while len(num) != 1:
        #     num = str(sum([int(digit) for digit in num]))
            
        # return int(num)

        return num if num == 0 else num % 9 or 9

    def isUgly_263(self, num):
        """
        Description
        -----------
        Write a program to check whether a given number is an ugly number. Ugly numbers are positive numbers whose prime factors only include 2, 3, 5.
        
        Parameters
        ----------
        :type num:  int
        :rtype:     bool
        """
        
        for p in 2, 3, 5:
            while num % p == 0 < num:
                num /= p
        return num == 1

    def canPermutePalindrome_266(self, s):
        """
        Description
        -----------
        Given a string, determine if a permutation of the string could form a palindrome.
        
        Parameters
        ----------
        :type s:    str
        :rtype:     bool
        """
        
        dic = {}
        for item in s:
            dic[item] = dic.get(item, 0) + 1
        count1 = 0
        for val in dic.values():
            if val % 2 == 1:
                count1 += 1
            if count1 > 1:
                return False
        return True

    def missingNumber_268(self, nums):
        """
        Description
        -----------
        Given an array containing n distinct numbers taken from 0, 1, 2, ..., n, find the one that is missing from the array. Your algorithm should run in linear runtime complexity. Could you implement it using only constant extra space complexity?
        
        Parameters
        ----------
        :type nums: List[int]
        :rtype: int
        """
        
        n = len(nums)
        return n * (n+1) / 2 - sum(nums)

    def moveZeroes_283(self, nums):
        """
        Description
        -----------
        Given an array nums, write a function to move all 0's to the end of it while maintaining the relative order of the non-zero elements.
        
        Parameters
        ----------
        :type nums:     List[int]
        :rtype:         None Do not return anything, modify nums in-place instead.
        """
        
        # left = 0
        # rite = len(nums)
        
        # while left < rite:
        #     if nums[left] == 0:
        #         nums[-1], nums[:-1] = 0, nums[:left] + nums[left+1:]
        #         rite -= 1
        #     else:
        #         left += 1

        non_zero = [num for num in nums if num]
        zeros    = sum([1 if num == 0 else 0 for num in nums])
        
        result = non_zero + [0]*zeros
        
        for i in range(len(nums)):
            nums[i] = result[i] if result[i] else 0

    def wordPattern_290(self, pattern, str):
        """
        Description
        -----------
        Given a pattern and a string str, find if str follows the same pattern. Here follow means a full match, such that there is a bijection between a letter in pattern and a non-empty word in str.
        
        Parameters
        ----------
        :type pattern:  str
        :type str:      str
        :rtype:         bool
        """
        
        unique_p = []
        for i in pattern:
            if i not in unique_p:
                unique_p.append(i)
        
        order_p = [unique_p.index(letter) for letter in pattern]
        
        unique_s = []
        for i in str.split(' '):
            if i not in unique_s:
                unique_s.append(i)
        
        order_s = [unique_s.index(word) for word in str.split(' ')]
        
        return order_p == order_s

    def canWinNim_292(self, n):
        """
        Description
        -----------
        You are playing the following Nim Game with your friend: There is a heap of stones on the table, each time one of you take turns to remove 1 to 3 stones. The one who removes the last stone will be the winner. You will take the first turn to remove the stones. Both of you are very clever and have optimal strategies for the game. Write a function to determine whether you can win the game given the number of stones in the heap.
        
        Parameters
        ----------
        :type n:    int
        :rtype:     bool
        """
        
        # return n % 4 != 0

        return n>>2<<2 != n

    def generatePossibleNextMoves_293(self, s):
        """
        Description
        -----------
        You are playing the following Flip Game with your friend: Given a string that contains only these two characters: + and -, you and your friend take turns to flip two consecutive "++" into "--". The game ends when a person can no longer make a move and therefore the other person will be the winner. Write a function to compute all possible states of the string after one valid move.
        
        Parameters
        ----------
        :type s:    str
        :rtype:     List[str]
        """
        
        results = []
        
        for i in range(len(s)-1):
            if s[i] == '+' and s[i+1] == '+':
                results.append(s[:i] + '--' + s[i+2:])
                
        return results

    def isPowerOfThree_326(self, n):
        """
        Description
        -----------
        Given an integer, write a function to determine if it is a power of three.
        
        Parameters
        ----------
        :type n:    int
        :rtype:     bool
        """
        
        # if n <= 0:
        #     return False
        
        # while n > 1:
        #     n /= 3.

        # return int(n)

        return n > 0 and 1162261467 % n == 0

    def depthSum_339(self, nestedList):
        """
        Description
        -----------
        Given a nested list of integers, return the sum of all integers in the list weighted by their depth. Each element is either an integer, or a list -- whose elements may also be integers or other lists.
        
        Parameters
        ----------
        :type nestedList:   List[NestedInteger]
        :rtype:             int
        """
        
        def scanList(curr_list, depth):
            return sum(depth * x.getInteger() if x.isInteger() else scanList(x.getList(), depth + 1) for x in curr_list)
    
        return scanList(nestedList, 1)

    def isPowerOfFour_342(self, num):
        """
        Description
        -----------
        Given an integer (signed 32 bits), write a function to check whether it is a power of 4.
        
        Parameters
        ----------
        :type num:  int
        :rtype:     bool
        """
        
        return num > 0 and num & (num-1) == 0 and len(bin(num)[3:]) % 2 == 0

    def reverseString_344(self, s):
        """
        Description
        -----------
        Write a function that reverses a string. The input string is given as an array of characters char[]. Do not allocate extra space for another array, you must do this by modifying the input array in-place with O(1) extra memory.

        Assumptions
        -----------
        You may assume all the characters consist of printable ascii characters.
        
        Parameters
        ----------
        :type s:    List[str]
        :rtype:     None Do not return anything, modify s in-place instead.
        """
        
        left = 0
        rite = len(s)-1
        
        while left < rite:
            s[left], s[rite] = s[rite], s[left]
            left += 1
            rite -= 1

        # return s

    def reverseVowels_345(self, s):
        """
        Description
        -----------
        Write a function that takes a string as input and reverse only the vowels of a string.
        
        Parameters
        ----------
        :type s:    str
        :rtype:     str
        """
        
        vowels = set(['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'])
        
        s = list(s)
        
        left = 0
        rite = len(s)-1
        
        while left < rite:
            if s[left] in vowels and s[rite] in vowels:
                s[left], s[rite] = s[rite], s[left]
                left += 1
                rite -= 1
            elif s[left] in vowels:
                rite -= 1
            elif s[rite] in vowels:
                left += 1
            else:
                left += 1
                rite -= 1
                
        return ''.join(s)

    def intersection_349(self, nums1, nums2):
        """
        Description
        -----------
        Given two arrays, write a function to compute their intersection.
        
        Parameters
        ----------
        :type nums1:    List[int]
        :type nums2:    List[int]
        :rtype:         List[int]
        """
        
        # longer  = nums1 if len(nums1) > len(nums2) else nums2
        # shorter = nums2 if len(nums1) > len(nums2) else nums1
        
        # result = []
        
        # for i in shorter:
        #     if i in longer and i not in result:
        #         result.append(i)
                
        # return result

        return list(set(nums1) & set(nums2))

    def intersect_350(self, nums1, nums2):
        """
        Description
        -----------
        Given two arrays, write a function to compute their intersection. Note: Each element in the result should appear as many times as it shows in both arrays. The result can be in any order. Follow up: What if the given array is already sorted? How would you optimize your algorithm? What if nums1's size is small compared to nums2's size? Which algorithm is better? What if elements of nums2 are stored on disk, and the memory is limited such that you cannot load all elements into the memory at once?
        
        Parameters
        ----------
        :type nums1:    List[int]
        :type nums2:    List[int]
        :rtype:         List[int]
        """
        
        shortr = nums1 if len(nums1) < len(nums2) else nums2
        longer = nums2 if len(nums1) < len(nums2) else nums1
        
        result = []
        
        for num in shortr:
            if num in longer:
                result.append(num)
                longer.pop(longer.index(num))
                
        return result

    def firstUniqChar_387(self, s):
        """
        Description
        -----------
        Given a string, find the first non-repeating character in it and return it's index. If it doesn't exist, return -1. Note: You may assume the string contain only lowercase letters.
        
        Parameters
        ----------
        :type s:    str
        :rtype:     int
        """

        for i, j in enumerate(s):
            if j not in s[:i] + s[i+1:]:
                return i
            
        return -1

    def findTheDifference_389(self, s, t):
        """
        Description
        -----------
        Given two strings s and t which consist of only lowercase letters. String t is generated by random shuffling string s and then add one more letter at a random position. Find the letter that was added in t.
        
        Parameter
        ---------
        :type s:    str
        :type t:    str
        :rtype:     str
        """
        
        # sum_s = sum([ord(i) for i in s])
        # sum_t = sum([ord(i) for i in t])
        
        # return chr(sum_t - sum_s)

        while len(t)-len(s)==1 and len(s)>0:
            ss=s[0]
            s=s.replace(ss,'')
            t=t.replace(ss,'')

        return ss if len(t) == len(s) else t

    def fizzBuzz_412(self, n):
        """
        Description
        -----------
        Write a program that outputs the string representation of numbers from 1 to n. But for multiples of three it should output “Fizz” instead of the number and for the multiples of five output “Buzz”. For numbers which are multiples of both three and five output “FizzBuzz”.
        
        Parameters
        ----------
        :type n:    int
        :rtype:     List[str]
        """
        
        result = []
        
        for i in range(1, n+1):
            temp = ''
            if i % 3 == 0:
                temp += 'Fizz'
            if i % 5 == 0:
                temp += 'Buzz'
            if i % 5 != 0 and i % 3 != 0:
                temp = str(i)
            result.append(temp)
            
        return result

    def hammingDistance_461(self, x, y):
        """
        Description
        -----------
        The Hamming distance between two integers is the number of positions at which the corresponding bits are different. Given two integers x and y, calculate the Hamming distance.
        
        Parameters
        ----------
        :type x:    int
        :type y:    int
        :rtype:     int
        """

        # xor = x ^ y
        # count = 0
        # while xor > 0:
        #     if xor & 1:
        #         count += 1
        #     xor = xor >> 1
        # return count
        
        distance = 0
        
        x = '{0:b}'.format(x)
        y = '{0:b}'.format(y)
        
        x = '0' * (len(y)-len(x)) + x if len(y) > len(x) else x
        y = '0' * (len(x)-len(y)) + y if len(x) > len(y) else y
        
        for i, j in zip(x, y):
            if i != j:
                distance += 1
        
        return distance

    def licenseKeyFormatting_482(self, S, K):
        """
        Description
        -----------
        You are given a license key represented as a string S which consists only alphanumeric character and dashes. The string is separated into N+1 groups by N dashes. Given a number K, we would want to reformat the strings such that each group contains exactly K characters except for the first group which could be shorter than K, but still must contain at least one character. Furthermore, there must be a dash inserted between two groups and all lowercase letters should be converted to uppercase. Given a non-empty string S and a number K format the string according to the rules described above.
        
        Parameters
        ----------
        :type S:    str
        :type K:    int
        :rtype:     str
        """
        
        S = S.replace('-','')[::-1].upper()
        
        groups = int(len(S)/K)
        
        result = ''
        for i in range(groups):
            result += S[i*K:i*K+K] + '-'
            
        result += S[groups*K:]
        
        result = result[::-1]
        
        if len(result) == 0:
            return ''
        
        if result[0] == '-':
            result = result[1:]
        
        return result

    def findMaxConsecutiveOnes_485(self, nums):
        """
        Description
        -----------
        Given a binary array, find the maximum number of consecutive 1s in this array. Note: The input array will only contain 0 and 1. The length of input array is a positive integer and will not exceed 10,000.
        
        Parameters
        ----------
        :type nums:     List[int]
        :rtype:         int
        """
        
        result = 0
        i = 0
        while i < len(nums):
            temp = 0
            while i < len(nums) and nums[i] == 1:
                temp += 1
                i += 1
            if temp > result:
                result = temp
            i += 1
        
        return result

    def fib_509(self, N):
        """
        Description
        -----------
        The Fibonacci numbers, commonly denoted F(n) form a sequence, called the Fibonacci sequence, such that each number is the sum of the two preceding ones, starting from 0 and 1. Given N, calculate F(N).
        
        Parameters
        ----------
        :type N:    int
        :rtype:     int
        """
        
        if N <= 1:
            return N
        
        fib = [0, 1]
        
        for _ in range(N-1):
            fib[0], fib[1] = fib[1], sum(fib)
            
        return fib[1]

    def reverseWords_557(self, s):
        """
        Description
        -----------
        Given a string, you need to reverse the order of characters in each word within a sentence while still preserving whitespace and initial word order. Note: In the string, each word is separated by single space and there will not be any extra space in the string.
        
        Parameters
        ----------
        :type s:    str
        :rtype:     str
        """
        
        return ' '.join(w[::-1] for w in s.split())

    def arrayPairSum_561(self, nums):
        """
        Description
        -----------
        Given an array of 2n integers, your task is to group these integers into n pairs of integer, say (a1, b1), (a2, b2), ..., (an, bn) which makes sum of min(ai, bi) for all i from 1 to n as large as possible.
        
        Parameters
        ----------
        :type nums: List[int]
        :rtype: int
        """
        
        # nums = sorted(nums)
        
        # return sum([min([nums[i], nums[i+1]]) for i in range(0, len(nums), 2)])

        return sum(sorted(nums)[::2])

    def postorder_590(self, root):
        """
        Description
        -----------
        Given an n-ary tree, return the postorder traversal of its nodes' values. Recursive solution is trivial, could you do it iteratively?
        
        Parameters
        ----------
        :type root:     Node
        :rtype:         List[int]
        """
        
        # return [x for y in [self.postorder(child) for child in root.children] for x in y] + [root.val] if root else []

        return 0

    def judgeCircle_657(self, moves):
        """
        Description
        -----------
        There is a robot starting at position (0, 0), the origin, on a 2D plane. Given a sequence of its moves, judge if this robot ends up at (0, 0) after it completes its moves. The move sequence is represented by a string, and the character moves[i] represents its ith move. Valid moves are R (right), L (left), U (up), and D (down). If the robot returns to the origin after it finishes all of its moves, return true. Otherwise, return false. Note: The way that the robot is "facing" is irrelevant. "R" will always make the robot move to the right once, "L" will always make it move left, etc. Also, assume that the magnitude of the robot's movement is the same for each move.
        
        Parameters
        ----------
        :type moves:    str
        :rtype:         bool
        """
        
        directions = {"U": 1, "D": -1, "L": 1j, "R": -1j}
        
        # movement = sum([directions[i] for i in moves])

        movement = sum(map(directions.get, moves))
        
        return not movement

    def toLowerCase_709(self, str):
        """
        Description
        -----------
        Implement function ToLowerCase() that has a string parameter str, and returns the same string in lowercase.
        
        Parameters
        ----------
        :type str:  str
        :rtype:     str
        """
        
        result = ''
        
        for i in str:
            if ord(i) < 97 and ord(i) >= 65:
                result += chr(ord(i) + 32)
            else:
                result += i
                
        return result

    def selfDividingNumbers_728(self, left, right):
        """
        Description
        -----------
        A self-dividing number is a number that is divisible by every digit it contains. For example, 128 is a self-dividing number because 128 % 1 == 0, 128 % 2 == 0, and 128 % 8 == 0. Also, a self-dividing number is not allowed to contain the digit zero. Given a lower and upper number bound, output a list of every possible self dividing number, including the bounds if possible.
        
        Parameters
        ----------
        :type left:     int
        :type right:    int
        :rtype:         List[int]
        """
        
        result = []
        
        for number in range(left, right+1):
            str_form = str(number)
            temp = 0
            for l in str_form:
                if (int(l) != 0) and (number % int(l) == 0):
                    temp += 1
            if temp == len(str_form):
                result.append(number)
                
        return result

    def nextGreatestLetter_744(self, letters, target):
        """
        Description
        -----------
        Given a list of sorted characters letters containing only lowercase letters, and given a target letter target, find the smallest element in the list that is larger than the given target. Letters also wrap around. For example, if the target is target = 'z' and letters = ['a', 'b'], the answer is 'a'.
        
        Parameters
        ----------
        :type letters:  List[str]
        :type target:   str
        :rtype:         str
        """
        
        if len(letters) < 1 or target == None: 
            return None
        
        left = 0
        rite = len(letters)
        while left < rite:
            mid = (left + rite)//2
            if target >= letters[mid]:
                left = mid + 1
            else:
                rite = mid
                
        if left == len(letters): 
            return letters[0]
        else: 
            return letters[left]

    def anagramMappings_760(self, A, B):
        """
        Description
        -----------
        Given two lists Aand B, and B is an anagram of A. B is an anagram of A means B is made by randomizing the order of the elements in A. We want to find an index mapping P, from A to B. A mapping P[i] = j means the ith element in A appears in B at index j. These lists A and B may contain duplicates. If there are multiple answers, output any of them.
        
        Parameters
        ----------
        :type A:    List[int]
        :type B:    List[int]
        :rtype:     List[int]
        """
        
        result = []
        
        for i in A:
            result.append(B.index(i))
        
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

    def uniqueMorseRepresentations_804(self, words):
        """
        Description
        -----------
        International Morse Code defines a standard encoding where each letter is mapped to a series of dots and dashes, as follows: "a" maps to ".-", "b" maps to "-...", "c" maps to "-.-.", and so on. For convenience, the full table for the 26 letters of the English alphabet is given. Now, given a list of words, each word can be written as a concatenation of the Morse code of each letter. For example, "cba" can be written as "-.-..--...", (which is the concatenation "-.-." + "-..." + ".-"). We'll call such a concatenation, the transformation of a word. Return the number of different transformations among all words we have.
        
        Parameters
        ----------
        :type words:    List[str]
        :rtype:         int
        """
        
        code = [".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."]
        
        results = []
        
        for word in words:
            temp = ''.join([code[ord(l)-97] for l in word])
            results.append(temp)
        
        return len(set(results))

    def mostCommonWord_819(self, paragraph, banned):
        """
        Description
        -----------
        Given a paragraph and a list of banned words, return the most frequent word that is not in the list of banned words.  It is guaranteed there is at least one word that isn't banned, and that the answer is unique. Words in the list of banned words are given in lowercase, and free of punctuation.  Words in the paragraph are not case sensitive.  The answer is in lowercase.
        
        Parameters
        ----------
        :type paragraph: str
        :type banned: List[str]
        :rtype: str
        """
        
        bannedset = set(banned)
        counts = dict()
        
        paragraph = paragraph.replace("!", "").replace("?", "").replace("'", "").replace(", ", " ").replace(",", " ").replace(";", "").replace(".", "")
        for W in paragraph.split(" "):
            w = W.lower()
            if w not in bannedset:
                counts[w] = counts.get(w, 0) + 1
        
        return sorted(list(counts.items()), reverse=True, key=lambda x: x[1])[0][0]

    def flipAndInvertImage_832(self, A):
        """
        Description
        -----------
        Given a binary matrix A, we want to flip the image horizontally, then invert it, and return the resulting image. To flip an image horizontally means that each row of the image is reversed.  For example, flipping [1, 1, 0] horizontally results in [0, 1, 1]. To invert an image means that each 0 is replaced by 1, and each 1 is replaced by 0. For example, inverting [0, 1, 1] results in [1, 0, 0].
        
        Parameters
        ----------
        :type A:    List[List[int]]
        :rtype:     List[List[int]]
        """
        
        return [[abs(element - 1) for element in row] for row in [row[::-1] for row in A]]

    def peakIndexInMountainArray_852(self, A):
        """
        Description
        -----------
        Let's call an array A a mountain if the following properties hold: A.length >= 3 There exists some 0 < i < A.length - 1 such that A[0] < A[1] < ... A[i-1] < A[i] > A[i+1] > ... > A[A.length - 1] Given an array that is definitely a mountain, return any i such that A[0] < A[1] < ... A[i-1] < A[i] > A[i+1] > ... > A[A.length - 1].
        
        Parameters
        ----------
        :type A:    List[int]
        :rtype:     int
        """
        
        left = 0
        rite = len(A)
        mid  = len(A)/2
        
        while A[mid-1] >= A[mid] or A[mid+1] >= A[mid]:
            if A[mid-1] >= A[mid]:
                rite = mid
                mid  = (left + rite)/2
            if A[mid+1] >= A[mid]:
                left = mid
                mid  = (left + rite)/2
        
        return mid

    def sortArrayByParity_905(self, A):
        """
        Description
        -----------
        Given an array A of non-negative integers, return an array consisting of all the even elements of A, followed by all the odd elements of A. You may return any answer array that satisfies this condition. 1 <= A.length <= 5000 0 <= A[i] <= 5000
        
        Parameters
        ----------
        :type A: List[int]
        :rtype: List[int]
        """
        
        # left = 0
        # rite = len(A)
        
        # while left < rite:
        #     if A[left] % 2 == 0:
        #         left += 1
        #     else:
        #         A[-1], A[:-1] = A[left], A[:left] + A[left+1:]
        #         rite -= 1
        
        # return A

        # even = []
        # odd = []
        # for number in A:
        #     if number % 2 == 0:
        #         even.append(number)
        #     else:
        #         odd.append(number)
            
        # return even+odd

        return [num for num in A if num % 2 == 0] + [num for num in A if num % 2 == 1]

    def sortArrayByParityII_922(self, A):
        """
        Description
        -----------
        Given an array A of non-negative integers, half of the integers in A are odd, and half of the integers are even. Sort the array so that whenever A[i] is odd, i is odd; and whenever A[i] is even, i is even. You may return any answer array that satisfies this condition.
        
        Parameters
        ----------
        :type A:    List[int]
        :rtype:     List[int]
        """
        
        evn = 0
        odd = 1
        l = len(A)
        
        while evn < l and odd < l:
            if A[evn] % 2 == 0:
                evn += 2
            elif A[odd] % 2 == 1:
                odd += 2
            else: 
                A[evn], A[odd] = A[odd], A[evn]
                evn += 2
                odd += 2
        
        return A

    def numUniqueEmails_929(self, emails):
        """
        Description
        -----------
        Every email consists of a local name and a domain name, separated by the @ sign. For example, in alice@leetcode.com, alice is the local name, and leetcode.com is the domain name. Besides lowercase letters, these emails may contain '.'s or '+'s. If you add periods ('.' between some characters in the local name part of an email address, mail sent there will be forwarded to the same address without dots in the local name.  For example, "alice.z@leetcode.com" and "alicez@leetcode.com" forward to the same email address. (Note that this rule does not apply for domain names.) If you add a plus ('+') in the local name, everything after the first plus sign will be ignored. This allows certain emails to be filtered, for example m.y+name@email.com will be forwarded to my@email.com. (Again, this rule does not apply for domain names.) It is possible to use both of these rules at the same time. Given a list of emails, we send one email to each address in the list.  How many different addresses actually receive mails?

        Company
        -------
        Google
        
        Parameters
        ----------
        :type emails:   List[str]
        :rtype:         int
        """
        
        ends   = [email.split('@')[1] for email in emails]
        starts = [email.split('@')[0] for email in emails]
        
        starts = [email.split('+')[0] if '+' in email else email for email in starts]
        starts = [email.replace('.','') for email in starts]
        
        emails = list(set([starts[i] + '@' + ends[i] for i in range(len(emails))]))
        
        return len(emails)

    def validMountainArray_941(self, A):
        """
        Description
        -----------
        Given an array A of integers, return true if and only if it is a valid mountain array. Recall that A is a mountain array if and only if: A.length >= 3 There exists some i with 0 < i < A.length - 1 such that: A[0] < A[1] < ... A[i-1] < A[i] A[i] > A[i+1] > ... > A[B.length - 1]
        
        Parameters
        ----------
        :type A: List[int]
        :rtype: bool
        """
        
        left = 0
        rite = len(A)-1
        
        l_prev = left
        r_prev = rite
        while left < rite:
            l_prev = left
            r_prev = rite
            if A[left+1] > A[left]:
                left += 1
            if A[rite-1] > A[rite]:
                rite -= 1
            if l_prev == left and r_prev == rite:
                return False
            
        if rite == len(A)-1 or left == 0:
            return False
        
        return True

    def minDeletionSize_944(self, A):
        """
        Description
        -----------
        We are given an array A of N lowercase letter strings, all of the same length. Now, we may choose any set of deletion indices, and for each string, we delete all the characters in those indices. For example, if we have an array A = ["abcdef","uvwxyz"] and deletion indices {0, 2, 3}, then the final array after deletions is ["bef", "vyz"], and the remaining columns of A are ["b","v"], ["e","y"], and ["f","z"].  (Formally, the c-th column is [A[0][c], A[1][c], ..., A[A.length-1][c]].) Suppose we chose a set of deletion indices D such that after deletions, each remaining column in A is in non-decreasing sorted order. Return the minimum possible value of D.length.
        
        Parameters
        ----------
        :type A:    List[str]
        :rtype:     int
        """
        
        check = 0
        for i in range(len(A[0])):
            stack = [ord(x[i]) for x in A]
            if stack != sorted(stack):
                check += 1
        
        return check

    def repeatedNTimes_961(self, A):
        """
        Description
        -----------
        In a array A of size 2N, there are N+1 unique elements, and exactly one of these elements is repeated N times. Return the element repeated N times.

        Parameters
        ----------
        :type A:    List[int]
        :rtype:     int
        """
        
        unique = []
        
        for num in A:
            if num in unique:
                return num
            else:
                unique.append(num)

    def sortedSquares_977(self, A):
        """
        Description
        -----------
        Given an array of integers A sorted in non-decreasing order, return an array of the squares of each number, also in sorted non-decreasing order. 1 <= A.length <= 10000 -10000 <= A[i] <= 10000 A is sorted in non-decreasing order.
        
        Parameters
        ----------
        :type A: List[int]
        :rtype: List[int]
        """
        
        for i in range(len(A)):
            A[i] = A[i]**2
            
        return sorted(A)

    def bitwiseComplement_1009(self, N):
        """
        Description
        -----------
        Every non-negative integer N has a binary representation.  For example, 5 can be represented as "101" in binary, 11 as "1011" in binary, and so on.  Note that except for N = 0, there are no leading zeroes in any binary representation. The complement of a binary representation is the number in binary you get when changing every 1 to a 0 and 0 to a 1.  For example, the complement of "101" in binary is "010" in binary. For a given number N in base-10, return the complement of it's binary representation as a base-10 integer.
        
        Parameters
        ----------
        :type N:    int
        :rtype:     int
        """
            
        result = [str(abs(int(i)-1)) for i in '{0:b}'.format(N)]
            
        return int(''.join(result), 2)

    def removeOuterParentheses_1021(self, S):
        """
        :type S: str
        :rtype: str
        """
        
        result = ''
        queue  = 0
        for i in S:
            if i == ')':
                queue -= 1
            if queue > 0:
                result += i
            if i == '(':
                queue += 1
                
        return result

    def divisorGame_1025(self, N):
        """
        Description
        -----------
        Alice and Bob take turns playing a game, with Alice starting first. Initially, there is a number N on the chalkboard.  On each player's turn, that player makes a move consisting of: Choosing any x with 0 < x < N and N % x == 0. Replacing the number N on the chalkboard with N - x. Also, if a player cannot make a move, they lose the game. Return True if and only if Alice wins the game, assuming both players play optimally.
        
        Parameters
        ----------
        :type N:    int
        :rtype:     bool
        """
        
        return True if N % 2 == 0 else False


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

    def addTwoNumbers_2(self, l1, l2):
        """
        Description
        -----------
        You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list. You may assume the two numbers do not contain any leading zero, except the number 0 itself.
        
        Parameters
        ----------
        :type l1:   ListNode
        :type l2:   ListNode
        :rtype:     ListNode
        """
        
        # dummy = current = ListNode(0)
        # remainder = 0
        # while l1 or l2 or remainder:
        #     if l1:
        #         remainder += l1.val
        #         l1 = l1.next
        #     if l2:
        #         remainder += l2.val
        #         l2 = l2.next
        #     current.next = ListNode(remainder % 10)
        #     current = current.next
        #     remainder //= 10
        # return dummy.next

        return 0

    def lengthOfLongestSubstring_3(self, s):
        """
        Description
        -----------
        Given a string, find the length of the longest substring without repeating characters.
        
        Parameters
        ----------
        :type s: str
        :rtype: int
        """
        
        if len(s) == 0:
            return 0
        
        temp = s[0]
        result = 1
        for letter in s[1:]:
            if letter in temp:
                i = temp.find(letter)
                temp = temp[i+1:]
            temp += letter
            if len(temp) > result:
                result = len(temp)
        
        return result

    def longestPalindrome_5(self, s):
        """
        Description
        -----------
        Given a string s, find the longest palindromic substring in s. You may assume that the maximum length of s is 1000.
        
        Parameters
        ----------
        :type s:    str
        :rtype:     str
        """
        
        def expand(l, r):
            while l >= 0 and r < N and s[l] == s[r]:
                l, r = l-1, r+1
            return l+1, r
        
        N, l, r = len(s), 0, 0
        if N < 2 or s == s[::-1]: 
            return s
        
        for i in range(N):
            l, r = max(((l, r), expand(i,i), expand(i,i+1)), key=lambda x: x[1]-x[0])
        return s[l:r]

    def maxArea_11(self, height):
        """
        Description
        -----------
        Given n non-negative integers a1, a2, ..., an , where each represents a point at coordinate (i, ai). n vertical lines are drawn such that the two endpoints of line i is at (i, ai) and (i, 0). Find two lines, which together with x-axis forms a container, such that the container contains the most water. Note: You may not slant the container and n is at least 2.
        
        Parameters
        ----------
        :type height:   List[int]
        :rtype:         int
        """

        left = result = 0
        rite = width = len(height) - 1
        for w in range(width, 0, -1):
            if height[left] < height[rite]:
                result = max(result, height[left]*w)
                left += 1
            else:
                result = max(result, height[rite]*w)
                rite -= 1
        return result

    def intToRoman_12(self, num):
        """
        Description
        -----------
        For example, two is written as II in Roman numeral, just two one's added together. Twelve is written as, XII, which is simply X + II. The number twenty seven is written as XXVII, which is XX + V + II. Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not IIII. Instead, the number four is written as IV. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as IX. There are six instances where subtraction is used: I can be placed before V (5) and X (10) to make 4 and 9. X can be placed before L (50) and C (100) to make 40 and 90. C can be placed before D (500) and M (1000) to make 400 and 900. Given an integer, convert it to a roman numeral. Input is guaranteed to be within the range from 1 to 3999.
        
        Parameters
        ----------
        :type num:  int
        :rtype:     str
        """
        
        romans = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
        values = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
        result = ""
        for i, val in enumerate(values):
            result += (num//val) * romans[i]
            num %= val
        return result

    def threeSum_15(self, nums):
        """
        Description
        -----------
        Given an array nums of n integers, are there elements a, b, c in nums such that a + b + c = 0? Find all unique triplets in the array which gives the sum of zero. Note: The solution set must not contain duplicate triplets.
        
        Parameters
        ----------
        :type nums: List[int]
        :rtype:     List[List[int]]
        """
        
        result = []
        nums.sort()
        length = len(nums)
        for i in range(length-2):
            if nums[i] > 0:
                break
            if i > 0 and nums[i] == nums[i-1]:
                continue
            left, rite = i+1, length-1
            while left < rite:
                total = nums[i] + nums[left] + nums[rite]
                
                if total < 0:
                    left += 1
                elif total > 0:
                    rite -= 1
                else:
                    result.append([nums[i], nums[left], nums[rite]])
                    while left < rite and nums[left]==nums[left+1]:
                        left += 1
                    while left < rite and nums[rite]==nums[rite-1]:
                        rite -= 1
                    left += 1
                    rite -= 1
        return result

    def letterCombinations_17(self, digits):
        """
        Description
        -----------
        Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent. A mapping of digit to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.
        
        Parameters
        ----------
        :type digits: str
        :rtype: List[str]
        """
        
        n = len(digits)
        translate = ['abc', 'def', 'ghi', 'jkl', 'mno', 'pqrs', 'tuv', 'wxyz']
        
        if n == 1:
            return list(translate[int(digits)-2])
        elif n == 0:
            return []
        
        result = []
        for i in list(translate[int(digits[0])-2]):
            for j in self.letterCombinations_17(digits[1:]):
                result.append(i + j)

        return result

    def searchRange_34(self, nums, target):
        """
        Description
        -----------
        Given an array of integers nums sorted in ascending order, find the starting and ending position of a given target value. Your algorithm's runtime complexity must be in the order of O(log n). If the target is not found in the array, return [-1, -1].
        
        Parameters
        ----------
        :type nums:     List[int]
        :type target:   int
        :rtype:         List[int]
        """
        
        length = len(nums)
        if length == 0 or target < nums[0] or target > nums[-1]:
            return [-1, -1]
        
        left, rite = 0, length-1
        while left < rite:
            mid = (left + rite)/2
            if nums[mid] >= target:
                rite = mid
            else:
                left = mid + 1
        
        if nums[left] != target:
            return [-1, -1]
        
        k, rite = left, length-1
        while k+1 < rite:
            mid = (k + rite)/2
            if nums[mid] == target:
                k = mid
            else:
                rite = mid - 1
        
        return [left, k+1] if (k+1 < length) and (nums[k+1] == target) else [left, k]

    def multiply_43(self, num1, num2):
        """
        Description
        -----------
        Given two non-negative integers num1 and num2 represented as strings, return the product of num1 and num2, also represented as a string.
        
        Parameters
        ----------
        :type num1: str
        :type num2: str
        :rtype:     str
        """
        
        return str(int(num1) * int(num2))

    def permute_46(self, nums):
        """
        Description
        -----------
        Given a collection of distinct integers, return all possible permutations.
        
        Parameters
        ----------
        :type nums: List[int]
        :rtype:     List[List[int]]
        """
        
        n = len(nums)
        if n <= 1: 
            return [nums]
        
        result = []
        for i in range(n):  
            s = nums[:i] + nums[i+1:]
            p = self.permute_46(s)  
            for x in p:  
                result.append([nums[i]] + x)  
        return result

    def rotate_48(self, matrix):
        """
        Description
        -----------
        You are given an n x n 2D matrix representing an image. Rotate the image by 90 degrees (clockwise). Note: You have to rotate the image in-place, which means you have to modify the input 2D matrix directly. DO NOT allocate another 2D matrix and do the rotation.
        
        Parameters
        ----------
        :type matrix:   List[List[int]]
        :rtype:         None Do not return anything, modify matrix in-place instead.
        """
        
        n = len(matrix)
        matrix.reverse()
        for i in range(n):
            for j in range(i+1, n):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

    def merge_56(self, intervals):
        """
        Description
        -----------
        Given a collection of intervals, merge all overlapping intervals.
        
        Parameters
        ----------
        :type intervals:    List[List[int]]
        :rtype:             List[List[int]]
        """
        
        out = []
        for i in sorted(intervals, key=lambda i: i[0]):
            if out and i[0] <= out[-1][-1]:
                out[-1][-1] = max(out[-1][-1], i[-1])
            else:
                out += i,
        return out

    def setZeroes_73(self, matrix):
        """
        Description
        -----------
        Given a m x n matrix, if an element is 0, set its entire row and column to 0. Do it in-place.
        
        Parameters
        ----------
        :type matrix:   List[List[int]]
        :rtype:         None Do not return anything, modify matrix in-place instead.
        """
        
        m = len(matrix)
        n = len(matrix[0])
        
        rows = []
        cols = []
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 0:
                    rows += [i]
                    cols += [j]
                    
        for i in rows:
            matrix[i] = [0]*n
            
        for j in cols:
            for i in range(m):
                matrix[i][j] = 0

    def searchMatrix_74(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        
        if not matrix or target is None:
            return False

        rows, cols = len(matrix), len(matrix[0])
        left, rite = 0, rows*cols - 1
        
        while left <= rite:
            mid = (left + rite)/2
            num = matrix[mid/cols][mid%cols]

            if num == target:
                return True
            elif num < target:
                left = mid + 1
            else:
                rite = mid - 1
        
        return False


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

    def findMedianSortedArrays_4(self, nums1, nums2):
        """
        Description
        -----------
        There are two sorted arrays nums1 and nums2 of size m and n respectively. Find the median of the two sorted arrays. The overall run time complexity should be O(log (m+n)). You may assume nums1 and nums2 cannot be both empty.
        
        Parameters
        ----------
        :type nums1:    List[int]
        :type nums2:    List[int]
        :rtype:         float
        """
        
        whole = sorted(nums1 + nums2)
        
        n = len(whole)
        
        return (whole[n/2-1] + whole[n/2])/2. if n % 2 == 0 else whole[n/2]

    def firstMissingPositive_41(self, nums):
        """
        Description
        -----------
        Given an unsorted integer array, find the smallest missing positive integer.
        
        Parameters
        ----------
        :type nums: List[int]
        :rtype:     int
        """
        
        nums = sorted([num for num in nums if num > 0])
        
        if len(nums) <= 1:
            if nums == [] or nums[0] != 1:
                return 1
            else:
                return 2
        
        i = 1
        while i in nums:
            i += 1
            
        return i


class SQLSolutions(object):
    """
    Example
    -------
    sql = SQLSolutions

    Notes
    -----
    All answers are simply described by commented chunks.
    """

    def easy_bigCountries_595(self):

        # SELECT    name, population, area
        # FROM      World
        # WHERE     area > 3000000 OR population > 25000000

        return 0

    def easy_shortestDistance_613(self):

        # SELECT  min(abs(p1.x - p2.x)) AS shortest
        # FROM    point p1, point p2
        # WHERE   p1.x != p2.x

        return 0

    def easy_notBoringMovies_620(self):

        # SELECT    *
        # FROM      cinema
        # WHERE     MOD(id, 2) = 1 AND description != 'boring'
        # ORDER BY  rating DESC

        return 0

    def easy_swapSalary_627(self):

        # UPDATE  salary
        # SET     sex = CASE
        # WHEN    sex = 'm' THEN 'f' WHEN sex = 'f' THEN 'm'
        # END

        return 0

    def easy_combineTwoTables_175(self):

        # SELECT      FirstName, LastName, City, State
        # FROM        Person 
        # LEFT JOIN   Address USING (PersonId)

        return 0
