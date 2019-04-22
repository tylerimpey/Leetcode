
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