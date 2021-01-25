import test
import bisect
import time
from typing import List

from tqdm import tqdm
from queue import Queue

#
#
# def log_fun_name(f):
#     def wrapper(a, b):
#         print(f.__name__)
#         f(a)
#         print(b)
#         return a + b
#     return wrapper
#
# @log_fun_name
# def foo(a):
#     print("hello" + str(a))
#
# c = foo(1, 2)
# print(c)

#
# class Test(object):
#
#     def __init__(self, filename, mode):
#         self.filename = filename
#         self.mode = mode
#         print("in init method")
#
#     def __enter__(self):
#         self.f = open(self.filename, mode=self.mode)
#         print("in enter")
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         self.f.close()
#         print("in exit")
#
# with Test('record.txt', 'r') as t:
#     pass

# from collections import OrderedDict
#
# class Test(object):
#     def __init__(self, name):
#         self.name = name
#
#
# a = OrderedDict()
# a.setdefault('wang', Test("wang"))
#
# a.setdefault('wang', Test("li"))
#
# print(a['wang'].name)

# 2021-01-12 14:35:05 | INFO | train_inner | {"epoch": 3, "update": 2.174, "loss": "9.368", "nll_loss": "8.846",
# "ppl": "460.11", "wps": "14185.4", "ups": "0.5", "wpb": "28201.6", "bsz": "1117.8", "num_updates": "300",
# "lr": "3.75925e-05", "gnorm": "1.724", "train_wall": "93", "wall": "0"}

# 2021-01-12 14:37:14 | INFO | valid | {"epoch": 3, "valid_loss": "8.751", "valid_nll_loss": "8.101",
# "valid_ppl": "274.56", "valid_wps": "202259", "valid_wpb": "11163.9", "valid_bsz": "455.2", "valid_num_updates": "414",
# "valid_best_loss": "8.751"}


# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next


# class Solution:
#     def lengthOfLongestSubstring(self, s: str) -> int:
#         max_len_dict = {}
#         for i in reversed(range(1, len(s) + 1)):
#             for j in range(len(s) - i):
#                 if len(s[j: j + i]) == len(set(s[j: j + i])):
#                     print(len(s[j: j + i]))
#                     print(s[j: j + i])
#                     return None
#
#                     # max_len_dict[s[j: j + i]] = len(s[j: j + i])
#
#
#
# s = "pwwkew"
# solu = Solution()
# solu.lengthOfLongestSubstring(s)

# def multiply(num):
#     if len(num) == 1:
#         return 1
#     elif num[0] == num[-1]:
#         return 1 * multiply(num[1: -1])
#     else:
#         return 0
#
# num = 4123214
#
# a = [int(i) for i in str(num)]
# b = multiply(a)
#
# print(b)

# def combine(elements, k):
#     if k == 1:
#         return elements
#     else:
#         return [i + j for j in elements for i in combine(elements, k-1)]
#
# ele = ['a', 'b', 'c']
# k = 3
# test = combine(ele, k)
# print(test)
# print(len(test))

# count = 0
# result = []


# def find(s, window_size):
#     global count
#     if window_size == 1:
#         for i in s:
#             result.append(i)
#             count += 1
#     else:
#         for i in range(len(s) - window_size + 1):
#             if s[i] == s[i + window_size - 1]:
#                 result.append(s[i: i + window_size])
#                 count += 1
#         find(s, window_size - 1)
#
# S = "abcab"
#
# find(S, len(S))
# print(count)
# print(result)


class Node(object):

    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

    def insert(self, value):
        if value < self.val:
            if self.left is None:
                self.left = Node(value)
            else:
                self.left.insert(value)
        elif value > self.val:
            if self.right is None:
                self.right = Node(value)
            else:
                self.right.insert(value)

    def inorder(self):
        if self.left is not None:
            self.left.inorder()
        print(self.val)
        if self.right is not None:
            self.right.inorder()

    def bfs(self, q):
        if self.left is not None:
            q.put(self.left)
        if self.right is not None:
            q.put(self.right)
        ele = q.get()
        print(ele.val)
        ele.bfs(q)



a = [2, 4, 6, 3, 13, 7, 5]

root = Node(8)

for i in a:
    root.insert(i)

q = Queue()

root.bfs(q)

