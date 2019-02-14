#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 18:09:49 2019

@author: karthikchowdary
"""

def uniquesubstring(input_string):

  last_occurrence = {}
  longest_length = 0
  longest_position = 0
  starting_position = 0
  current_length = 0


  for a, b in enumerate(input_string):
    l = last_occurrence.get(b, -1)
    
    if l < starting_position:
        current_length += 1
    else:
       
        if current_length > longest_length:
            longest_position = starting_position
            longest_length = current_length
        
        current_length -= l - starting_position
        starting_position = l + 1
    
    last_occurrence[b] = a
 
  if current_length > longest_length:
    longest_position = starting_position
    longest_length = current_length

  return input_string[longest_position:longest_position + longest_length]




input = 'karthik'

print(f"The Longest unique substring in '{input}' is '{uniquesubstring(input)}' Size: {len(uniquesubstring(input))}")