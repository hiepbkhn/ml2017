'''
Created on Feb 27, 2017

@author: Administrator
'''

import slate

with open('How to Read a Paper.pdf') as f:
    doc = slate.PDF(f)
    doc[1]
    
    
    