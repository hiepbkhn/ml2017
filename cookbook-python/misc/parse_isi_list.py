'''
Created on Feb 27, 2017

@author: Administrator
'''

def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False
    

lines = open('SCI_doc.txt').read().splitlines()
numline = len(lines)
print "numline = " + str(numline)

for line in lines:
    if RepresentsInt(line):
        print line


    