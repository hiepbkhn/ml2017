'''
Created on Mar 3, 2017

@author: Administrator
'''

######
def brute_force(t, p):
    n = len(t)
    m = len(p)
    print " n = ", n
    print " m = ", m
    for i in range(0, n-m+1):
        j = 0
        while (j < m) and (t[i+j] == p[j]):
            j = j + 1
        if (j == m):
            return i
    return -1

###### Boyer Moore algo
s = [' ']
s.extend(list(map(chr, range(97, 123))))
print s

def last_occur(p):
    l = [-1]*len(s)
    for i in range(len(p)-1,-1,-1):
        if (l[i] != -1):
            continue
        if (p[i] == ' '):
            l[0] = i;
        else:
            l[ord(p[i])-96] = i
    
    return l
    

def boyer_moore(t, p):
    l = last_occur(p)
    n = len(t)
    m = len(p)
    i = m-1
    j = m-1
    while (True):
        if (t[i] == p[j]):
            if j == 0:
                return i
            else:
                i = i - 1
                j = j - 1
        else:
            # character-jump
            last = l[ord(t[i])-96]
            i = i + m - min(j,1+last)
            j = m-1
        if (i > n-1):
            break

    return -1

###### KMP algo
def failure_func(p):
    m = len(p)
    f = [0]*m;
    i = 1
    j = 0
    while (i < m):
        if (p[i] == p[j]):
            # we have matched j + 1 chars
            f[i] = j + 1
            i = i + 1;
            j = j + 1;
        elif (j > 0):
            # use failure function to shift P
            j = f[j-1]
        else:
            f[i] = 0    # no match 
            i = i + 1;
    
    return f

def kmp(t, p):
    n = len(t)
    m = len(p)
    f = failure_func(p)
    print f
    i = 0
    j = 0
    while (i < n):
        if (t[i] == p[j]):
            if (j == m-1):
                return i-j
            else:
                i = i + 1;
                j = j + 1;
        else:
            if (j > 0):
                j = f[j-1]
            else: # j == 0
                i = i + 1
    return -1

#######################
t = 'a pattern matching algorithm'
p = 'g al'
# p = 'abaaba'  
  
# print brute_force(t, p)

# print last_occur(p)
print boyer_moore(t, p)

# print failure_func(p)
# print kmp(t, p)


