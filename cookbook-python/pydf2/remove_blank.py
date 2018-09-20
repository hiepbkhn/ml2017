'''
Created on Sep 20, 2018

@author: Administrator
'''

from PyPDF2 import PdfFileWriter, PdfFileReader

####
def is_blank(p):
    if len(p) != 5:
        return False
    if '/ExtGState' not in p['/Resources']:
        return False
    if len(p['/Resources']['/ExtGState']) > 1:
        return False
    #
    return True 

infile = PdfFileReader('C:/Users/Administrator/Desktop/wordpress-plugin-development-cookbook-2nd.pdf', 'rb')
output = PdfFileWriter()

# p = infile.getPage(0)
# print(p)
# print(p.getContents())
# 
# p = infile.getPage(1)
# print(p)
# print(type(p.getContents()))
# print(p.getContents())
# print('Resources = ', p['/Resources'])

# p = infile.getPage(2)
# print('Resources = ', p['/Resources'])
# print(p)
# print(p.getContents())
# 
# p = infile.getPage(3)
# print(p)
# print(p.getContents())

######## check blank pages
# for i in [0,1,3,4,5,67]:    # 1,2,4,5,6
# #     print(infile.getPage(i).getContents())
#     print(len(infile.getPage(i)), '\t', infile.getPage(i))
#     
# for i in [2,6,8,11,14]:    # 3,7,9,12,15
# #     print(infile.getPage(i).getContents())
#     print(len(infile.getPage(i)), '\t', infile.getPage(i))
    
# for i in range(infile.getNumPages()):
#     p = infile.getPage(i)
#     if to_string(p['/Resources']) == '':
#         print(i)     

########
# for i in range(infile.getNumPages()):
#     p = infile.getPage(i)
#     if p.getContents(): # getContents is None if  page is blank
#         output.addPage(p)
# 
# with open('newfile.pdf', 'wb') as f:
#     output.write(f)

########
for i in range(infile.getNumPages()):
    p = infile.getPage(i)
    if not is_blank(p):
        output.addPage(p)

with open('newfile.pdf', 'wb') as f:
    output.write(f)

