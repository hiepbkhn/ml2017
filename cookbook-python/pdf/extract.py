#!/usr/bin/env python

'''
usage:   extract.py <some.pdf>

Locates Form XObjects and Image XObjects within the PDF,
and creates a new PDF containing these -- one per page.

Resulting file will be named extract.<some.pdf>

'''

import sys
import os

from pdfrw import PdfReader, PdfWriter
from pdfrw.findobjs import page_per_xobj


# inpfn, = sys.argv[1:]
# inpfn = 'files/sig-alternate.pdf'
inpfn = 'C:/Users/Administrator/Desktop/wordpress-plugin-development-cookbook-2nd.pdf'

outfn = 'extract.' + os.path.basename(inpfn)
org_pages = PdfReader(inpfn).pages

print(len(org_pages))   # print number of pages

# print(org_pages[0])
# print(org_pages[1])
# print(org_pages[2])

bytestream = org_pages[0].Contents.stream 
print(bytestream)
# print(org_pages[1].Contents.stream)
# print(org_pages[2].Contents.stream)

# for page in org_pages:
#     print(page)

# pages = list(page_per_xobj(org_pages, margin=0.5*72))
# if not pages:
#     raise IndexError("No XObjects found")
# writer = PdfWriter(outfn)
# writer.addpages(pages)
# writer.write()
